import torch
from transformers import BertModel
import torch.nn as nn
from configs import parse_arguments
from peft import get_peft_model, LoraConfig, TaskType


args = parse_arguments()
device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")  # type: ignore

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, num_layers=1, dropout=0.1):
        """
        Args:
            input_dim: int, kích thước đầu vào
            hidden_dim: int, kích thước lớp ẩn
            class_num: int, số lớp đầu ra
            num_layers: int, số lượng lớp ẩn (không tính lớp output)
            dropout: float, tỷ lệ dropout
        """
        super().__init__()
        layers = []

        # Lớp đầu tiên: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Các lớp ẩn tiếp theo: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Lớp output: hidden_dim -> class_num
        layers.append(nn.Linear(hidden_dim, class_num))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BertED(nn.Module):
    def __init__(self, class_num=args.class_num + 1, input_map=False):
        super().__init__()
        self.backbone = BertModel.from_pretrained(args.backbone)
        if not args.no_freeze_bert:
            print("Freeze bert parameters")
            for _, param in list(self.backbone.named_parameters()):
                param.requires_grad = False
        else:
            print("Update bert parameters")
            
        if args.freeze_embedding_layer:
            print("Freeze embedding layer")
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False
                
        if args.freeze_encoder_layers > 0:
            print(f"Freeze encoder layers from 0 to {args.freeze_encoder_layers}")
            for i in range(args.freeze_encoder_layers):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = False
            
                    
        self.is_input_mapping = input_map
        self.input_dim = self.backbone.config.hidden_size
        
        if args.classifier_layer > 1:
            self.hidden_dim = args.hidden_dim
            self.fc = Classifier(self.input_dim, self.hidden_dim, class_num, num_layers=args.classifier_layer, dropout=args.dropout)
            print(f"Classifier with {args.classifier_layer} layers")
        else:
            self.fc = nn.Linear(self.input_dim, class_num)
            
        if self.is_input_mapping:
            self.map_hidden_dim = 512 # 512 is implemented by the paper
            self.map_input_dim =  self.input_dim * 2
            self.input_map = nn.Sequential(
                nn.Linear(self.map_input_dim, self.map_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.map_hidden_dim, self.map_hidden_dim),
                nn.ReLU(),
            )
            self.fc = nn.Linear(self.map_hidden_dim, class_num)
            
        if args.use_lora:
            print("Apply LoRA to backbone encoder")
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["query", "value"],  # hoặc ["q_proj", "v_proj"] tùy mô hình
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.SEQ_CLS  # hoặc TaskType.TOKEN_CLS nếu cần
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            
        # MoLE: Mixture of LoRA Experts
        self.use_mole = getattr(args, 'use_mole', False)
        if self.use_mole:
            print("Using MoLE (Mixture of LoRA Experts)")
            self.n_experts = getattr(args, 'num_lora_experts', 4)
            self.top_k = getattr(args, 'top_k_experts', 2)
            self.experts = nn.ModuleList()
            self.expert_keys = nn.Parameter(torch.randn(self.n_experts, self.input_dim))  # Learnable keys

            for i in range(self.n_experts):
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["query", "value"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.SEQ_CLS
                )
                expert = get_peft_model(BertModel.from_pretrained(args.backbone), lora_config)
                if not args.no_freeze_bert:
                    for _, p in expert.named_parameters():
                        if 'lora' not in _:
                            p.requires_grad = False
                self.experts.append(expert)

            # Keep one frozen backbone for query gating
            self.backbone = BertModel.from_pretrained(args.backbone)
            for p in self.backbone.parameters():
                p.requires_grad = False

            
        print("Trainable parameters:")
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n, p.shape)
                
        def mole_backbone_forward(self, x, masks):
            with torch.no_grad():
                base_output = self.backbone(x, attention_mask=masks)
                cls_embedding = base_output.last_hidden_state[:, 0, :]  # shape (B, H)

            # Gating scores: (B, n_experts)
            sim_scores = torch.matmul(cls_embedding, self.expert_keys.T)
            topk_scores, topk_indices = torch.topk(sim_scores, self.top_k, dim=-1)  # shape (B, k)
            topk_weights = torch.softmax(topk_scores, dim=-1)  # shape (B, k)

            expert_outputs = []
            for i in range(self.top_k):
                expert_idx = topk_indices[:, i]  # shape (B,)
                grouped_batch = {}  # map from expert_id to list of indices
                for b, idx in enumerate(expert_idx):
                    idx = idx.item()
                    if idx not in grouped_batch:
                        grouped_batch[idx] = []
                    grouped_batch[idx].append(b)

                batch_outputs = torch.zeros_like(base_output.last_hidden_state)
                for expert_id, batch_indices in grouped_batch.items():
                    input_ids_subset = x[batch_indices]
                    mask_subset = masks[batch_indices]
                    expert = self.experts[expert_id]
                    output = expert(input_ids_subset, attention_mask=mask_subset).last_hidden_state
                    for i, b in enumerate(batch_indices):
                        batch_outputs[b] = output[i]

                weighted_output = batch_outputs * topk_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # shape (B, S, H)
                expert_outputs.append(weighted_output)

            mixed_output = torch.stack(expert_outputs).sum(0)  # shape (B, S, H)
            pooled_output = mixed_output[:, 0, :]  # optional for downstream use

            return mixed_output, pooled_output

    def forward(self, x, masks, span=None, aug=None):
        # x = self.backbone(x) #TODO: test use
        return_dict = {}
        # backbone_output = self.backbone(x, attention_mask = masks)
        # x, pooled_feat = backbone_output[0], backbone_output[1]
        if self.use_mole:
            x, pooled_feat = self.mole_backbone_forward(x, masks)
        else:
            backbone_output = self.backbone(x, attention_mask = masks)
            x, pooled_feat = backbone_output[0], backbone_output[1]

        context_feature = x.view(-1, x.shape[-1])
        return_dict['reps'] = x[:, 0, :].clone()
        if span != None:
            outputs, trig_feature = [], []
            for i in range(len(span)):
                if self.is_input_mapping:
                    x_cdt = torch.stack([torch.index_select(x[i], 0, span[i][:, j]) for j in range(span[i].size(-1))])
                    x_cdt = x_cdt.permute(1, 0, 2)
                    x_cdt = x_cdt.contiguous().view(x_cdt.size(0), x_cdt.size(-1) * 2)
                    opt = self.input_map(x_cdt)
                else:
                    # x is (batchsize, seq_len, hidden_size)
                    opt = torch.index_select(x[i], 0, span[i][:, 0]) + torch.index_select(x[i], 0, span[i][:, 1]) # opt is (len(label), hidden_size)
                    # x = x_cdt.permute(1, 0, 2) 
                trig_feature.append(opt) # (batch_size, len(label), hidden_size)
            trig_feature = torch.cat(trig_feature) # (sum of len(label), hidden_size)
        outputs = self.fc(trig_feature) # (sum of len(label), output_size)
        return_dict['outputs'] = outputs
        return_dict['context_feat'] = context_feature
        return_dict['trig_feat'] = trig_feature
        # if args.single_label:
        #     return_outputs = self.fc(enc_out_feature).view(-1, args.class_num + 1)
        # else:
        #     return_outputs = self.fc(feature)
        if aug is not None:
            feature_aug = trig_feature + torch.randn_like(trig_feature) * aug
            outputs_aug = self.fc(feature_aug)
            return_dict['feature_aug'] = feature_aug
            return_dict['outputs_aug'] = outputs_aug
        return return_dict

    def forward_backbone(self, x, masks):
        x = self.backbone(x, attention_mask = masks)
        x = x.last_hidden_state
        return x
    
    def forward_cls(self, x, masks):
        with torch.no_grad():
            backbone_output = self.backbone(x, attention_mask=masks)
            last_hidden_state = backbone_output.last_hidden_state 
            cls_embedding = last_hidden_state[:, 0, :]              
            return cls_embedding                                    

    def forward_input_map(self, x):
        return self.input_map(x)