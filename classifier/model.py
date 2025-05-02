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
        self.is_input_mapping = input_map
        self.input_dim = self.backbone.config.hidden_size
        self.num_experts = args.mole_num_experts
        self.top_k = args.mole_top_k
        self.use_mole = args.use_mole
        self.use_lora = args.use_lora

        if args.classifier_layer > 1:
            self.hidden_dim = args.hidden_dim
            self.fc = Classifier(self.input_dim, self.hidden_dim, class_num, num_layers=args.classifier_layer, dropout=args.dropout)
        else:
            self.fc = nn.Linear(self.input_dim, class_num)

        if self.is_input_mapping:
            self.map_hidden_dim = 512
            self.map_input_dim = self.input_dim * 2
            self.input_map = nn.Sequential(
                nn.Linear(self.map_input_dim, self.map_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.map_hidden_dim, self.map_hidden_dim),
                nn.ReLU(),
            )
            self.fc = nn.Linear(self.map_hidden_dim, class_num)
            

        if args.use_lora:
            print("Apply LoRA with shared backbone + LoRA adapter")
            self.peft_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.backbone = get_peft_model(self.backbone, self.peft_config)
            print(f"type(self.backbone): {type(self.backbone)}")
            try:
                self.backbone.freeze_base_model()
            except:
                for name, param in self.backbone.named_parameters():
                    if 'lora_' not in name:
                        param.requires_grad = False
                        
                        
            self.backbone.print_trainable_parameters()
            
        # MoLE setup
        elif args.use_mole:
            print("Apply MoLE with shared backbone + multiple LoRA experts")
            self.peft_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            self.backbone = get_peft_model(self.backbone, self.peft_config)
            try:
                self.backbone.freeze_base_model()
            except:
                for name, param in self.backbone.named_parameters():
                    if 'lora_' not in name:
                        param.requires_grad = False
            
            for i in range(self.num_experts):
                adapter_name = f"expert_{i}"
                self.backbone.add_adapter(adapter_name, self.peft_config)

            self.expert_keys = nn.Parameter(torch.randn(self.num_experts, self.input_dim))  # Learnable keys for gating
            self.softmax = nn.Softmax(dim=-1)

        print("Trainable parameters:")
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n, p.shape)

    def forward(self, x, masks, span=None, aug=None):
        return_dict = {}

        if not self.use_mole:
            # Forward thông thường không MoLE
            out = self.backbone(x, attention_mask=masks)
            x = out.last_hidden_state  # (batch_size, seq_len, hidden_size)
            return_dict['reps'] = x[:, 0, :].clone()

            context_feature = x.view(-1, x.shape[-1])
            return_dict['context_feat'] = context_feature

            if span is not None:
                trig_feature = []
                for i in range(len(span)):
                    if self.is_input_mapping:
                        x_cdt = torch.stack([torch.index_select(x[i], 0, span[i][:, j]) for j in range(span[i].size(-1))])
                        x_cdt = x_cdt.permute(1, 0, 2)
                        x_cdt = x_cdt.contiguous().view(x_cdt.size(0), x_cdt.size(-1) * 2)
                        opt = self.input_map(x_cdt)
                    else:
                        opt = torch.index_select(x[i], 0, span[i][:, 0]) + torch.index_select(x[i], 0, span[i][:, 1])
                    trig_feature.append(opt)

                trig_feature = torch.cat(trig_feature)
                return_dict['trig_feat'] = trig_feature
                outputs = self.fc(trig_feature)
                return_dict['outputs'] = outputs

                if aug is not None:
                    feature_aug = trig_feature + torch.randn_like(trig_feature) * aug
                    outputs_aug = self.fc(feature_aug)
                    return_dict['feature_aug'] = feature_aug
                    return_dict['outputs_aug'] = outputs_aug

            return return_dict

        # === MoLE Forward ===
        with torch.no_grad():
            base_output = self.backbone.base_model(x, attention_mask=masks)
            cls_embedding = base_output.last_hidden_state[:, 0, :]

        gating_logits = torch.matmul(cls_embedding, self.expert_keys.T)
        gating_weights = self.softmax(gating_logits)

        topk_weights, topk_indices = torch.topk(gating_weights, self.top_k, dim=-1)

        final_hidden_states = []

        for b in range(x.size(0)):
            weights_b = topk_weights[b]
            indices_b = topk_indices[b]

            expert_outputs = []
            for i, expert_idx in enumerate(indices_b):
                expert_name = f"expert_{expert_idx.item()}"
                self.backbone.set_adapter(expert_name)
                out = self.backbone(x[b:b+1], attention_mask=masks[b:b+1])
                expert_outputs.append(out.last_hidden_state)

            expert_outputs = torch.stack(expert_outputs)
            weighted_output = torch.sum(weights_b.view(-1, 1, 1, 1) * expert_outputs, dim=0)
            final_hidden_states.append(weighted_output)

        x = torch.cat(final_hidden_states, dim=0)
        return_dict['reps'] = x[:, 0, :].clone()
        context_feature = x.view(-1, x.shape[-1])
        return_dict['context_feat'] = context_feature

        if span is not None:
            trig_feature = []
            for i in range(len(span)):
                if self.is_input_mapping:
                    x_cdt = torch.stack([torch.index_select(x[i], 0, span[i][:, j]) for j in range(span[i].size(-1))])
                    x_cdt = x_cdt.permute(1, 0, 2)
                    x_cdt = x_cdt.contiguous().view(x_cdt.size(0), x_cdt.size(-1) * 2)
                    opt = self.input_map(x_cdt)
                else:
                    opt = torch.index_select(x[i], 0, span[i][:, 0]) + torch.index_select(x[i], 0, span[i][:, 1])
                trig_feature.append(opt)

            trig_feature = torch.cat(trig_feature)
            return_dict['trig_feat'] = trig_feature
            outputs = self.fc(trig_feature)
            return_dict['outputs'] = outputs

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