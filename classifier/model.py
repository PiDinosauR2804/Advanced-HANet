import torch
from transformers import BertModel
import torch.nn as nn
from configs import parse_arguments
from peft import get_peft_model, LoraConfig, TaskType
from loguru import logger


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
    def __init__(self, args):
        super().__init__()
        self.is_input_mapping = args.input_map
        self.class_num = args.class_num + 1
        self.use_mole = args.use_mole
        self.use_lora = args.use_lora
        self.top_k = args.mole_top_k
        self.num_experts = args.mole_num_experts
        self.use_general_expert = args.use_general_expert
        self.uniform_expert = False
        self.general_expert_weight = args.general_expert_weight

        # Load backbone
        self.backbone = BertModel.from_pretrained(args.backbone)
        self.input_dim = self.backbone.config.hidden_size

        # Classifier
        if args.classifier_layer > 1:
            self.hidden_dim = args.hidden_dim
            self.fc = Classifier(self.input_dim, self.hidden_dim, self.class_num,
                                 num_layers=args.classifier_layer, dropout=args.dropout)
        else:
            self.fc = nn.Linear(self.input_dim, self.class_num)

        # Optional input mapping
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
            self.fc = nn.Linear(self.map_hidden_dim, self.class_num)

        # Setup LoRA or MoLE
        if self.use_lora or self.use_mole:
            self.peft_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, self.peft_config)
            try:
                self.backbone.freeze_base_model()
            except:
                for name, param in self.backbone.named_parameters():
                    if 'lora_' not in name:
                        param.requires_grad = False

            if self.use_mole:
                self.backbone.add_adapter("general_expert", self.peft_config)
                for i in range(self.num_experts):
                    self.backbone.add_adapter(f"expert_{i}", self.peft_config)

                self.expert_keys = nn.Parameter(torch.randn(self.num_experts, self.input_dim))
                self.softmax = nn.Softmax(dim=-1)

            self.backbone.print_trainable_parameters()

        print("Trainable parameters:")
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n, p.shape)
                
    def turn_uniform_expert(self, turn_on=True):
        self.uniform_expert = turn_on
        logger.info(f"Uniform expert: {turn_on}")

    def forward(self, x, masks, span=None, aug=None):
        if self.use_mole:
            return self._forward_mole(x, masks, span, aug)
        else:
            return self._forward_normal(x, masks, span, aug, use_lora=self.use_lora)

    def _forward_normal(self, x, masks, span=None, aug=None, use_lora=False):
        out = self.backbone(x, attention_mask=masks)
        hidden = out.last_hidden_state
        return_dict = {
            'reps': hidden[:, 0, :].clone(),
            'context_feat': hidden.view(-1, hidden.shape[-1])
        }

        if span is not None:
            trig_feature = self._extract_trigger(hidden, span)
            return_dict['trig_feat'] = trig_feature
            return_dict['outputs'] = self.fc(trig_feature)

            if aug is not None:
                feature_aug = trig_feature + torch.randn_like(trig_feature) * aug
                return_dict['feature_aug'] = feature_aug
                return_dict['outputs_aug'] = self.fc(feature_aug)

        return return_dict

    def _forward_mole(self, x, masks, span=None, aug=None):
        with torch.no_grad():
            base_output = self.backbone.base_model(x, attention_mask=masks)
            cls_embedding = base_output.last_hidden_state[:, 0, :]

        # Gating
        gating_logits = torch.matmul(cls_embedding, self.expert_keys.T)
        gating_weights = self.softmax(gating_logits)

        # Nếu uniform_expert được bật, sử dụng trọng số đồng đều cho các expert
        if self.uniform_expert:
            gating_weights = torch.full_like(gating_weights, 1.0 / self.num_experts)

        avg_weights = gating_weights.mean(dim=0)
        uniform = torch.full_like(avg_weights, 1.0 / self.num_experts)
        load_balancing_loss = torch.sum((avg_weights - uniform) ** 2)
        entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=-1).mean()

        return_dict = {
            'load_balance_loss': load_balancing_loss,
            'entropy_loss': entropy,
        }

        # Top-k routing
        topk_weights, topk_indices = torch.topk(gating_weights, self.top_k, dim=-1)
        final_hidden_states = []

        for b in range(x.size(0)):
            weighted_hidden = 0
            if self.use_general_expert:
                self.backbone.set_adapter("general_expert")
                general_output = self.backbone(x[b:b+1], attention_mask=masks[b:b+1])
                weighted_hidden += self.general_expert_weight * general_output.last_hidden_state
            for i, expert_idx in enumerate(topk_indices[b]):
                weight = topk_weights[b, i]
                self.backbone.set_adapter(f"expert_{expert_idx.item()}")
                expert_output = self.backbone(x[b:b+1], attention_mask=masks[b:b+1])
                weighted_hidden += weight * expert_output.last_hidden_state
            final_hidden_states.append(weighted_hidden)

        x_out = torch.cat(final_hidden_states, dim=0)
        return_dict['reps'] = x_out[:, 0, :].clone()
        return_dict['context_feat'] = x_out.view(-1, x_out.shape[-1])

        if span is not None:
            trig_feature = self._extract_trigger(x_out, span)
            return_dict['trig_feat'] = trig_feature
            return_dict['outputs'] = self.fc(trig_feature)

            if aug is not None:
                feature_aug = trig_feature + torch.randn_like(trig_feature) * aug
                return_dict['feature_aug'] = feature_aug
                return_dict['outputs_aug'] = self.fc(feature_aug)

        return return_dict


    def _extract_trigger(self, x, span):
        trig_feature = []
        for i in range(len(span)):
            if self.is_input_mapping:
                x_cdt = torch.stack([torch.index_select(x[i], 0, span[i][:, j]) for j in range(span[i].size(-1))])
                x_cdt = x_cdt.permute(1, 0, 2).contiguous().view(x_cdt.size(1), -1)
                opt = self.input_map(x_cdt)
            else:
                opt = torch.index_select(x[i], 0, span[i][:, 0]) + torch.index_select(x[i], 0, span[i][:, 1])
            trig_feature.append(opt)
        return torch.cat(trig_feature)

    def forward_backbone(self, x, masks):
        out = self.backbone(x, attention_mask=masks)
        return out.last_hidden_state

    def forward_cls(self, x, masks):
        with torch.no_grad():
            out = self.backbone(x, attention_mask=masks)
            return out.last_hidden_state[:, 0, :]

    def forward_input_map(self, x):
        return self.input_map(x)