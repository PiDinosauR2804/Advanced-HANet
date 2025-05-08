import torch
from transformers import BertModel
import torch.nn as nn
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
        if self.uniform_expert != turn_on:
            self.uniform_expert = turn_on
            logger.info(f"Uniform expert: {turn_on}")

    def forward(self, x, masks, span=None, aug=None, batch_size=32, train=True):
        if self.use_mole:
            return self._forward_mole(x, masks, span, aug, batch_size, train)
        else:
            return self._forward_normal(x, masks, span, aug, batch_size, train)

    def _forward_normal(self, x, masks, span=None, aug=None, batch_size=32, train=True):
        # ========== [1] Forward qua backbone base_model theo batch size ==========
        B = x.size(0)
        out = []
        for batch in range(0, B, batch_size):
            start = batch
            end = min(batch + batch_size, B)
            out_x = self.backbone(x[start:end], attention_mask=masks[start:end])
            out.append(out_x.last_hidden_state)
            
        out = torch.cat(out, dim=0)
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

    def _forward_mole(self, x, masks, span=None, aug=None, batch_size=32, train=True):
        B = x.size(0)

        # ========== [1] Tính gating và top-k ==========
        if self.uniform_expert:
            gating_weights = torch.ones(B, self.num_experts, device=x.device) / self.num_experts  # Trọng số đều cho tất cả expert
            # random expert id cho mỗi sample
            topk_indices = torch.randint(0, self.num_experts, (B, self.top_k), device=x.device)  # (B, k)
            topk_mask = torch.zeros(B, self.num_experts, device=x.device, dtype=torch.bool).scatter_(1, topk_indices, True)  # (B, E)
        else:
        # ========== [2] Forward qua backbone base_model theo batch size ==========
            with torch.no_grad():
                cls_embedding = []
                for batch in range(0, B, batch_size):
                    start = batch
                    end = min(batch + batch_size, B)
                    base_output = self.backbone.base_model(x[start:end], attention_mask=masks[start:end])
                    cls_embedding.append(base_output.last_hidden_state[:, 0, :])  # Lấy embedding của [CLS]
                cls_embedding = torch.cat(cls_embedding, dim=0)  # (B, H)
            gating_logits = torch.matmul(cls_embedding, self.expert_keys.T)  # (B, E)
            gating_weights = self.softmax(gating_logits)
        # ======== 3. Lấy top-k expert cho mỗi sample và tạo mask ========
            topk_weights, topk_indices = torch.topk(gating_weights, self.top_k, dim=-1)  # (B, k)
            topk_mask = torch.zeros(B, self.num_experts, device=x.device, dtype=torch.bool).scatter_(1, topk_indices, True)

        return_dict = {}

        if not self.uniform_expert and train:
            avg_weights = gating_weights.mean(dim=0)
            uniform = torch.full_like(avg_weights, 1.0 / self.num_experts)
            return_dict['load_balance_loss'] = torch.sum((avg_weights - uniform) ** 2)
            return_dict['entropy_loss'] = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=-1).mean()

        # ======== 4. Gom nhóm các sample theo expert ========
        expert_outputs = torch.zeros(B, self.num_experts, self.input_dim, device=x.device)  # (B, E, H)
        
        for expert_id in range(self.num_experts):
            mask = topk_mask[:, expert_id]  # Chọn tất cả sample thuộc expert_id
            if mask.sum() == 0:
                continue

            self.backbone.set_adapter(f"expert_{expert_id}")
            
            # Chia batch để forward
            out_x = []
            for i in range(0, mask.sum().item(), batch_size):
                sub_x = x[mask][i:i + batch_size]
                sub_masks = masks[mask][i:i + batch_size]
                out_x_batch = self.backbone(sub_x, attention_mask=sub_masks).last_hidden_state
                out_x.append(out_x_batch)
            
            # Kết hợp batch
            out_x = torch.cat(out_x, dim=0)
            
            # Gán vào expert_outputs output có trọng số
            expert_outputs[mask, expert_id] = out_x * gating_weights[mask, expert_id].unsqueeze(-1)  # (B, E, H)
            
        # ======== 5. Tính tổng đầu ra các expert ========
        expert_outputs = torch.sum(expert_outputs, dim=1)  # (B, H)

        # ======== 5. Optional: add general expert ========
        if self.use_general_expert:
            self.backbone.set_adapter("general_expert")
            general_out = []
            for i in range(0, B, batch_size):
                start = i
                end = min(i + batch_size, B)
                general_out_batch = self.backbone(x[start:end], attention_mask=masks[start:end]).last_hidden_state
                general_out.append(general_out_batch)
            general_out = torch.cat(general_out, dim=0)
            expert_outputs += self.general_expert_weight * general_out

        # ======== 6. Tổng hợp kết quả và trả về ========
        return_dict['reps'] = expert_outputs[:, 0, :].clone()
        return_dict['context_feat'] = expert_outputs.view(-1, expert_outputs.shape[-1])

        if span is not None:
            trig_feature = self._extract_trigger(expert_outputs, span)
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