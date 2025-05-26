import torch
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSdpaSelfAttention
import torch.nn as nn
from torch.nn import functional as F
from peft import get_peft_model, LoraConfig, TaskType
from loguru import logger
import math
from typing import Optional, Tuple
from packaging import version
from transformers.utils import get_torch_version

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
    
class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.
    ):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_alpha = lora_alpha

        self.out_features = out_features

        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        self.scaling = self.lora_alpha / self.r
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor):
        result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result.reshape(x.shape[0], -1, self.out_features)

class LoraRouter(nn.Module):
    def __init__(self, hidden_size, experts_num=8, experts_pool_num=4, fixed_experts_num=1, 
                 task_experts_num=1, select_experts_num=2, task_num=3, prompt_config=None):
        super().__init__()
        self.experts_num = experts_num
        self.select_experts_num = select_experts_num
        self.experts_pool_num = experts_pool_num
        self.task_experts_num = task_experts_num
        self.fixed_experts_num = fixed_experts_num
        self.fixed_experts_weight = prompt_config.general_expert_weight
        self.hidden_size = hidden_size
        self.gate = prompt_config.gate
        self.level = prompt_config.mole_level

        if self.gate == "tanh":
            self.router_network = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, experts_pool_num, bias=False),
                torch.nn.Tanh(),
                torch.nn.Linear(experts_pool_num, experts_pool_num, bias=False),
            )
            
            self.softmax = nn.Softmax(1)
            
        elif self.gate == "softmax":
            self.router_network = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, experts_pool_num, bias=False),
            )
            
            self.softmax = nn.Softmax(1)
            
        elif self.gate == "sigmoid":
            self.router_network = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, experts_pool_num, bias=False),
                torch.nn.Sigmoid(),
            )
            
        # task_keys = torch.randn(task_num, hidden_size)
        # self.task_keys = nn.Parameter(task_keys, requires_grad = True)
        self.router_bias = torch.ones(experts_pool_num, device=prompt_config.device, dtype=torch.float32)
        self.gamma = prompt_config.gamma_router

        
    def tune_bias(self, tune: torch.Tensor):
        assert tune.shape == self.router_bias.shape, "Mismatch shape"
        scale = self.gamma ** tune.to(dtype=self.router_bias.dtype)
        self.router_bias *= scale

        
    def forward(self, hidden_state):
        batch_size, seq_length, hz = hidden_state.shape
        if self.level == "token":
            hidden_state = hidden_state.view(-1, hz)
        elif self.level == "sequence":
            cls_token = hidden_state[:, 0, :]  # shape: [batch_size, hz]
            hidden_state = cls_token.unsqueeze(1).expand(-1, seq_length, -1)  # shape: [batch_size, seq_length, hz]
            hidden_state = hidden_state.contiguous().view(-1, hz)  # flatten
        else:
            raise ValueError(f"Unsupported level: {self.level}. Supported levels are 'token' and 'sequence'.")

        # TODO
        logits_router = self.router_network(hidden_state)
        _, top_k_indices = torch.topk(logits_router + self.router_bias, min(self.select_experts_num, self.experts_pool_num), dim=1)  # get top k logits and indices
        top_k_logits = logits_router.gather(1, top_k_indices)  # gather top k logits
        
        if self.gate == "sigmoid":
            # normalize scores to sum to 1
            top_k_scores = top_k_logits / top_k_logits.sum(dim=1, keepdim=True)
        else:
            top_k_scores = self.softmax(top_k_logits.to(torch.float32))
            
        top_k_scores = top_k_scores.to(hidden_state.dtype)
        top_k_indices = top_k_indices.view(batch_size, seq_length, -1)
        top_k_scores = top_k_scores.view(batch_size, seq_length, -1)

        # 1. Xử lý fixed experts nếu có
        if self.fixed_experts_num != 0:
            # a. Tạo các chỉ số fixed experts (giả sử liên tiếp từ experts_pool_num)
            fixed_indices = torch.arange(
                self.experts_pool_num,
                self.experts_pool_num + self.fixed_experts_num,
                device=top_k_indices.device,
            )  # shape: (fixed_experts_num,)

            # b. Expand thành shape: (batch_size, seq_length_len, fixed_experts_num)
            fixed_indices = fixed_indices.view(1, 1, -1).expand(batch_size, seq_length, -1)

            # c. Fixed scores: cùng shape và giá trị
            fixed_scores = torch.full_like(fixed_indices, fill_value=self.fixed_experts_weight, dtype=top_k_scores.dtype)

            # d. Nối với top-k indices và scores
            top_k_indices = torch.cat([top_k_indices, fixed_indices], dim=-1)     # (batch_size, seq_length, select_experts_num + fixed_experts_num)
            top_k_scores  = torch.cat([top_k_scores, fixed_scores], dim=-1)
            
        expert_mask = torch.nn.functional.one_hot(top_k_indices.view(batch_size*seq_length, -1), num_classes=self.experts_num).permute(2, 1, 0)
        top_k_scores = top_k_scores.view(batch_size*seq_length, -1)

        return top_k_indices, top_k_scores, expert_mask, logits_router

    def model_replay(self, inputs_embeds):
        inputs_embeds = inputs_embeds.view(-1, self.hidden_size)
        logits_router = self.router_network(inputs_embeds)
        
        return logits_router.to(torch.float32)


class BertSelfAttentionWrapper(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, old_attention_layer: BertSelfAttention, config, prompt_config, position_embedding_type=None):
        super().__init__()
        self.dropout_prob = config.attention_probs_dropout_prob
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")
        self.head_dim = self.hidden_size // self.num_heads
        self.prompt_config = prompt_config
        self.is_decoder = old_attention_layer.is_decoder
        self.position_embedding_type = position_embedding_type if position_embedding_type is not None else config.position_embedding_type
        self.all_head_size = self.num_heads * self.head_dim

        self.experts_pool_num = prompt_config.mole_num_experts
        self.fixed_experts_num = prompt_config.mole_num_general_expert
        self.fixed_experts_weight = prompt_config.general_expert_weight
        self.task_experts_num = 0
        self.experts_num = self.experts_pool_num + self.fixed_experts_num + self.task_experts_num
        self.select_experts_num = prompt_config.mole_top_k
        self.task_num = prompt_config.task_num
        self.lora_r = prompt_config.lora_rank
        self.lora_alpha = prompt_config.lora_alpha
        self.lora_dropout = prompt_config.lora_dropout
        self.num_choose = torch.zeros(self.experts_num, device=prompt_config.device, dtype=torch.int64)
        self.logits_router = None
        self.balance_ratio = prompt_config.balance_ratio

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = old_attention_layer.query
        self.k_proj = old_attention_layer.key
        self.v_proj = old_attention_layer.value

        # LoRA layers for previous task

        self.lora_router = LoraRouter(self.hidden_size, experts_num=self.experts_num, experts_pool_num=self.experts_pool_num, 
                                      fixed_experts_num=self.fixed_experts_num, task_experts_num=self.task_experts_num, 
                                      select_experts_num=self.select_experts_num, task_num=self.task_num, prompt_config=prompt_config)

        self.lora_experts_q, self.lora_experts_v = None, None
        self.lora_experts_q = nn.ModuleList()
        for i in range(self.experts_num):
            layer = LoRALayer(self.hidden_size, self.num_heads * self.head_dim, 
                              r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
            self.lora_experts_q.append(layer)

        self.lora_experts_v = nn.ModuleList()
        for i in range(self.experts_num):
            layer = LoRALayer(self.hidden_size, self.num_heads * self.head_dim, 
                              r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
            self.lora_experts_v.append(layer)
        

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def clear_num_choose(self):
        self.num_choose = torch.zeros(self.experts_num, device=self.num_choose.device, dtype=self.num_choose.dtype)
    
    def get_num_choose(self):
        return self.num_choose
    
    def tune_bias(self):
        # Khởi tạo vector tune
        tune = torch.zeros(self.experts_pool_num, device=self.num_choose.device, dtype=torch.int8)

        # Mức cân bằng lý tưởng (phần trăm chọn đều cho mỗi expert)
        balance_choose = 1 / self.experts_pool_num

        # Tính phần trăm chọn của mỗi expert
        if self.fixed_experts_num == 0:
            percent = self.num_choose / self.num_choose.sum()
        else:
            percent = self.num_choose[:self.experts_pool_num] / self.num_choose[:self.experts_pool_num].sum()

        # Xác định expert nào nên giảm (-1), tăng (+1), hoặc giữ nguyên (0)
        tune[percent < (balance_choose * (1 - self.balance_ratio))] = 1
        tune[percent > (balance_choose * (1 + self.balance_ratio))] = -1

        # logger.info("="*100)
        # logger.info(f"balance_choose: {balance_choose}")
        # logger.info(f"percent: {percent}")
        # logger.info(f"tune_bias: {tune}")
        # logger.info(f"bias before: {self.lora_router.router_bias}")
        
        # Gọi hàm tune_bias từ lora_router
        self.lora_router.tune_bias(tune.to(dtype=torch.float32))
        
        # logger.info(f"bias after: {self.lora_router.router_bias}")
        

    # Adapted from BertSelfAttention
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
            raise NotImplementedError(
                f"position_embedding_type: {self.position_embedding_type} and output_attentions: {output_attentions} "
                f"and head_mask: {head_mask} are not supported in this wrapper."
            )
        bsz, tgt_len, _ = hidden_states.size()
        
        def agg_lora_states(hidden_states, lora_layer, top_k_indices, top_k_scores, expert_mask):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
            for expert_idx in range(self.experts_num):
                expert_layer = lora_layer[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])
                top_x_list = top_x.tolist()
                idx_list = idx.tolist()
                if len(top_x_list) == 0:
                    continue
                current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state).squeeze() * top_k_scores[top_x_list, idx_list, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            return final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        top_k_indices, top_k_scores, expert_mask, self.logits_router = self.lora_router(hidden_states)
        
        self.num_choose += expert_mask.sum(dim=(1, 2)).to(self.num_choose.dtype)
            
        lora_experts_q = self.lora_experts_q
        lora_experts_v = self.lora_experts_v
        
        ## quangnm
        query_layer = (self.q_proj(hidden_states) + 
                       agg_lora_states(hidden_states, lora_experts_q, top_k_indices, 
                                       top_k_scores, expert_mask)).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        # If this is instantiated as a cross-attention module, the keys and values come from an encoder; the attention
        # mask needs to be such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning
        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            key_layer, value_layer = past_key_value
        else:
            value_layer = (self.v_proj(current_states) + 
                           agg_lora_states(current_states, lora_experts_v, top_k_indices, 
                                           top_k_scores, expert_mask)).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_layer = self.k_proj(current_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

            if past_key_value is not None and not is_cross_attention:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
        # a causal mask in case tgt_len == 1.
        is_causal = (
            True if self.is_decoder and not is_cross_attention and attention_mask is None and tgt_len > 1 else False
        )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

        outputs = (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
        
        
class BertED(nn.Module):
    def __init__(self, args, backbone_path=None):
        super().__init__()
        self.is_input_mapping = args.input_map
        self.class_num = args.class_num + 1
        self.use_mole = args.use_mole
        self.use_lora = args.use_lora
        self.num_experts = args.mole_num_experts
        self.uniform_expert = False
        self.args = args

        # Load backbone
        if backbone_path is not None:
            self.backbone = BertModel.from_pretrained(args.backbone)
            self.input_dim = self.backbone.config.hidden_size
            self.backbone.load_state_dict(torch.load(backbone_path)) 
            logger.info(f"Load backbone from {backbone_path}")
        else:
            self.backbone = BertModel.from_pretrained(args.backbone)
            self.input_dim = self.backbone.config.hidden_size
        self.seqlen = args.max_seqlen + 2  # +2 for [CLS] and [SEP]

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
        if self.use_lora:
            self.peft_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, self.peft_config, adapter_name="general_expert")

        elif self.use_mole:
            bert_config = self.backbone.config
            for layer in self.backbone.encoder.layer:
                layer.attention.self = BertSelfAttentionWrapper(layer.attention.self, bert_config, self.args)
                
            logger.info(f"Use MoLE with {self.num_experts} experts, top-k {args.mole_top_k}, route level {args.mole_level}, "
                        f"general expert weight {args.general_expert_weight}, balance ratio {args.balance_ratio}, ")

        if not args.no_freeze_bert:
            self.freeze_backbone()
        
        if args.print_trainable_params:
            self.print_trainable_parameters()

    def print_trainable_parameters(self):
        print("Trainable parameters:")
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n, p.shape)
            
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        logger.info("Freeze backbone parameters")
        
    def turn_uniform_expert(self, turn_on=False):
        pass
    
    def clear_num_choose(self):
        if self.use_mole:
            for layer in self.backbone.encoder.layer:
                layer.attention.self.clear_num_choose()
        else:
            pass
        
    def get_num_choose(self):
        if self.use_mole:
            num_choose = []
            for layer in self.backbone.encoder.layer:
                num_choose.append(layer.attention.self.get_num_choose())
                
            num_choose = torch.stack(num_choose, dim=0)
            return num_choose
        else:
            return None
        
    def tune_bias(self):
        if self.use_mole:
            for layer in self.backbone.encoder.layer:
                layer.attention.self.tune_bias()
        else:
            pass
        
    def get_logits_router(self):
        if self.use_mole:
            logits_router = []
            for layer in self.backbone.encoder.layer:
                logits_router.append(layer.attention.self.logits_router)
                
            logits_router = torch.stack(logits_router, dim=0)
            return logits_router
        else:
            return None

    def forward(self, x, masks, span=None, aug=None, train=True):
        out = self.backbone(x, attention_mask=masks)
        hidden = out.last_hidden_state
        return_dict = {
            'reps': hidden[:, 0, :].clone(),
            'context_feat': hidden.view(-1, hidden.shape[-1]),
            'logits_router': self.get_logits_router(),
        }
        
        if self.use_mole:
            return_dict['entropy_loss'] = 0
            return_dict['load_balance_loss'] = 0

        if span is not None:
            trig_feature = self._extract_trigger(hidden, span)
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