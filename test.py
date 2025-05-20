class LoraRouter(nn.Module):
    def __init__(self, hidden_size, experts_num=8, experts_pool_num=4, fixed_experts_num=1, 
                 task_experts_num=1, select_experts_num=2, task_num=3, fixed_experts_weight=0.5):
        super().__init__()
        self.experts_num = experts_num
        self.select_experts_num = select_experts_num
        self.experts_pool_num = experts_pool_num
        self.task_experts_num = task_experts_num
        self.fixed_experts_num = fixed_experts_num
        self.fixed_experts_weight = fixed_experts_weight

        self.router_network = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, experts_pool_num, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(experts_pool_num, experts_pool_num, bias=False),
        )
        # task_keys = torch.randn(task_num, hidden_size)
        # self.task_keys = nn.Parameter(task_keys, requires_grad = True)

        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(1)
        
    def forward(self, hidden_state):
        batch_size, seq_length, hz = hidden_state.shape
        hidden_state = hidden_state.view(-1, hz)

        # TODO
        logits_router = self.router_network(hidden_state)
        top_k_logits, top_k_indices = logits_router.topk(min(self.select_experts_num, self.experts_pool_num), dim=1)  # 选择并排序前k+1个权重

        top_k_scores = self.softmax(top_k_logits.to(torch.float32))
        top_k_scores = top_k_scores.to(hidden_state.dtype)
        top_k_indices = top_k_indices.view(batch_size, seq_length, -1)
        top_k_scores = top_k_scores.view(batch_size, seq_length, -1)

        ## quangnm
        fixed_indices = torch.full((batch_size, seq_length), self.experts_pool_num).to(device=top_k_indices.device)
        fixed_score = torch.full((batch_size, seq_length), 1).to(device=top_k_indices.device)
        ## quangnm

        
        btz, seq, _ = top_k_indices.shape
        if self.fixed_experts_num != 0:
            fixed_values = fixed_indices.unsqueeze(1).clone().detach().to(device=top_k_indices.device).expand(btz, seq, -1)
            top_k_indices = torch.cat([top_k_indices, fixed_values], dim=-1)
            fixed_score = fixed_score.unsqueeze(1).clone().detach().to(device=top_k_indices.device).expand(btz, seq, -1)
            # top_k_scores = torch.cat([top_k_scores*select_weight, fixed_score*fixed_weight], dim=-1)
            ## quangnm
            top_k_scores = torch.cat([top_k_scores, fixed_score*self.fixed_experts_weight], dim=-1)
            ## quangnm
        expert_mask = torch.nn.functional.one_hot(top_k_indices.view(batch_size*seq_length, -1), num_classes=self.experts_num).permute(2, 1, 0)
        top_k_scores = top_k_scores.view(batch_size*seq_length, -1)

        return top_k_indices, top_k_scores, expert_mask

    def model_replay(self, inputs_embeds):
        hidden_state_mean = inputs_embeds.mean(dim=1)
        similarity = F.cosine_similarity(hidden_state_mean.unsqueeze(1), self.task_keys.unsqueeze(0), dim=-1)

        inputs_embeds = inputs_embeds.view(-1, self.hidden_size)
        logits_router = self.router_network(inputs_embeds)
        
        return logits_router.to(torch.float32), similarity.to(torch.float32)


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
                                      select_experts_num=self.select_experts_num, task_num=self.task_num, fixed_experts_weight=self.fixed_experts_weight)

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

        top_k_indices, top_k_scores, expert_mask = self.lora_router(hidden_states)
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