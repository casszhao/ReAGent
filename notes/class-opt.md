
- site-packages/transformers/models/opt/modeling_opt.py
- embed_tokens: Embedding(50265, 4096, padding_idx=1) = model.model.decoder.embed_tokens
- embed_positions: OPTLearnedPositionalEmbedding(2050, 4096) = model.model.decoder.embed_positions


forward Args
```py
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
```


```
OPTForCausalLM(
  (model): OPTModel(
    (decoder): OPTDecoder(
      (embed_tokens): Embedding(50265, 4096, padding_idx=1)
      (embed_positions): OPTLearnedPositionalEmbedding(2050, 4096)
      (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
      (layers): ModuleList(
        (0): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (1): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (2): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (3): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (4): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (5): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (6): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (7): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (8): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (9): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (10): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (11): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (12): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (13): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (14): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (15): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (16): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (17): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (18): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (19): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (20): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (21): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (22): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (23): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (24): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (25): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (26): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (27): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (28): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (29): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (30): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
        (31): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=4096, out_features=16384, bias=True)
          (fc2): Linear(in_features=16384, out_features=4096, bias=True)
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (lm_head): Linear(in_features=4096, out_features=50265, bias=False)
)
```


