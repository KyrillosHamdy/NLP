# 5-llama — model overview

This folder contains a compact LLaMA-style transformer implementation in `model.py` (see [5-llama/model.py](5-llama/model.py)). The README below summarizes what is implemented, how it maps to selected papers, what is currently validated by tests, and the next steps to finish a working training + inference pipeline.

**Implemented components**

- **RoPE (rotary embeddings)**: rotary positional embeddings are implemented following (Su et al., 2021.)
- **RMS-style normalization & pre-norm blocks**: a local `RMSNorm` is defined and used as pre-normalization before attention and FFN (aligns with normalization/optimization insights from (Zhang & Sennrich 2019) and (Xiong et al. 2020)).
- **Gated feed-forward (SwiGLU-like)**: `FeedForward` implements a gated FFN using a SwiGLU-style formulation (`w1 -> SiLU` gating multiplied by `w3`, followed by `w2`).
- **Incremental-generation design**: the model expects one token at a time and slices precomputed rotary freqs per `start_pos`, enabling autoregressive incremental decoding with KV cache.
- **Multi-Head Attention (MHA) with KV cache**: `SelfAttention` is implemented with Q/K/V projections, RoPE applied to Q/K, scaled dot-product attention, and KV cache (`cache_k`, `cache_v`) for efficient token-by-token generation.
- **Grouped Query Attention (GQA) support**: implemented via the `n_kv_heads` parameter and `repeat_kv` function, allowing flexible head configurations where query heads exceed key/value heads per (Ainslie et al., 2023).

**Validation status from `test_model.py`**

The current test suite exercises the core model building blocks and integration points:

- `RMSNorm` shape and scale behavior.
- `repeat_kv` identity and GQA expansion behavior.
- RoPE norm preservation and zero-position identity behavior.
- `SelfAttention` output shape for standard MHA.
- Transformer initialization, forward-pass output shape, and multi-step generation stability.
- KV cache updates during autoregressive decoding.
- Determinism and token-sensitivity checks for the full transformer.

This gives good coverage for the architecture and cache logic, though the tests are still focused on correctness and shape/stability checks rather than full training or sampling quality.

**How this maps to the cited papers**

- (Su et al., 2021): RoPE is implemented directly and used to inject position information into Q/K.
- (Zhang & Sennrich, 2019) & (Xiong et al., 2020): the implementation uses RMS-style normalization and pre-norm ordering, and adopts gated/alternative FFN patterns and practical sizing heuristics discussed in those works.
- (Ainslie et al., 2023): the model includes GQA-style head grouping via `n_kv_heads`, which is a key architectural optimization for memory-efficient multi-query decoding.

**References**

- (Zhang & Sennrich 2019) - [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)
- (Xiong et al 2020) - [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745)
- (Su et al 2021) - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- (Ainslie et al 2023) - [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)

**Next steps (requested)**

1. Add an `inference` helper: token-by-token autoregressive sampling with KV cache utilization, temperature and top-p/top-k controls, and batched decoding support.
2. Implement an `AdamW` optimizer from scratch that matches PyTorch semantics (decoupled weight decay, correct bias-correction, stable numerics).
3. Build a training pipeline with data loading, loss computation, and gradient updates; train a small model from scratch (reduced dims/layers/head counts) for both standard MHA and GQA variants, and compare training curves and sample outputs.


