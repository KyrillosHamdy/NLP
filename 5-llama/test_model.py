import sys
import torch

from model import (
    ModelArgs,
    Transformer,
    RMSNorm,
    SelfAttention,
    apply_rotary_embeddings,
    repeat_kv,
    precompute_freqs_cis,
)


def make_args(**overrides):
    # n_kv_heads=None -> MHA (n_kv_heads == n_heads), matching your setup
    defaults = dict(
        dim=64,
        n_layers=2,
        n_heads=8,
        n_kv_heads=None,
        vocab_size=100,
        multiple_of=32,
        max_batch_size=4,
        max_seq_len=16,
        device="cpu",
    )
    defaults.update(overrides)
    return ModelArgs(**defaults)


# ---------- building blocks ----------

def test_rmsnorm_shape_and_scale():
    x = torch.randn(2, 5, 32)
    norm = RMSNorm(32)
    out = norm(x)
    assert out.shape == x.shape
    # scale is init'd to all-ones, so output RMS per vector should be ~1
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_repeat_kv_identity_when_n_reps_1():
    # this is the path your MHA config actually exercises (n_reps == 1)
    x = torch.randn(2, 5, 4, 8)
    out = repeat_kv(x, 1)
    assert torch.equal(out, x)


def test_repeat_kv_expands_heads_gqa():
    # bonus: repeat_kv should still work generically for n_reps > 1
    # even though this run is MHA-only
    batch, seq, n_kv, head_dim = 2, 5, 4, 8
    n_reps = 2
    x = torch.randn(batch, seq, n_kv, head_dim)
    out = repeat_kv(x, n_reps)
    assert out.shape == (batch, seq, n_kv * n_reps, head_dim)
    assert torch.equal(out[:, :, 0], x[:, :, 0])
    assert torch.equal(out[:, :, 1], x[:, :, 0])
    assert torch.equal(out[:, :, 2], x[:, :, 1])


def test_rotary_embeddings_preserve_norm():
    # RoPE is a rotation -> must preserve vector norm
    batch, seq, n_heads, head_dim = 2, 1, 4, 16
    x = torch.randn(batch, seq, n_heads, head_dim)
    freqs = precompute_freqs_cis(head_dim, max_seq_len=10, device="cpu")
    out = apply_rotary_embeddings(x, freqs[0:seq], device="cpu")
    assert out.shape == x.shape
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-3)


def test_rotary_embeddings_position_zero_is_identity():
    # at position 0, angle=0 -> rotation is a no-op
    x = torch.randn(1, 1, 2, 8)
    freqs = precompute_freqs_cis(8, max_seq_len=10, device="cpu")
    out = apply_rotary_embeddings(x, freqs[0:1], device="cpu")
    assert torch.allclose(out, x, atol=1e-5)


# ---------- SelfAttention / Transformer integration ----------

def test_self_attention_output_shape_mha():
    args = make_args()
    attn = SelfAttention(args)
    freqs = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len, device="cpu")
    x = torch.randn(2, 1, args.dim)
    out = attn(x, start_pos=0, complex_freqs=freqs[0:1])
    assert out.shape == (2, 1, args.dim)


def test_self_attention_respects_batch_smaller_than_max():
    # catches the cache-slicing bug: batch_size < max_batch_size must not
    # pull zero/unrelated rows from the cache into the computation
    args = make_args(max_batch_size=8)
    attn = SelfAttention(args)
    freqs = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len, device="cpu")
    batch_size = 3
    x = torch.randn(batch_size, 1, args.dim)
    out = attn(x, start_pos=0, complex_freqs=freqs[0:1])
    assert out.shape == (batch_size, 1, args.dim)


def test_transformer_init_layer_count():
    # catches the n_layer / n_layers typo
    args = make_args(n_layers=5)
    model = Transformer(args)
    assert len(model.layers) == 5


def test_transformer_forward_shape():
    args = make_args()
    model = Transformer(args)
    model.eval()
    tokens = torch.randint(0, args.vocab_size, (2, 1))
    with torch.no_grad():
        out = model(tokens, start_pos=0)
    assert out.shape == (2, 1, args.vocab_size)


def test_transformer_forward_no_nans_over_multiple_steps():
    args = make_args()
    model = Transformer(args)
    model.eval()
    with torch.no_grad():
        for start_pos in range(5):
            tokens = torch.randint(0, args.vocab_size, (2, 1))
            out = model(tokens, start_pos)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()


def test_transformer_kv_cache_updates():
    args = make_args()
    model = Transformer(args)
    model.eval()
    layer0_attn = model.layers[0].attention
    before = layer0_attn.cache_k[:2, 3].clone()
    tokens = torch.randint(0, args.vocab_size, (2, 1))
    with torch.no_grad():
        model(tokens, start_pos=3)
    after = layer0_attn.cache_k[:2, 3].clone()
    assert not torch.equal(before, after)


def test_transformer_deterministic_given_seed():
    args = make_args()
    torch.manual_seed(0)
    model1 = Transformer(args)
    torch.manual_seed(0)
    model2 = Transformer(args)
    model1.eval(); model2.eval()
    tokens = torch.tensor([[5], [7]])
    with torch.no_grad():
        out1 = model1(tokens, start_pos=0)
        out2 = model2(tokens, start_pos=0)
    assert torch.allclose(out1, out2)


def test_transformer_different_tokens_different_output():
    # sanity: changing one batch row's token shouldn't change the other's output
    args = make_args()
    model = Transformer(args)
    model.eval()
    t1 = torch.tensor([[3], [3]])
    t2 = torch.tensor([[3], [9]])
    with torch.no_grad():
        out1 = model(t1, start_pos=0)
        out2 = model(t2, start_pos=0)
    assert not torch.allclose(out1[1], out2[1])
    assert torch.allclose(out1[0], out2[0])


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(1 if failed else 0)
