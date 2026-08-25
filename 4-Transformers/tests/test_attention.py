import torch
from transformer.config import GPTConfig
from transformer.attention import Head, MultiHeadAttention

def make_config():
    return GPTConfig(
        vocab_size=50,
        block_size=8,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0
    )
    
def test_shape_preserved():
    config = make_config()
    x = torch.randn(4, config.block_size, config.n_embd) 
    out = MultiHeadAttention(config)(x)
    assert out.shape == x.shape

@ torch.no_grad()
def test_causal_masking():
    """Changing a future token must not change the output at earlier positions."""
    
    config = make_config()
    torch.manual_seed(1337)
    x = torch.randn(4, config.block_size, config.n_embd)
    attn = MultiHeadAttention(config)
    attn.eval()  # disable dropout for testing
    out = attn(x)
    x2 = x.clone()
    last_pos = config.block_size - 1
    x2[:, last_pos, :] = 100.0 
    out2 = attn(x2)
    
    # every position except the last one must be untouched
    assert torch.allclose(out[:, :last_pos, :], out2[:, :last_pos, :], atol=1e-6),\
    "Future token leaked into past position's output — causal mask is broken"
    # the last position's output SHOULD change (it attends to itself)
    assert not torch.allclose(out[:, last_pos, :], out2[:, last_pos, :], atol=1e-6), \
    "Last position didn't change at all — attention isn't working"

def test_scaling_uses_head_size_not_embd():
    """Regression test for the C vs head_size scaling bug."""
    config = make_config()
    head = Head(config)
    # head_size should be n_embd // n_head, and the key projection's out_features
    # should match config.head_size, not config.n_embd
    assert head.key.out_features == config.head_size
    assert head.key.out_features == config.n_embd // config.n_head
    
def test_pad_tokens_dont_affect_real_token_output():
    config = make_config()  # pad_index set
    torch.manual_seed(0)
    attn = MultiHeadAttention(config)
    attn.eval()

    B, T = 1, config.block_size
    x = torch.randn(B, T, config.n_embd)
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    pad_mask[:, -2:] = True   # last 2 positions are PAD

    with torch.no_grad():
        out_masked = attn(x, key_padding_mask=pad_mask)

        x2 = x.clone()
        x2[:, -2:, :] += 100.0   # perturb only the PAD positions' input
        out_perturbed = attn(x2, key_padding_mask=pad_mask)

    # real (non-pad) positions must be completely unaffected by changing PAD content
    assert torch.allclose(out_masked[:, :-2, :], out_perturbed[:, :-2, :], atol=1e-6), \
        "PAD token content leaked into real tokens' attention output"