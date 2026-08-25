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