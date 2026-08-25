import torch.nn as nn
from transformer.feedforward import FeedForward
from transformer.attention import MultiHeadAttention

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # pre-norm
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        
    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.ln2(x))
        return x