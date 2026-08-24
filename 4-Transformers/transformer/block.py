import torch.nn as nn
from feedforward import FeedForward
from attention import MultiHeadAttention

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # pre-norm
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x