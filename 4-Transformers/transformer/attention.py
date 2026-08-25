import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        
    def forward(self, x, key_padding_mask=None):
        _,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)
        # compute attention scores ("affinities")
        weight = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,head_size) @ (B,head_size,T) ---> (B,T,T)
        weight = weight.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        
        if key_padding_mask is not None:
            # (B, T) -> (B, 1, T) so it broadcasts over the query dimension,
            # masking out PAD *keys* regardless of which query position is attending
            mask = key_padding_mask[:, None, :]
            weight = weight.masked_fill(mask, float('-inf'))
        
        weight = F.softmax(weight, dim=-1) # (B,T,T)
        weight = self.dropout(weight)
        # perform the weighted aggregation of the values
        out = weight @ v # (B,T,T) @ (B,T,head_size) ---> (B,T,head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, key_padding_mask=None):
        out = torch.cat([h(x, key_padding_mask) for h in self.heads], dim=-1) # (B,T,C)
        out = self.dropout(self.proj(out)) 
        return out 