import torch
import torch.nn as nn
from block import Block

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
    
    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # token embeddings of shape (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # position embeddings of shape (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # apply transformer blocks
        x = self.ln_f(x) # final layer norm
        logits = self.lm_head(x) # (B,T,vocab_size)
        return logits