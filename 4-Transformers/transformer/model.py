import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.block import Block

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
    
    def forward(self, idx, targets=None, pad_index=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # token embeddings of shape (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # position embeddings of shape (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        
        key_padding_mask = (idx == pad_index) if pad_index is not None else None  # (B, T) bool
        
        for block in self.blocks:
            x = block(x, key_padding_mask)
        x = self.ln_f(x) # final layer norm
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        loss = None
        if targets is not None:
            # reshape for cross-entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=self.config.pad_index)
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:] # crop context to block size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # focus on last time step
            probs = F.softmax(logits, dim=-1) # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to sequence
        return idx