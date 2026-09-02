import torch
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_freqs_cis(head_dim: int, max_seq_len: int, device: str, theta: float = 10000.0):
    """
    Precompute the complex frequencies for rotary embeddings.
    """
    assert head_dim % 2 == 0, "Head dimension must be even for complex numbers."
    
    # shape: (head_dim // 2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device)
    # shape: (seq_len, head_dim // 2)
    freqs = torch.outer(t, freqs)
    # precompute complex numbers in polar form: exp(i * freqs) = cos(freqs) + i * sin(freqs)
    complex_freqs = torch.polar(torch.ones_like(freqs), freqs)
    return complex_freqs

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """
    Apply rotary embeddings to the input tensor.
    """
    original_shape = x.shape
    # shape: (batch_size, seq_len, n_head, head_dim) -> (batch_size, seq_len, n_head, head_dim // 2, 2)
    x = x.float().reshape(*x.shape[:-1], -1, 2)
    # shape: (batch_size, seq_len, n_head, head_dim // 2, 2) -> (batch_size, seq_len, n_head, head_dim // 2)
    x_complex = torch.view_as_complex(x)
    # shape: (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2) to match x_complex shape
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    # shape: (batch_size, seq_len, n_head, head_dim // 2) -> (batch_size, seq_len, n_head, head_dim // 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # shape: (batch_size, seq_len, n_head, head_dim // 2, 2) -> (batch_size, seq_len, n_head, head_dim)
    x_out = x_out.reshape(*original_shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_reps: int):
    """
    Repeat the keys and values for multi-head attention.
    """
    batch_size, seq_len, n_kv_heads, head_dim = x.size()
    if n_reps == 1:
        return x    
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_reps, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_reps, head_dim)
    ) 

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Initialize a learnable scale parameter for RMS normalization (gamma parameter)
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        """
        Compute the RMS normalization of the input tensor.
        """
        
        # Compute the root mean square (RMS) of the input tensor along the last dimension
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # shape: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_embd)
        x_norm = x * rms
        return x_norm

    def forward(self, x: torch.Tensor):
        """
        Apply RMS normalization to the input tensor.
        """
        
        # shape: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_embd)
        x_norm = self._norm(x.float()).type_as(x)
        return x_norm * self.scale


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.n_heads_q = args.n_heads
        # how many times keys and values should be repeated
        self.n_reps = self.n_heads_q // self.n_kv_heads
        
        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)

    def forward(self, x: torch.Tensor, start_pos: int, complex_freqs: torch.Tensor):
        """
        Forward pass through the self-attention layer.
        """
        
        batch_size, seq_len, _ = x.size()
        
        # shape: (batch_size, seq_len, n_heads_q * head_dim) -> (batch_size, seq_len, n_heads_q, head_dim)
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads * head_dim) -> (batch_size, seq_len, n_kv_heads, head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads * head_dim) -> (batch_size, seq_len, n_kv_heads, head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # apply RoPE to q and k
        # shape: (batch_size, 1, n_heads_q, head_dim) -> (batch_size, 1, n_heads_q, head_dim)
        xq = apply_rotary_embeddings(xq, complex_freqs, x.device)
        # shape: (batch_size, 1, n_kv_heads, head_dim) -> (batch_size, 1, n_kv_heads, head_dim)
        xk = apply_rotary_embeddings(xk, complex_freqs, x.device)
        
        # update the cache with the new keys and values
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv
        # retrieve the cached keys and values for the current sequence
        # shape: (batch_size, total_seq_len, n_kv_heads, head_dim)
        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]
        
        # repeat the heads for k and v to match the number of q heads
        # shape: (batch_size, total_seq_len, n_heads_q, head_dim)
        keys = repeat_kv(keys, self.n_reps)
        values = repeat_kv(values, self.n_reps)
        
        keys = keys.transpose(1, 2)  # shape: (batch_size, n_heads_q, total_seq_len, head_dim)
        values = values.transpose(1, 2)  # shape: (batch_size, n_heads_q, total_seq_len, head_dim)
        xq = xq.transpose(1, 2)  # shape: (batch_size, n_heads_q, seq_len, head_dim)
        
        # compute the scores for the attention mechanism using scaled dot-product attention
        # shape: (batch_size, n_heads_q, seq_len = 1, total_seq_len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / (self.head_dim ** 0.5)
        scores = F.softmax(scores, dim=-1)
        # multiplying with v 
        # shape: (batch_size, n_heads_q, seq_len = 1, head_dim)
        output = torch.matmul(scores, values)
        # we need to match the shape of the next layer
        # shape: (batch_size, seq_len = 1, n_heads_q * head_dim)  
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) 
        return self.wo(output) # shape: (batch_size, 1, dim)
         
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        """
        Initialize a llama encoder block 
        consisting of a self-attention layer followed by a feed-forward layer.
        """
        
        self.args = args
        self.head_dim = self.args.dim // self.args.n_heads
        # normalization right before attention layer
        self.attention_norm = RMSNorm(self.args.dim, self.args.norm_eps)
        self.attention = SelfAttention(self.args)
        # normalization right before feed forward layer
        self.ffn_norm = RMSNorm(self.args.dim, self.args.norm_eps)
        self.feed_forward = FeedForward(self.args)
    
    def forward(self, x: torch.Tensor, start_pos: int, complex_freqs: torch.Tensor):
        """
        Forward pass through the llama encoder block.
        """
        
        # shape: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_embd)
        h = x + self.attention(self.attention_norm(x), start_pos, complex_freqs)
        # shape: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_embd)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.dim)
        self.layers = nn.ModuleList([EncoderBlock(self.args) for _ in range(self.args.n_layers)])
        self.norm = RMSNorm(self.args.dim, self.args.norm_eps)
        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=False)

        self.freqs_complex = precompute_freqs_cis(
                        self.args.dim // self.args.n_heads, 
                        self.args.max_seq_len, 
                        device=self.args.device
        )
    
    def forward(self, tokens: torch.tensor, start_pos: int):
        batch_size, seq_len = tokens.size()
        assert seq_len == 1, "Only one token at a time"
        
        # (B, seq_len) -> (B, seq_len, n_embd)
        h = self.embedding(tokens)
        
        # retrieve (m, theta) for positions (start_pos, start_pos + seq_len)
        complex_freqs = self.freqs_complex[start_pos:start_pos + seq_len]
        
        # stack the layers
        for layer in self.layers:
            h = layer(h, start_pos, complex_freqs)
        h = self.norm(h)
        output = self.output(h)
        return output
        