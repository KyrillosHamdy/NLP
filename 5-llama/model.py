import torch
import torch.nn as nn
from typing import Optional
from torch.nn import RMSNorm, functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    vocab_size: int = -1
    dim: int = 4092
    n_layer: int = 32
    n_heads: int = 32
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    max_seq_len: int = 2048
    norm_eps: float = 1e-5
    device: str

def precompute_freqs_cis(head_dim: int, max_seq_len: int, device: str, theta: float = 10000.0):
    """
    Precompute the complex frequencies for rotary embeddings.
    """
    assert head_dim % 2 == 0, "Head dimension must be even for complex numbers."
    
    # shape: (head_dim // 2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
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
    x_out = x_out.reshape(*x_out.shape)
    return x_out.type_as(x).to(device)


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
    pass


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
        Initialize an llama encoder block consisting of a self-attention layer followed by a feed-forward layer.
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
        self.layers = nn.ModuleList([EncoderBlock(self.args) for _ in range(self.args.n_layer)])
        self.norm = RMSNorm(self.args.dim, self.args.norm_eps)
        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=False)

        self.freqs_complex = precompute_freqs_cis(
                        self.args.dim // self.args.n_head, 
                        self.args.max_seq_len, 
                        end=self.args.max_seq_len, 
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
        