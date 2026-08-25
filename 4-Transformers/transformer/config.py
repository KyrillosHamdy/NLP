from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int 
    block_size: int = 128
    n_embd: int = 32
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    head_size: int = None
    pad_index: int = 27
    num_epochs: int = 10
    
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        if self.head_size is None:
            self.head_size = self.n_embd // self.n_head