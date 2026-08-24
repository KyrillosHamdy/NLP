import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

    def forward(self, x):
        x = self.net(x)
        return x