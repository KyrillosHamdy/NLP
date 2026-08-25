import torch
import torch.nn as nn
import torch.optim as optim
from transformer.config import GPTConfig
from transformer.model import GPT
from transformer.data import get_dataloaders

token_to_index, index_to_token, train_loader, dev_loader, _ = get_dataloaders('names.txt', batch_size=16)

config = GPTConfig(
    vocab_size=len(token_to_index),
    block_size=32, 
    n_embd=64,
    n_head=4,
    n_layer=2,
    dropout=0.1,
    pad_index=token_to_index['<PAD>'],
    num_epochs=10
)

model = GPT(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(config.num_epochs):
    model.train()
    total_loss = 0.0
    
    for X_batch, Y_batch in train_loader:
        logits, loss = model(X_batch, targets=Y_batch) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{config.num_epochs}] train loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in dev_loader:
            _, loss = model(X_batch, targets=Y_batch)
            eval_loss += loss.item()
    print(f"Epoch [{epoch+1}/{config.num_epochs}] val loss:   {eval_loss/len(dev_loader):.4f}")