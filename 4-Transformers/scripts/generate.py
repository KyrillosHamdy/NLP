import torch
from transformer.config import GPTConfig
from transformer.model import GPT
from transformer.data import get_dataloaders

token_to_index, index_to_token, _, _, _ = get_dataloaders('.\\imports\\data\\names.txt', batch_size=16)

ckpt = torch.load('.\\imports\\checkpoints\\checkpoint.pt', weights_only=False)
model = GPT(ckpt['config'])
model.load_state_dict(ckpt['model_state_dict'])

def sample(model, token_to_index, index_to_token, prompt='', max_length=100):
    start_idx = token_to_index['<S>']
    context = [start_idx] + [token_to_index[c] for c in prompt]
    idx = torch.tensor([context], dtype=torch.long)

    out = model.generate(idx, max_new_tokens=max_length)
    ids = out[0, len(context):].tolist()

    tokens = []
    for i in ids:
        if i == start_idx:
            break
        tokens.append(index_to_token[i])
    return ''.join(tokens)

for _ in range(10):
    print(sample(model, token_to_index, index_to_token))