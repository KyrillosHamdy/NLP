import torch
from torch.utils.data import Dataset, DataLoader

def build_vocab():
    token_to_index = {tok: i for i, tok in enumerate('abcdefghijklmnopqrstuvwxyz')}
    token_to_index['<S>'] = 26
    token_to_index['<PAD>'] = 27
    index_to_token = {i: tok for tok, i in token_to_index.items()}
    return token_to_index, index_to_token

def build_examples(names, char_to_idx):
    """One example per name: <S> + chars + <S>, shifted by one for (x, y)."""
    
    X, Y = [], []
    for name in names:
        tokens = ['<S>'] + list(name) + ['<S>']
        indices = [char_to_idx[ch] for ch in tokens]
        X.append(indices[:-1])
        Y.append(indices[1:])
    return X, Y

def pad_batch(X_batch, Y_batch, pad_index):
    max_len = max(len(x) for x in X_batch)
    X_padded = torch.full((len(X_batch), max_len), pad_index, dtype=torch.long)
    Y_padded = torch.full((len(Y_batch), max_len), pad_index, dtype=torch.long)
    for i, (x, y) in enumerate(zip(X_batch, Y_batch)):
        X_padded[i, :len(x)] = torch.tensor(x)
        Y_padded[i, :len(y)] = torch.tensor(y)
    return X_padded, Y_padded

class NameDataset(Dataset):
    """Returns raw (unpadded) token id lists for each name, to be padded in the collate_fn."""
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
def make_collate_fn(pad_index):
    """Returns a collate_fn that pads a batch of (X, Y) pairs."""
    def collate(batch):
        X_batch, Y_batch = zip(*batch)
        return pad_batch(X_batch, Y_batch, pad_index)
    return collate

def get_dataloaders(names_path, batch_size=16, val_frac=0.1, test_frac=0.1, seed=123):
    import random
    token_to_index, index_to_token = build_vocab()

    names = open(names_path).read().splitlines()
    random.seed(seed)
    random.shuffle(names)

    n1 = int((1 - val_frac - test_frac) * len(names))
    n2 = int((1 - test_frac) * len(names))
    train_names, dev_names, test_names = names[:n1], names[n1:n2], names[n2:]

    X_train, Y_train = build_examples(train_names, token_to_index)
    X_dev, Y_dev = build_examples(dev_names, token_to_index)
    X_test, Y_test = build_examples(test_names, token_to_index)

    collate = make_collate_fn(token_to_index['<PAD>'])
    train_loader = DataLoader(NameDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(NameDataset(X_dev, Y_dev), batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(NameDataset(X_test, Y_test), batch_size=batch_size, shuffle=False, collate_fn=collate)

    return token_to_index, index_to_token, train_loader, dev_loader, test_loader