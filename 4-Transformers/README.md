# Transformer-based GPT Model

A PyTorch implementation of a GPT-style transformer model for character-level language modeling.

## Overview

This project implements a generative pre-trained transformer (GPT) model from scratch using PyTorch. The model is trained on character-level sequences and can generate new text based on learned patterns in the training data.

## Project Structure

```
4-Transformers/
├── transformer/          # Core transformer implementation
│   ├── model.py         # GPT model architecture
│   ├── config.py        # Configuration dataclass
│   ├── block.py         # Transformer block (attention + FFN)
│   ├── attention.py     # Multi-head self-attention
│   ├── feedforward.py   # Feed-forward network
│   └── data.py          # Data loading and preprocessing
├── scripts/
│   ├── train.py         # Training script
│   └── generate.py      # Text generation script
├── tests/
│   └── test_attention.py # Unit tests
└── README.md
```

## Components

### Transformer Architecture

**GPT Model** (`model.py`):
- Token and position embeddings
- Stack of transformer blocks
- Output layer for token predictions
- Support for language modeling with optional padding masks

**Transformer Block** (`block.py`):
- Multi-head self-attention
- Feed-forward network
- Layer normalization and residual connections

**Multi-Head Attention** (`attention.py`):
- Multiple attention heads in parallel
- Causal masking to prevent looking at future tokens
- Dropout for regularization
- Optional key padding mask support

**Feed-Forward Network** (`feedforward.py`):
- Two-layer feed-forward network with ReLU activation
- Expansion and projection layers
- Dropout regularization

### Configuration

The model is configured using the `GPTConfig` dataclass, which specifies:
- `vocab_size`: Size of the vocabulary
- `block_size`: Maximum context length (default: 128)
- `n_embd`: Embedding dimension (default: 32)
- `n_head`: Number of attention heads (default: 4)
- `n_layer`: Number of transformer blocks (default: 2)
- `dropout`: Dropout rate (default: 0.1)
- `pad_index`: Token index for padding (default: 27)
- `num_epochs`: Training epochs (default: 10)

## Usage

### Training

Train the model on a text file (character-level):

```bash
python -m scripts.train
```

The training script:
- Loads character sequences from `names.txt`
- Creates train/dev/test splits
- Trains the GPT model with AdamW optimizer
- Evaluates on validation set after each epoch

### Text Generation

Generate new text sequences after training:

```bash
python -m scripts.generate
```

The generation script uses the trained model to:
- Start with a special start token (`<S>`)
- Iteratively sample the next token
- Continue until reaching a maximum length or end token

## Data Format

The model expects a plain text file where:
- Each line represents one training sequence
- Characters are tokenized individually
- Special tokens are added: `<S>` (start) and `<PAD>` (padding)

Example `names.txt`:
```
alice
bob
charlie
...
```

## Model Features

- **Character-level tokenization**: Works with individual characters
- **Causal masking**: Prevents looking at future tokens during training
- **Padding support**: Handles variable-length sequences with padding masks
- **Multi-head attention**: Parallel attention heads for rich feature extraction
- **Dropout regularization**: Prevents overfitting during training
- **Scalable architecture**: Easily adjustable depth and width

## Token Vocabulary

The vocabulary is automatically constructed from the training data:
- Characters: `a-z` (indices 0-25)
- Special tokens:
  - `<S>`: Start token (index 26)
  - `<PAD>`: Padding token (index 27)

## Testing

Run unit tests to verify attention mechanism:

```bash
pytest tests/test_attention.py
```

## Example Output

### Generated Names

After training the model, it can generate realistic-looking names:

```
dala
tanefen
lya
anyla
sulin
mayla
```

### Training Results

Training performance on character-level language modeling:

```
Epoch [10/10] train loss: 2.0197
Epoch [10/10] val loss:   2.0185
```

The model converges well with training and validation loss very close together, indicating good generalization.

## Dependencies

- PyTorch

## Notes

- The model is trained for character-level language modeling
- Adjust `GPTConfig` parameters to control model size and training
- The `block_size` parameter limits the context window the model can see
- Larger `n_head` and `n_layer` increase model capacity but also training time

## Key Implementation Details

1. **Positional Encoding**: Uses learned position embeddings
2. **Attention Mechanism**: Implements scaled dot-product attention with proper masking
3. **Gradient Flow**: Residual connections and layer normalization enable training of deeper models
4. **Efficient Batching**: Data loader handles variable-length sequences with padding
