# NLP Learning Journey

A comprehensive self-directed learning repository documenting my exploration of Natural Language Processing, from foundational concepts to modern architectures. This repo contains implementations, experiments, and educational notebooks covering the my evolution in NLP techniques.

## Repository Structure

This repository is organized as a progressive learning path, building from fundamental concepts to advanced architectures:

### 1. Learned Representations

**Directory**: `1-learned representations/`

Introduction to how neural networks learn meaningful text representations.

- **Byte Pair Encoding** (`byte_pair_encoding.ipynb`): 
  - Tokenization technique that recursively merges frequent character pairs
  - Foundation for subword tokenization in modern NLP models
  
- **Neural Bag-of-Words** (`Neural BoW/`):
  - Simple neural network approach to text representation
  - Demonstrates how word order-invariant embeddings work
  - Includes vocabulary files and training data

### 2. Auto-Regressive Language Modeling

**Directory**: `2-auto_regressive language modeling/`

Building language models that predict the next token based on previous context.

- **Bigrams** (`bigrams.ipynb`): 
  - Statistical approach to language modeling using two-word sequences
  - Baseline for more complex models
  
- **Neural Bigrams** (`neural_bigrams.ipynb`): 
  - Neural network implementation of bigram language modeling
  - Bridges statistical and deep learning approaches
  
- **Neural Language Model** (`neural_lm.ipynb`): 
  - More sophisticated neural language model with embeddings and hidden layers
  - Demonstrates how neural networks can learn language patterns

### 3. Recurrent Neural Networks

**Directory**: `3-recurrent neural networks/`

Exploring sequential models that maintain hidden state across time steps.

- **RNN** (`RNN.ipynb`): 
  - Fundamentals of recurrent neural networks
  - How RNNs process sequences and maintain context
  
- **Attention Mechanism** (`attention.ipynb`): 
  - Introduction to attention: allowing models to focus on relevant parts of input
  - Bridge between RNNs and transformers

### 4. Transformers

**Directory**: `4-Transformers/`

Modern architecture based on self-attention mechanisms. This is a full production-quality implementation.

- **Multi-Head Attention** (`multihead_attention.ipynb`): 
  - Detailed exploration of parallel attention heads
  - Key component of transformer models
  
- **Transformer Implementation** (`transformer/`):
  - Complete GPT-style transformer implementation from scratch
  - Includes model architecture, training, and inference scripts
  - Full documentation in [4-Transformers/README.md](4-Transformers/README.md)

- **Scripts**:
  - `train.py`: Train transformer on character-level text
  - `generate.py`: Generate new text using trained model
  
- **Tests**:
  - `test_attention.py`: Unit tests for attention mechanism

## Learning Path

This repository follows the natural progression of NLP research and development:

```
Learned Representations 
    ↓
Auto-Regressive Language Modeling
    ↓
Recurrent Neural Networks
    ↓
Transformers (Modern State-of-the-Art)
```

Each section builds upon concepts from previous ones:
- **Representations** teach how tokenization libraries are working under the hood
- **Language Modeling** shows how to predict sequences
- **RNNs** introduce sequential processing with memory
- **Transformers** replace recurrence with pure attention mechanisms

## Key Concepts Covered

### Tokenization & Embeddings
- Bag‑of‑words and subword tokenization methods (BPE, SentencePiece).
- Character-level and word-level embeddings
- Learned vs. static representations

### Language Modeling
- Probabilistic prediction of next tokens
- From n-gram statistics to neural approaches
- Loss functions and evaluation metrics

### Sequential Processing
- Recurrent architectures and their variants
- Vanishing/exploding gradients
- Attention as a solution to sequence limitations

### Modern Architectures
- Self-attention mechanisms
- Multi-head attention for parallel processing
- Positional encodings for sequence information
- Layer normalization and residual connections
- Transformer blocks and stacking

## Technologies Used

- **Python**: Primary programming language
- **PyTorch**: Deep learning framework
- **Jupyter Notebooks**: Interactive learning and experimentation

## Learning Outcomes

Through building and experimenting with these models, I've gained understanding of:

1. How text is represented as numerical vectors
2. Fundamentals of statistical language modeling
3. How RNNs process sequential information
4. The attention mechanism and why it's powerful
5. Complete transformer architecture design
6. Model training, evaluation, and inference
7. Practical challenges in NLP (tokenization, padding, batching)

## Notes

- Each section includes both theoretical understanding (via notebooks) and practical implementation
- The progression from simple to complex mirrors the historical development of NLP
- Code is written for educational clarity, with extensive comments
- Experiments are documented with results and insights

## References

The implementations in this repository are guided by foundational and modern papers in NLP:

### Learned Representations
- Cho 2015 (Ch.2, Ch.3) - [Natural Language Understanding with Distributed Representation](https://arxiv.org/abs/1511.07916)

### Auto-Regressive Language Modeling
- Cho 2015 (Ch.5 up to 5.4.2) - [Natural Language Understanding with Distributed Representation](https://arxiv.org/abs/1511.07916)
- Bengio et al. (2003) - [Neural Probabilistic Language Models](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- Glorot & Bengio (2010) - [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

### Recurrent Neural Networks & Attention
- Cho 2015 (Ch.4, Ch.5.5-5.6, Ch.6) - [Natural Language Understanding with Distributed Representation](https://arxiv.org/abs/1511.07916)
- Mikolov et al. (2010) - [Recurrent Neural Network based Language Model](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.pdf)
- Cho et al. (2014) - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078)
- Bahdanau et al. (2015) - [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Weber (2017) - [Why LSTMs Stop Your Gradients From Vanishing: A View from the Backwards Pass](https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html)

### Transformers
- Vaswani et al. (2017) - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- Ba et al. (2016) - [Layer Normalization](https://arxiv.org/pdf/1607.06450)

---

This repository represents a hands-on learning journey through modern NLP. Each implementation is built from scratch to develop deep understanding rather than relying on pre-built libraries for core components.