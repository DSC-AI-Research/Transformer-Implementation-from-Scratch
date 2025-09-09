# Transformer Implementation from Scratch

This repository contains a complete implementation of the Transformer architecture from the paper "Attention is All You Need" by Vaswani et al. (2017). The implementation is designed to be beginner-friendly with extensive comments explaining each component.

## Architecture Overview

The Transformer is a sequence-to-sequence model that relies entirely on attention mechanisms, dispensing with recurrence and convolutions. It consists of:

### Core Components

1. **Multi-Head Attention** (`attention.py`)
   - Scaled dot-product attention
   - Multiple attention heads for different representation subspaces
   - Self-attention and cross-attention mechanisms

2. **Positional Encoding** (`positional_encoding.py`)
   - Sine and cosine functions to inject positional information
   - No learnable parameters, fixed mathematical encoding

3. **Position-wise Feed Forward** (`feed_forward.py`)
   - Two linear transformations with ReLU activation
   - Applied to each position separately and identically

4. **Encoder** (`encoder.py`)
   - Stack of N identical layers
   - Each layer: Multi-head self-attention + Feed forward
   - Residual connections and layer normalization

5. **Decoder** (`decoder.py`)
   - Stack of N identical layers  
   - Each layer: Masked self-attention + Cross-attention + Feed forward
   - Residual connections and layer normalization

6. **Complete Model** (`transformer.py`)
   - Combines encoder and decoder
   - Input/output embeddings
   - Final linear projection to vocabulary

## Key Features from the Paper

### Attention Mechanism
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

### Multi-Head Attention
- 8 attention heads (h=8)
- Each head has dimension d_k = d_v = d_model/h = 64
- Allows model to attend to different positions and representation subspaces

### Positional Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Model Architecture
- 6 layers in both encoder and decoder (N=6)
- Model dimension d_model = 512
- Feed-forward dimension d_ff = 2048
- 8 attention heads
- Dropout rate = 0.1

## File Structure

```
├── attention.py           # Multi-head attention implementation
├── positional_encoding.py # Positional encoding
├── feed_forward.py        # Position-wise feed forward network
├── encoder.py            # Encoder stack
├── decoder.py            # Decoder stack
├── transformer.py        # Complete Transformer model
├── utils.py              # Utility functions
├── training_example.py   # Example training script
└── README.md            # This file
```

## Usage

### Basic Model Creation

```python
from transformer import create_transformer

# Create model with vocabulary sizes
model = create_transformer(src_vocab_size=1000, tgt_vocab_size=1000)

# Count parameters
from utils import count_parameters
print(f"Model has {count_parameters(model):,} parameters")
```

### Training Example

```python
# See training_example.py for complete training script
python training_example.py
```

### Inference Example

```python
from utils import greedy_decode

# Assuming you have a trained model and input
src = torch.tensor([[1, 2, 3, 4, 5]])  # Source sequence
src_mask = model.make_src_mask(src)

# Generate translation
output = greedy_decode(
    model, src, src_mask, 
    max_len=50, start_symbol=1, end_symbol=2
)
```

## Key Concepts Explained

### Self-Attention vs Cross-Attention

- **Self-Attention**: Query, Key, and Value come from the same sequence
  - Used in encoder for input sequence
  - Used in decoder for output sequence (with masking)

- **Cross-Attention**: Query from decoder, Key and Value from encoder
  - Allows decoder to attend to encoder representations

### Masking

1. **Padding Mask**: Prevents attention to padding tokens
2. **Look-ahead Mask**: Prevents decoder from attending to future positions

### Residual Connections and Layer Normalization

Each sub-layer uses:
```
output = LayerNorm(x + Sublayer(x))
```

This helps with:
- Gradient flow during training
- Model stability
- Faster convergence

## Training Details

### Learning Rate Schedule
```python
lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

### Label Smoothing
- Reduces overfitting
- Improves generalization
- Standard in modern NLP

### Optimization
- Adam optimizer with β1=0.9, β2=0.98, ε=1e-9
- Gradient clipping for stability
- Warmup steps for learning rate

## Dependencies

```
torch >= 1.7.0
numpy
```

## Paper Reference

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## Understanding the Code

This implementation prioritizes clarity and educational value:

1. **Extensive Comments**: Each function and important line is commented
2. **Modular Design**: Each component is in its own file for clarity
3. **Clear Variable Names**: Self-documenting code style
4. **Type Hints**: Tensor shapes specified in comments
5. **Educational Examples**: Complete training example included

## Extensions and Modifications

This basic implementation can be extended with:

- Relative positional encoding
- Different attention patterns (sparse, local, etc.)
- Layer-wise learning rate decay
- Different normalization schemes
- Encoder-only or decoder-only variants

The modular design makes it easy to experiment with different components while maintaining the core architecture.