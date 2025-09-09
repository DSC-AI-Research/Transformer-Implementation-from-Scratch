"""
Positional Encoding Implementation

Since Transformers don't have inherent sequence order (unlike RNNs),
we need to inject positional information into the input embeddings.

The paper uses sine and cosine functions of different frequencies:
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        """
        Args:
            d_model: The dimension of the model (embedding size)
            max_length: Maximum sequence length we expect
        """
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix to store positional encodings
        pe = torch.zeros(max_length, d_model)
        
        # Create position column [0, 1, 2, ..., max_length-1]
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # Create the division term for frequency calculation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings [seq_length, batch_size, d_model]
        
        Returns:
            x + positional encodings
        """
        return x + self.pe[:x.size(0), :]