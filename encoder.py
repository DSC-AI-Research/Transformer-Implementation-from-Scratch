"""
Transformer Encoder Implementation

The encoder is composed of a stack of N identical layers.
Each layer has two sub-layers:
1. Multi-head self-attention mechanism
2. Position-wise fully connected feed-forward network

We employ a residual connection around each of the two sub-layers,
followed by layer normalization.
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Single encoder layer
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed forward dimension
            dropout: Dropout probability
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass with residual connections and layer normalization
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Stack of encoder layers
        
        Args:
            num_layers: Number of encoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed forward dimension
            dropout: Dropout probability
        """
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        """
        Forward pass through all encoder layers
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return x