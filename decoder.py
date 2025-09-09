"""
Transformer Decoder Implementation

The decoder is also composed of a stack of N identical layers.
Each layer has three sub-layers:
1. Masked multi-head self-attention
2. Multi-head attention over encoder output (cross-attention)
3. Position-wise fully connected feed-forward network

Each sub-layer has a residual connection around it, followed by layer normalization.
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Single decoder layer
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed forward dimension
            dropout: Dropout probability
        """
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass through decoder layer
        
        Args:
            x: Decoder input [batch_size, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch_size, src_seq_len, d_model]
            src_mask: Source sequence mask
            tgt_mask: Target sequence mask (causal mask)
        
        Returns:
            Output tensor [batch_size, tgt_seq_len, d_model]
        """
        # Masked self-attention
        self_attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attention_output))
        
        # Cross-attention with encoder output
        cross_attention_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attention_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Stack of decoder layers
        
        Args:
            num_layers: Number of decoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed forward dimension
            dropout: Dropout probability
        """
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass through all decoder layers
        
        Args:
            x: Decoder input [batch_size, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch_size, src_seq_len, d_model]
            src_mask: Source sequence mask
            tgt_mask: Target sequence mask
        
        Returns:
            Output tensor [batch_size, tgt_seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x