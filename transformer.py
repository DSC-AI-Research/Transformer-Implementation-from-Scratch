"""
Complete Transformer Model Implementation

This file brings together all components to create the full Transformer model
as described in "Attention is All You Need" paper.

The Transformer follows an encoder-decoder architecture using stacked
self-attention and point-wise, fully connected layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
from positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_seq_length=5000, dropout=0.1):
        """
        Complete Transformer model
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension (512 in paper)
            num_heads: Number of attention heads (8 in paper)
            num_layers: Number of encoder/decoder layers (6 in paper)
            d_ff: Feed forward dimension (2048 in paper)
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Encoder and Decoder
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # Final linear layer to project to vocabulary
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.init_parameters()
    
    def init_parameters(self):
        """Initialize parameters with Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src):
        """
        Create mask for source sequence (to mask padding tokens)
        
        Args:
            src: Source sequence [batch_size, src_seq_len]
        
        Returns:
            Source mask [batch_size, 1, 1, src_seq_len]
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        """
        Create mask for target sequence (causal mask + padding mask)
        
        Args:
            tgt: Target sequence [batch_size, tgt_seq_len]
        
        Returns:
            Target mask [batch_size, 1, tgt_seq_len, tgt_seq_len]
        """
        tgt_seq_len = tgt.size(1)
        
        # Create causal mask (lower triangular matrix)
        tgt_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len))).expand(
            tgt.size(0), 1, tgt_seq_len, tgt_seq_len
        )
        
        # Create padding mask
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        
        # Combine masks
        tgt_mask = tgt_mask & tgt_padding_mask
        
        return tgt_mask.to(tgt.device)
    
    def forward(self, src, tgt):
        """
        Forward pass of the Transformer
        
        Args:
            src: Source sequence [batch_size, src_seq_len]
            tgt: Target sequence [batch_size, tgt_seq_len]
        
        Returns:
            Output logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Create masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Encode source sequence
        src_embedded = self.dropout(self.positional_encoding(
            self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        ))
        encoder_output = self.encoder(src_embedded, src_mask)
        
        # Decode target sequence
        tgt_embedded = self.dropout(self.positional_encoding(
            self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        ))
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.linear(decoder_output)
        
        return output


def create_transformer(src_vocab_size, tgt_vocab_size):
    """
    Create a Transformer model with default hyperparameters from the paper
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
    
    Returns:
        Transformer model
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,      # Model dimension
        num_heads=8,      # Number of attention heads
        num_layers=6,     # Number of encoder/decoder layers
        d_ff=2048,        # Feed forward dimension
        max_seq_length=5000,
        dropout=0.1
    )
    
    return model