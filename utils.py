"""
Utility functions for the Transformer model

This file contains helper functions for training, inference, and
working with the Transformer model.
"""

import torch
import torch.nn.functional as F
import math


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_padding_mask(seq, pad_idx=0):
    """
    Create padding mask for sequences
    
    Args:
        seq: Input sequence [batch_size, seq_len]
        pad_idx: Padding token index
    
    Returns:
        Padding mask [batch_size, 1, 1, seq_len]
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    """
    Create look-ahead mask for decoder self-attention
    
    Args:
        size: Sequence length
    
    Returns:
        Look-ahead mask [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def learning_rate_schedule(d_model, step_num, warmup_steps=4000):
    """
    Learning rate schedule from the paper
    
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    
    Args:
        d_model: Model dimension
        step_num: Current step number
        warmup_steps: Number of warmup steps
    
    Returns:
        Learning rate
    """
    arg1 = step_num ** (-0.5)
    arg2 = step_num * (warmup_steps ** (-1.5))
    
    return (d_model ** (-0.5)) * min(arg1, arg2)


def label_smoothing_loss(pred, target, smoothing=0.1, pad_idx=0):
    """
    Label smoothing loss function
    
    Args:
        pred: Predictions [batch_size, seq_len, vocab_size]
        target: Target labels [batch_size, seq_len]
        smoothing: Smoothing factor
        pad_idx: Padding token index
    
    Returns:
        Loss value
    """
    vocab_size = pred.size(-1)
    
    # Create one-hot encoding
    one_hot = torch.zeros_like(pred).scatter(-1, target.unsqueeze(-1), 1)
    
    # Apply label smoothing
    smooth_one_hot = one_hot * (1 - smoothing) + smoothing / vocab_size
    
    # Calculate loss
    log_prob = F.log_softmax(pred, dim=-1)
    loss = -smooth_one_hot * log_prob
    
    # Mask padding tokens
    mask = (target != pad_idx).float()
    loss = loss.sum(dim=-1) * mask
    
    return loss.sum() / mask.sum()


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    """
    Greedy decoding for inference
    
    Args:
        model: Trained Transformer model
        src: Source sequence [1, src_len]
        src_mask: Source mask
        max_len: Maximum generation length
        start_symbol: Start token index
        end_symbol: End token index
    
    Returns:
        Generated sequence
    """
    model.eval()
    
    # Encode source
    memory = model.encoder(model.positional_encoding(
        model.src_embedding(src) * math.sqrt(model.d_model)
    ), src_mask)
    
    # Initialize target with start symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len - 1):
        # Create target mask
        tgt_mask = model.make_tgt_mask(ys)
        
        # Decode
        out = model.decoder(
            model.positional_encoding(
                model.tgt_embedding(ys) * math.sqrt(model.d_model)
            ),
            memory, src_mask, tgt_mask
        )
        
        # Get next token probabilities
        prob = model.linear(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        # Add to sequence
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        # Stop if end symbol is generated
        if next_word == end_symbol:
            break
    
    return ys