"""
Example training script for the Transformer model

This demonstrates how to use the Transformer model for a sequence-to-sequence task.
This is a simplified example - real training would require proper data loading,
validation, checkpointing, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer import create_transformer
from utils import learning_rate_schedule, label_smoothing_loss, count_parameters
import math


class DummyDataset(Dataset):
    """
    Dummy dataset for demonstration purposes
    In practice, you would load real data here
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, num_samples=1000, max_len=50):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_samples = num_samples
        self.max_len = max_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequences
        src_len = torch.randint(10, self.max_len, (1,)).item()
        tgt_len = torch.randint(10, self.max_len, (1,)).item()
        
        src = torch.randint(1, self.src_vocab_size, (src_len,))
        tgt = torch.randint(1, self.tgt_vocab_size, (tgt_len,))
        
        return src, tgt


def collate_fn(batch):
    """Collate function to handle variable length sequences"""
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_lengths = [len(seq) for seq in src_batch]
    tgt_lengths = [len(seq) for seq in tgt_batch]
    
    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)
    
    # Pad with zeros (assuming 0 is padding token)
    src_padded = torch.zeros(len(src_batch), max_src_len, dtype=torch.long)
    tgt_padded = torch.zeros(len(tgt_batch), max_tgt_len, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(batch):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt
    
    return src_padded, tgt_padded


def train_transformer():
    """
    Example training function
    """
    # Model parameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    
    # Create model
    model = create_transformer(src_vocab_size, tgt_vocab_size)
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Create dataset and dataloader
    dataset = DummyDataset(src_vocab_size, tgt_vocab_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # Optimizer (using Adam with custom learning rate schedule)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    
    # Training loop
    model.train()
    step = 0
    
    for epoch in range(10):  # Train for 10 epochs
        epoch_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            step += 1
            
            # Update learning rate
            lr = learning_rate_schedule(d_model, step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Prepare target input and output
            tgt_input = tgt[:, :-1]  # All tokens except last
            tgt_output = tgt[:, 1:]  # All tokens except first
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Calculate loss
            loss = label_smoothing_loss(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, LR: {lr:.6f}')
        
        print(f'Epoch {epoch} completed. Average loss: {epoch_loss / len(dataloader):.4f}')
    
    return model


if __name__ == "__main__":
    # Train the model
    trained_model = train_transformer()
    
    # Save the model
    torch.save(trained_model.state_dict(), 'transformer_model.pth')
    print("Model saved to transformer_model.pth")