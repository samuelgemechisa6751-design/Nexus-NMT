
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import NexusNMT
from dataset import NexusDataset
from tokenizers import ByteLevelBPETokenizer

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. Load Tokenizer & Dataset
    tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
    file_path = glob.glob("/kaggle/input/**/*en-am.clean_v4.parquet", recursive=True)[0]
    dataset = NexusDataset(file_path, tokenizer, max_len=64)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    
    # 2. Initialize Model
    vocab_size = tokenizer.get_vocab_size()
    model = NexusNMT(
        src_vocab_size=vocab_size, 
        tgt_vocab_size=vocab_size, 
        d_model=256, num_heads=8, num_layers=4, d_ff=1024
    ).to(device)
    
    # 3. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))
    
    # 4. Training Loop (1 Epoch for testing)
    model.train()
    print("Starting training loop...")
    
    for batch_idx, batch in enumerate(dataloader):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt)
        
        # Reshape for loss (batch * seq_len, vocab_size)
        output = output.view(-1, output.shape[-1])
        tgt = tgt.view(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    # 5. Save Checkpoint
    torch.save(model.state_dict(), "nexus_nmt_weights.pt")
    print("Training complete! Weights saved.")

if __name__ == "__main__":
    import glob
    train()
