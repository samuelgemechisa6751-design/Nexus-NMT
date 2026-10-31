
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import NexusNMT
from dataset import NexusDataset
from tokenizers import ByteLevelBPETokenizer
import glob
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training 300M Model on: {device}")

tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
file_path = glob.glob("/kaggle/input/**/*en-am.clean_v4.parquet", recursive=True)[0]
dataset = NexusDataset(file_path, tokenizer, max_len=64)

# 🚨 BATCH SIZE 4 FOR 300M MODEL 🚨
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

vocab_size = tokenizer.get_vocab_size()
model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

model.train()
print("🚀 Starting 300M Training from Scratch...")

for batch_idx, batch in enumerate(dataloader):
    src = batch["src"].to(device)
    tgt = batch["tgt"].to(device)
    
    optimizer.zero_grad()
    output = model(src, tgt)
    output = output[:, :-1, :].contiguous()
    tgt_shifted = tgt[:, 1:].contiguous()
    output = output.view(-1, output.shape[-1])
    tgt_shifted = tgt_shifted.view(-1)
    
    loss = criterion(output, tgt_shifted)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")
        
    if batch_idx > 0 and batch_idx % 500 == 0:
        torch.save(model.state_dict(), f"nexus_300M_batch_{batch_idx}.pt")
        print(f"  💾 Checkpoint saved at {batch_idx}")
