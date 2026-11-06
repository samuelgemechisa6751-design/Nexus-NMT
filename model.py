import torch
import torch.nn as nn
import math
import pandas as pd
from torch.utils.data import Dataset

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        Q = self.W_q(q).view(q.size(0), q.size(1), self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(k.size(0), k.size(1), self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(v.size(0), v.size(1), self.num_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 🚀 FIX: Use torch.finfo to prevent float16 overflow
        if mask is not None: 
            attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)
            
        out = torch.matmul(torch.softmax(attn, dim=-1), V)
        return self.W_o(out.transpose(1, 2).contiguous().view(q.size(0), q.size(1), self.d_model))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x): 
        return x + self.pe[:, :x.size(1), :]

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x): 
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
        return self.norm2(x + self.dropout2(self.feed_forward(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, enc_output, enc_output, src_mask)))
        return self.norm3(x + self.dropout3(self.feed_forward(x)))

class NexusNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        # 🚀 60M Model Architecture
        self.d_model = 512 
        self.src_embedding = nn.Embedding(src_vocab_size, self.d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.d_model, 8, 2048, 0.1) for _ in range(6)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.d_model, 8, 2048, 0.1) for _ in range(6)])
        
        self.fc_out = nn.Linear(self.d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.scale = math.sqrt(self.d_model)
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) 
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        nopeak = (1 - torch.triu(torch.ones(1, tgt.size(1), tgt.size(1)), diagonal=1)).bool().to(tgt.device)
        return src_mask, tgt_mask & nopeak
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_emb = self.dropout(self.positional_encoding(self.src_embedding(src) * self.scale))
        enc_out = src_emb
        for layer in self.encoder_layers: 
            enc_out = layer(enc_out, src_mask)
            
        tgt_emb = self.dropout(self.positional_encoding(self.tgt_embedding(tgt) * self.scale))
        dec_out = tgt_emb
        for layer in self.decoder_layers: 
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
        return self.fc_out(dec_out)

class NexusDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")
        self.tag_2am = tokenizer.token_to_id("<2am>")
        self.tag_2en = tokenizer.token_to_id("<2en>")
        
    def __len__(self): 
        return len(self.df) * 2
    
    def __getitem__(self, idx):
        row_idx = idx // 2
        direction = idx % 2 
        row = self.df.iloc[row_idx]
        en, am = str(row["src"]), str(row["tgt"])
        
        if direction == 0:
            src_ids = [self.tag_2am] + self.tokenizer.encode(en).ids + [self.eos_id]
            tgt_ids = [self.bos_id] + self.tokenizer.encode(am).ids + [self.eos_id]
        else:
            src_ids = [self.tag_2en] + self.tokenizer.encode(am).ids + [self.eos_id]
            tgt_ids = [self.bos_id] + self.tokenizer.encode(en).ids + [self.eos_id]
            
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]
        
        return {"src": torch.tensor(src_ids, dtype=torch.long), 
                "tgt": torch.tensor(tgt_ids, dtype=torch.long)}
