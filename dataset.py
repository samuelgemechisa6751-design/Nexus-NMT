import pandas as pd
import torch
from torch.utils.data import Dataset

class NexusDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_len=64):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.tag_2am = tokenizer.token_to_id("<2am>")
        self.tag_2en = tokenizer.token_to_id("<2en>")

    def __len__(self):
        return len(self.df) * 2

    def __getitem__(self, idx):
        row_idx = idx // 2
        direction = idx % 2 
        row = self.df.iloc[row_idx]
        en_text = str(row["src"])
        am_text = str(row["tgt"])

        if direction == 0:
            src_ids = [self.tag_2am] + self.tokenizer.encode(en_text).ids + [self.eos_id]
            tgt_ids = [self.bos_id] + self.tokenizer.encode(am_text).ids + [self.eos_id]
        else:
            src_ids = [self.tag_2en] + self.tokenizer.encode(am_text).ids + [self.eos_id]
            tgt_ids = [self.bos_id] + self.tokenizer.encode(en_text).ids + [self.eos_id]

        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]
        src_pad = src_ids + [self.pad_id] * (self.max_len - len(src_ids))
        tgt_pad = tgt_ids + [self.pad_id] * (self.max_len - len(tgt_ids))

        return {
            "src": torch.tensor(src_pad, dtype=torch.long),
            "tgt": torch.tensor(tgt_pad, dtype=torch.long)
        }
