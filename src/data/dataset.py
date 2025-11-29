# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from typing import List, Optional

class TokenDataset(Dataset):
    """
    Wraps a 1D sequence of token ids and produces (x, y) sliding windows.
    Can be constructed either from a list/torch tensor of ids, or from raw text + tokenizer.
    """
    def __init__(self,
                 token_ids: Optional[List[int]] = None,
                 seq_len: int = 128,
                 tokenizer=None,
                 text: Optional[str] = None):
        if token_ids is None:
            assert tokenizer is not None and text is not None, "Provide token_ids or (tokenizer and text)"
            token_ids = tokenizer.encode(text)
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + 1 + self.seq_len]
        return x, y
