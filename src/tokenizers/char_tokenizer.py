import json
from .base import Tokenizer

class CharTokenizer(Tokenizer):
    def __init__(self, stoi=None, itos=None):
        self.stoi = stoi or {}
        self.itos = itos or {}

    @classmethod
    def from_text(cls, text: str):
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}           #stoi = string-to-int
        itos = {i: c for c, i in stoi.items()}                    #itos = int-to-string
        return cls(stoi, itos)

    def encode(self, text: str) -> list:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list) -> str:
        return ''.join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.stoi = data["stoi"]
            self.itos = {i: c for c, i in self.stoi.items()}
