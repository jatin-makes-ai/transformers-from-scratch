import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """Simple token embedding layer."""
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, ids: torch.LongTensor):
        # ids: (B, T) -> returns (B, T, D)
        return self.embedding(ids)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (adds to token embeddings)."""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        # register as buffer so it moves with .to(device) and isn't a parameter
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_head = d_model // n_heads   # dimension per head
        self.n_heads = n_heads

        # single Linear for q,k,v: projects d_model → 3*d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # final linear projection after concatenating heads
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.size()

        # project to q, k, v
        qkv = self.qkv(x)                 # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)    # three tensors each (B, T, D)

        # reshape for multi-head: (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # attention scores: (B, n_heads, T, T)
        scores = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)

        # apply causal mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax across last dimension (the keys)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # weighted sum of values
        out = attn @ v                     # (B, n_heads, T, d_head)

        # concatenate heads: (B, T, D)
        out = out.transpose(1, 2).reshape(B, T, D)

        # final projection
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        # Attention sublayer
        attn_out = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out           # residual

        # FFN sublayer
        ff_out = self.ff(self.ln2(x))
        x = x + ff_out             # residual

        return x
