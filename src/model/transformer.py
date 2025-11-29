import torch
import torch.nn as nn
from .modules import TokenEmbedding, PositionalEncoding, TransformerBlock

class TinyTransformerLM(nn.Module):
    """
    Decoder-only transformer (causal LM).
    Input: token ids (B, T)
    Output: logits (B, T, vocab_size)
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 d_ff: int = 512,
                 max_len: int = 512,
                 dropout: float = 0.0):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.d_model = d_model
        self.max_len = max_len

    def make_causal_mask(self, T: int, device: torch.device):
        # (1, 1, T, T) lower-triangular mask of 1s for allowed positions
        mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

    def forward(self, idx: torch.LongTensor):
        """
        idx: (B, T)
        returns logits: (B, T, V)
        """
        B, T = idx.size()
        assert T <= self.max_len, f"Sequence length {T} > max_len {self.max_len}"

        x = self.tok_emb(idx)           # (B, T, D)
        x = self.pos_enc(x)             # (B, T, D)

        causal_mask = self.make_causal_mask(T, idx.device)  # broadcastable

        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        x = self.ln_f(x)                # (B, T, D)
        logits = self.head(x)           # (B, T, V)
        return logits

    def get_input_embeddings(self):
        """Return the underlying nn.Embedding used for token embeddings."""
        # tok_emb is a TokenEmbedding wrapper; it holds the real embedding at .embedding
        if hasattr(self, "tok_emb"):
            te = self.tok_emb
            # if wrapper has .embedding attribute that is nn.Embedding, return it
            if hasattr(te, "embedding") and isinstance(te.embedding, nn.Embedding):
                return te.embedding
            # if tok_emb itself is an Embedding
            if isinstance(te, nn.Embedding):
                return te
        # fallback: search submodules
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Embedding):
                return mod
        raise AttributeError("No nn.Embedding found in model.")

    def set_input_embeddings(self, new_emb: nn.Embedding):
        """Replace underlying embedding. Tries to place into tok_emb.embedding if present."""
        if hasattr(self, "tok_emb"):
            te = self.tok_emb
            if hasattr(te, "embedding"):
                setattr(te, "embedding", new_emb)
                return
            # if tok_emb itself is an Embedding
            if isinstance(te, nn.Embedding):
                self.tok_emb = new_emb
                return
        # fallback: attach to tok_emb
        self.tok_emb = TokenEmbedding(new_emb.num_embeddings, new_emb.embedding_dim)
        # try to copy weights if dims match
        try:
            self.tok_emb.embedding = new_emb
        except Exception:
            pass

