# src/generate.py
import argparse
import os
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.transformer import TinyTransformerLM
from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.bpe_tokenizer import BPETokenizer

# -------------------------
# utilities
# -------------------------
def top_k_logits(logits, k):
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    min_v = v[..., -1, None]
    return torch.where(logits < min_v, torch.full_like(logits, -1e10), logits)

def find_embedding_module(model):
    """Return (full_name, module) for the most-likely token embedding (nn.Embedding)."""
    # Preferred: model.get_input_embeddings() if available and returns nn.Embedding
    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
            if isinstance(emb, nn.Embedding):
                return None, emb
        except Exception:
            pass

    # common wrapper: model.tok_emb.embedding
    if hasattr(model, "tok_emb"):
        te = getattr(model, "tok_emb")
        if hasattr(te, "embedding") and isinstance(te.embedding, nn.Embedding):
            return "tok_emb.embedding", te.embedding
        if isinstance(te, nn.Embedding):
            return "tok_emb", te

    # fallback: search named_modules for nn.Embedding and pick the one with the largest num_embeddings
    best_name = None
    best_mod = None
    best_rows = -1
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            try:
                rows = int(mod.weight.shape[0])
            except Exception:
                rows = -1
            if rows > best_rows:
                best_rows = rows
                best_name = name
                best_mod = mod
    return best_name, best_mod

def replace_module_by_name(model, full_name, new_module):
    """Replace a submodule given dotted `full_name` with new_module."""
    if full_name is None or full_name == "":
        raise ValueError("full_name empty")
    parts = full_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)

def auto_resize_embedding_to_tokenizer(model, tokenizer_stoi):
    """
    Ensure model embedding has rows >= max_id+1 from tokenizer_stoi.
    Returns (changed: bool, message: str)
    """
    if not tokenizer_stoi:
        return False, "no tokenizer stoi provided"

    try:
        max_id = max(int(v) for v in tokenizer_stoi.values())
    except Exception as e:
        return False, f"failed to parse stoi: {e}"
    required_vocab = max_id + 1

    name, emb = find_embedding_module(model)
    if emb is None:
        return False, "no embedding module found in model"

    cur_rows, emb_dim = tuple(emb.weight.shape)
    if cur_rows >= required_vocab:
        return False, f"embedding already large enough: {cur_rows} >= {required_vocab}"

    # create new embedding, copy weights
    device = emb.weight.device
    dtype = emb.weight.dtype
    old_w = emb.weight.data
    new_w = torch.randn(required_vocab, emb_dim, device=device, dtype=dtype) * 0.01
    ncopy = min(cur_rows, required_vocab)
    new_w[:ncopy] = old_w[:ncopy]

    new_emb = nn.Embedding(required_vocab, emb_dim).to(device)
    new_emb.weight.data.copy_(new_w)

    # attach new_emb to model
    try:
        if name in (None, ""):
            # if find_embedding returned (None, emb) because get_input_embeddings used, try set_input_embeddings
            if hasattr(model, "set_input_embeddings"):
                model.set_input_embeddings(new_emb)
            else:
                # fallback: try to find tok_emb wrapper and set tok_emb.embedding
                if hasattr(model, "tok_emb") and hasattr(model.tok_emb, "embedding"):
                    model.tok_emb.embedding = new_emb
                else:
                    # last resort: replace by searching name from named_modules (shouldn't happen)
                    raise RuntimeError("unable to attach new embedding (no attribute to set)")
        else:
            replace_module_by_name(model, name, new_emb)
    except Exception as e:
        return False, f"failed to attach resized embedding: {e}"

    return True, f"resized embedding {cur_rows} -> {required_vocab} and attached at {name or 'set_input_embeddings/tok_emb'}"

# -------------------------
# tokenizer builder
# -------------------------
def build_tokenizer_from_ckpt_or_file(ckpt, tokenizer_path=None, tokenizer_type_hint=None):
    # prefer explicit tokenizer file
    if tokenizer_path and os.path.exists(tokenizer_path):
        # inspect file: if it's BPE style use BPETokenizer
        tok = BPETokenizer()
        tok.load(tokenizer_path)
        return tok

    # else fallback to ckpt-stored stoi
    if isinstance(ckpt, dict) and "stoi" in ckpt:
        stoi = ckpt["stoi"]
        # make sure keys/values have correct types
        # stoi in ckpt is likely token->id
        stoi = {str(k): int(v) for k, v in stoi.items()}
        # build itos
        itos = {int(v): k for k, v in stoi.items()}
        # choose tokenizer type (hint or fallback char)
        ttype = tokenizer_type_hint or ckpt.get("tokenizer_type", "char")
        if ttype == "bpe":
            tok = BPETokenizer(stoi=stoi, itos=itos)
            # ensure unk exists
            if "<unk>" not in tok.stoi:
                next_id = max(tok.stoi.values()) + 1
                tok.stoi["<unk>"] = next_id
                tok.itos[next_id] = "<unk>"
            return tok
        else:
            return CharTokenizer(stoi=stoi, itos=itos)
    raise RuntimeError("Cannot build tokenizer: provide tokenizer file or checkpoint with 'stoi'.")

# -------------------------
# generation
# -------------------------
def generate(model, tokenizer, start_text, max_new_tokens=200, device="cpu", temperature=1.0, top_k=0):
    model.eval()

    # encode seed (tokenizer should be non-mutating)
    start_ids = tokenizer.encode(start_text)
    if len(start_ids) == 0:
        start_ids = [tokenizer.stoi.get("<unk>", 0)]
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    # check indices against model embedding BEFORE any forward
    try:
        name, emb = find_embedding_module(model)
        if emb is not None:
            vocab_rows = emb.weight.shape[0]
            min_id = int(idx.min().item())
            max_id = int(idx.max().item())
            if min_id < 0 or max_id >= vocab_rows:
                raise RuntimeError(f"TOKEN INDEX OUT OF RANGE BEFORE FORWARD: idx min/max = {min_id}/{max_id} vs embedding rows = {vocab_rows}")
    except RuntimeError:
        raise
    except Exception:
        # could not inspect embedding; continue but may hit CUDA assert
        pass

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if idx.size(1) > model.max_len:
                idx = idx[:, -model.max_len:]
            logits = model(idx)[:, -1, :]     # (1, V)
            logits = logits / max(1e-8, temperature)
            logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].cpu().tolist())

# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint path saved from train.py")
    p.add_argument("--tokenizer", default="src/tokenizers/bpe_vocab.json", help="optional tokenizer json file path (overrides checkpoint)")
    p.add_argument("--start", default="", help="seed text")
    p.add_argument("--max_new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # load checkpoint (cpu map to avoid GPU memory surprises)
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # reconstruct tokenizer (from file preferred)
    tokenizer = build_tokenizer_from_ckpt_or_file(ckpt, tokenizer_path=args.tokenizer, tokenizer_type_hint=ckpt.get("tokenizer_type", None))

    # model args from checkpoint if present
    model_kwargs = ckpt.get("model_args", {})
    d_model = model_kwargs.get("d_model", 128)
    n_heads = model_kwargs.get("n_heads", 4)
    n_layers = model_kwargs.get("n_layers", 4)
    d_ff = model_kwargs.get("d_ff", 512)
    max_len = model_kwargs.get("max_len", 512)

    # build model with tokenizer.vocab_size
    model = TinyTransformerLM(
        vocab_size=getattr(tokenizer, "vocab_size", None) or len(getattr(tokenizer, "stoi", {})),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len
    )

    # load model weights (support different checkpoint key names)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    # attempt to strictly load, fall back to non-strict if shapes mismatch
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # try non-strict to allow size mismatch (we'll resize embedding if needed)
        model.load_state_dict(state, strict=False)

    # move model to device before possible resizing/attachment (so created tensors are consistent)
    model.to(args.device)

    # auto-resize embedding if tokenizer requires larger vocab
    stoi = getattr(tokenizer, "stoi", None)
    changed, msg = auto_resize_embedding_to_tokenizer(model, stoi)
    if changed:
        # ensure model on desired device after replacement
        model.to(args.device)
    # print status for user
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}, tokenizer_vocab_size: {getattr(tokenizer, 'vocab_size', len(stoi) if stoi else 'unknown')}, device: {args.device}")
    if msg:
        print("Embedding auto-resize status:", msg)

    # run generation
    out = generate(model, tokenizer, start_text=args.start, max_new_tokens=args.max_new, device=args.device, temperature=args.temperature, top_k=args.top_k)
    print("\n--- Generated text ---\n")
    print(out)
    print("\n----------------------\n")

if __name__ == "__main__":
    main()
