# src/generate.py
import argparse
import torch
import torch.nn.functional as F
from model.transformer import TinyTransformerLM
from tokenizers.char_tokenizer import CharTokenizer

def top_k_logits(logits, k):
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    min_v = v[..., -1, None]
    return torch.where(logits < min_v, torch.full_like(logits, -1e10), logits)

def generate(model, tokenizer, start_text, max_new_tokens=200, device="cpu", temperature=1.0, top_k=0):
    model.eval()
    stoi = tokenizer.stoi
    itos = tokenizer.itos
    start_ids = [stoi[c] for c in start_text]
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # ensure sequence length <= model.max_len
            if idx.size(1) > model.max_len:
                idx = idx[:, -model.max_len:]
            logits = model(idx)           # (1, T, V)
            logits = logits[:, -1, :]     # (1, V)
            logits = logits / max(1e-8, temperature)
            logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
            idx = torch.cat([idx, next_id], dim=1)

    out_ids = idx[0].tolist()
    return tokenizer.decode(out_ids)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt) saved from train.py")
    p.add_argument("--start", default="", help="Starting text (seed).")
    p.add_argument("--max_new", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    args = p.parse_args()

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # extract stoi if present
    if isinstance(ckpt, dict) and "stoi" in ckpt:
        stoi = ckpt["stoi"]
    elif isinstance(ckpt, dict) and "vocab" in ckpt:
        stoi = ckpt["vocab"]
    else:
        raise RuntimeError("Checkpoint must contain 'stoi' or 'vocab' mapping.")

    # build tokenizer
    # stoi: char -> int ; make itos: int -> char
    itos = {int(i): c for c, i in stoi.items()}
    tokenizer = CharTokenizer(stoi=stoi, itos=itos)

    # instantiate model (use small sensible defaults; seq_len = model.max_len must be >= starting length)
    vocab_size = tokenizer.vocab_size
    # try to detect saved model dims from checkpoint if available
    # if not present, use defaults compatible with training script
    model_kwargs = ckpt.get("model_args", {}) if isinstance(ckpt, dict) else {}
    # fallback defaults (these should match what you trained with)
    d_model = model_kwargs.get("d_model", 128)
    n_heads = model_kwargs.get("n_heads", 4)
    n_layers = model_kwargs.get("n_layers", 4)
    d_ff = model_kwargs.get("d_ff", 512)
    max_len = model_kwargs.get("max_len", 512)

    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len
    )

    # load state if present
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        # maybe whole state dict was saved directly
        state = ckpt
    model.load_state_dict(state)
    model.to(args.device)

    out = generate(model, tokenizer, start_text=args.start, max_new_tokens=args.max_new,
                   device=args.device, temperature=args.temperature, top_k=args.top_k)
    print("\n--- Generated text ---\n")
    print(out)
    print("\n----------------------\n")

if __name__ == "__main__":
    main()
