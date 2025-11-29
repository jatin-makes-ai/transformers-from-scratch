# src/train.py
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# tokenizers & dataset & model
from data.dataset import TokenDataset
from model.transformer import TinyTransformerLM
from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.bpe_tokenizer import BPETokenizer

def generate_sample(model, tokenizer, start_text, max_new_tokens=100, device="cpu", temperature=1.0, top_k=0):
    model.eval()
    # build start ids safely (tokenizer.encode is non-mutating after fix)
    start_ids = tokenizer.encode(start_text)
    if len(start_ids) == 0:
        start_ids = [tokenizer.stoi.get("<unk>", 0)]
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    # pre-check embedding coverage before any GPU indexing
    try:
        emb = model.get_input_embeddings()
        vocab_rows = emb.weight.shape[0]
        min_id = int(idx.min().item())
        max_id = int(idx.max().item())
        if min_id < 0 or max_id >= vocab_rows:
            raise RuntimeError(f"TOKEN INDEX OUT OF RANGE BEFORE FORWARD: idx min/max = {min_id}/{max_id} vs embedding rows = {vocab_rows}")
    except AttributeError:
        # if get_input_embeddings not available, try to inspect tok_emb
        try:
            vocab_rows = model.tok_emb.embedding.weight.shape[0]
            min_id = int(idx.min().item())
            max_id = int(idx.max().item())
            if min_id < 0 or max_id >= vocab_rows:
                raise RuntimeError(f"TOKEN INDEX OUT OF RANGE BEFORE FORWARD: idx min/max = {min_id}/{max_id} vs embedding rows = {vocab_rows}")
        except Exception:
            # last resort: continue, but may crash with CUDA assert later
            pass

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if idx.size(1) > model.max_len:
                idx = idx[:, -model.max_len:]
            logits = model(idx)[:, -1, :]              # (1, V)
            logits = logits / max(1e-8, temperature)
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                min_val = vals[..., -1, None]
                logits = torch.where(logits < min_val, torch.full_like(logits, -1e10), logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/large.txt")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char", help="tokenizer type to use")
    p.add_argument("--bpe_path", default="src/tokenizers/bpe_vocab.json", help="path to saved BPE vocab (if tokenizer=bpe)")
    p.add_argument("--top_k", type=int, default=0)
    args = p.parse_args()

    text = open(args.data, "r", encoding="utf-8").read()

    # build tokenizer and dataset
    if args.tokenizer == "bpe":
        tokenizer = BPETokenizer()
        tokenizer.load(args.bpe_path)
        token_ids = tokenizer.encode(text)
        dataset = TokenDataset(token_ids=token_ids, seq_len=args.seq_len)
    else:
        tokenizer = CharTokenizer.from_text(text)
        dataset = TokenDataset(tokenizer=tokenizer, text=text, seq_len=args.seq_len)

    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    model = TinyTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.seq_len
    ).to(args.device)

    # print param count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}, vocab_size: {tokenizer.vocab_size}, device: {args.device}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        epoch_loss = 0.0
        steps = 0
        for xb, yb in pbar:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            logits = model(xb)                    # (B, T, V)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), yb.view(B * T))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            epoch_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=epoch_loss / steps)

        # save checkpoint: include tokenizer mapping and model args
        ckpt_path = os.path.join(args.save_dir, f"model_epoch{epoch}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "stoi": tokenizer.stoi,
            "model_args": {
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "d_ff": args.d_ff,
                "max_len": args.seq_len
            },
            "tokenizer_type": args.tokenizer
        }, ckpt_path)

        # sample and print
        seed = text[: min(len(text), args.seq_len)]
        sample_ids = generate_sample(model, tokenizer, seed, max_new_tokens=200, device=args.device, temperature=1.0, top_k=args.top_k)
        sample_text = tokenizer.decode(sample_ids)
        print(f"\n--- Sample after epoch {epoch} ---\n{sample_text}\n")

if __name__ == "__main__":
    main()
