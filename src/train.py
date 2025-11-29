# src/train.py
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizers.char_tokenizer import CharTokenizer
from data.dataset import TokenDataset
from model.transformer import TinyTransformerLM

def generate(model, start_ids, itos, max_new_tokens=100, device="cpu", temperature=1.0):
    model.eval()
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(idx)                # (1, T, V)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
            idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tiny.txt")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", default="checkpoints")
    args = p.parse_args()

    text = open(args.data, "r", encoding="utf-8").read()
    tokenizer = CharTokenizer.from_text(text)
    dataset = TokenDataset(tokenizer=tokenizer, text=text, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    model = TinyTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=512         #args.seq_len
    ).to(args.device)

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

        # save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"model_epoch{epoch}.pt")
        torch.save({"model_state": model.state_dict(), "stoi": tokenizer.stoi}, ckpt_path)

        # sample
        start_text = text[:args.seq_len]
        start_ids = tokenizer.encode(start_text)
        sample_ids = generate(model, start_ids, tokenizer.itos, max_new_tokens=200, device=args.device)
        sample_text = tokenizer.decode(sample_ids)
        print(f"\n--- Sample after epoch {epoch} ---\n{sample_text}\n")

if __name__ == "__main__":
    main()
