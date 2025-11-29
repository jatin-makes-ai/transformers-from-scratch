# scripts/train_bpe.py
from tokenizers.bpe_tokenizer import BPETokenizer
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to raw text (e.g. data/large.txt)")
    p.add_argument("--merges", type=int, default=2000, help="number of BPE merges")
    p.add_argument("--out", default="src/tokenizers/bpe_vocab.json", help="output vocab path")
    args = p.parse_args()

    txt = open(args.input, "r", encoding="utf-8").read()
    tok = BPETokenizer()
    tok.train_from_text(txt, num_merges=args.merges)
    tok.save(args.out)
    print("Saved BPE vocab to", args.out)
    print("vocab_size=", tok.vocab_size, "merges=", len(tok.merges))

if __name__ == "__main__":
    main()
