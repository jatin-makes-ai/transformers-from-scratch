from src.tokenizers.bpe_tokenizer import BPETokenizer
from src.model.transformer import TinyTransformerLM
tk = BPETokenizer(); tk.load("src/tokenizers/bpe_vocab.json")
print("tk vocab_size:", tk.vocab_size, "max id:", max(int(v) for v in tk.stoi.values()))
m = TinyTransformerLM(vocab_size=tk.vocab_size)
emb = m.get_input_embeddings()
print("embedding shape:", tuple(emb.weight.shape))