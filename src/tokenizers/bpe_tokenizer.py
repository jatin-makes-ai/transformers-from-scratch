# src/tokenizers/bpe_tokenizer.py
import re
import json
from collections import defaultdict, Counter
from .base import Tokenizer

_whitespace_split_re = re.compile(r"\s+")

class BPETokenizer(Tokenizer):
    """
    Minimal BPE tokenizer (word-level training, character-initial symbols with </w>).
    Not optimized for huge corpora — intended for learning and small-medium corpora.
    """

    def __init__(self, merges=None, vocab=None, stoi=None, itos=None, unk_token="<unk>"):
        self.merges = merges or []
        self.vocab = set(vocab or [])
        self.stoi = dict(stoi or {})
        self.itos = dict(itos or {})
        self.unk_token = unk_token
        # ensure unk exists
        if self.unk_token not in self.stoi:
            next_id = len(self.stoi)
            self.stoi[self.unk_token] = next_id
            self.itos[next_id] = self.unk_token
        self.vocab.update(self.stoi.keys())

    @staticmethod
    def _get_word_tokens(word):
        # represent word as list of chars with end-of-word symbol
        return [c for c in word] + ['</w>']

    @staticmethod
    def _get_word_str(tokens):
        # join tokens to a single string separated by space (for pair counting)
        return ' '.join(tokens)

    def _get_stats(self, corpus_words):
        """Count frequency of adjacent symbol pairs in corpus_words.
        corpus_words: list of token lists (with </w>) repeated per frequency."""
        pairs = defaultdict(int)
        for word, freq in corpus_words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_pair(self, pair, corpus_words):
        """Merge a given pair (a,b) in all corpus_words representation."""
        a, b = pair
        merged = a + b
        pattern = re.compile(r'(?<!\S)' + re.escape(a) + r'\s+' + re.escape(b) + r'(?!\S)')
        new_corpus = {}
        for word, freq in corpus_words.items():
            # word is a string like "t h i s </w>"
            new_word = pattern.sub(merged, word)
            new_corpus[new_word] = new_corpus.get(new_word, 0) + freq
        return new_corpus

    def train_from_text(self, text: str, num_merges: int = 1000, min_count: int = 1):
        """
        Train BPE merges from raw text.
        - text: raw string
        - num_merges: number of merges to perform
        - min_count: skip pairs with counts < min_count
        """
        # 1) build word frequency map (split by whitespace)
        words = _whitespace_split_re.split(text.strip())
        freq = Counter(w for w in words if w != '')
        # build corpus representation: word -> token string "t h i s </w>"
        corpus_words = {}
        for w, f in freq.items():
            tokens = self._get_word_tokens(w)
            word_repr = ' '.join(tokens)
            corpus_words[word_repr] = corpus_words.get(word_repr, 0) + f

        merges = []
        for i in range(num_merges):
            pairs = self._get_stats(corpus_words)
            if not pairs:
                break
            # pick highest-frequency pair
            best_pair, best_count = max(pairs.items(), key=lambda x: x[1])
            if best_count < min_count:
                break
            merges.append(best_pair)
            corpus_words = self._merge_pair(best_pair, corpus_words)

        # build vocab set from final corpus_words
        vocab = set()
        for word in corpus_words.keys():
            for sym in word.split():
                vocab.add(sym)
        # record merges and vocab
        self.merges = merges
        self.vocab = vocab

        # build stoi/itos based on sorted vocab (deterministic)
        sorted_vocab = sorted(list(vocab))
        self.stoi = {s: i for i, s in enumerate(sorted_vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def _apply_merges_to_word(self, word):
        """Given a raw word string, apply merges greedily and return list of symbols."""
        # start with chars + </w>
        symbols = self._get_word_tokens(word)
        # represent as list of strings
        # repeatedly apply merges in the same order as trained
        for a, b in self.merges:
            merged = a + b
            i = 0
            new_symbols = []
            while i < len(symbols):
                # if a followed by b at i, merge
                if i + 1 < len(symbols) and symbols[i] == a and symbols[i+1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str) -> list:
        """
        Encode text -> ids. DOES NOT grow vocab; unknown symbols map to unk token.
        """
        ids = []
        words = _whitespace_split_re.split(text.strip())
        unk_id = self.stoi.get(self.unk_token, 0)
        for w in words:
            if w == '':
                continue
            symbols = self._apply_merges_to_word(w)
            for s in symbols:
                if s not in self.stoi:
                    ids.append(unk_id)
                else:
                    ids.append(self.stoi[s])
        return ids

    def decode(self, ids: list) -> str:
        """Decode list of ids back to text (approximate)."""
        # convert ids -> symbols, rebuild words by concatenating until '</w>' symbol
        out_words = []
        cur = []
        for i in ids:
            s = self.itos.get(int(i), '')
            if s == '</w>':
                out_words.append(''.join(cur))
                cur = []
            else:
                cur.append(s)
        # if leftover symbols, append as last word
        if cur:
            out_words.append(''.join(cur))
        # join words with spaces
        return ' '.join(out_words)

    def save(self, path: str):
        d = {
            "merges": [(a, b) for (a, b) in self.merges],
            "stoi": self.stoi
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.merges = [tuple(pair) for pair in d.get("merges", [])]
        self.stoi = {k: int(v) for k, v in d.get("stoi", {}).items()}
        # ensure unk exists
        if self.unk_token not in self.stoi:
            next_id = max(self.stoi.values()) + 1 if self.stoi else 0
            self.stoi[self.unk_token] = next_id
        self.itos = {int(i): s for s, i in self.stoi.items()}
        self.vocab = set(self.stoi.keys())

    @property
    def vocab_size(self) -> int:
        """ Number of tokens in the vocab currently"""
        return len(self.stoi)