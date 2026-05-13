"""
tokenizer.py — VoynichTokenizer
Advanced tokenizer for Voynich / cipher-text corpora.

Features vs the original
─────────────────────────
• Three tokenisation modes: char | word | bigram
• Special tokens: <PAD>, <UNK>, <BOS>, <EOS>
• Deep corpus cleaning (folio markers, annotations, punctuation)
• Shannon entropy calculation
• Index of Coincidence (IC) — key metric for classical cryptanalysis
• N-gram frequency analysis
• Rich CorpusStats dataclass
• encode / decode round-trip with special-token stripping
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Stats dataclass ─────────────────────────────────────────────────────────

@dataclass
class CorpusStats:
    total_chars:     int
    total_tokens:    int
    unique_tokens:   int
    vocab_size:      int
    entropy_bits:    float          # Shannon entropy H (bits)
    index_coincid:   float          # IC — ~0.065 for English, ~0.038 for random
    avg_token_len:   float
    hapax_legomena:  int            # tokens appearing exactly once
    type_token_ratio: float         # lexical diversity [0,1]
    top_tokens:      List[Tuple[str, int]] = field(default_factory=list)
    top_chars:       List[Tuple[str, int]] = field(default_factory=list)
    top_bigrams:     List[Tuple[str, int]] = field(default_factory=list)

    def render(self, color: bool = True) -> str:
        C = {
            "H": "\033[1m", "B": "\033[94m", "G": "\033[92m",
            "Y": "\033[93m", "R": "\033[0m"
        } if color else {k: "" for k in ("H","B","G","Y","R")}

        lines = [
            f"{C['H']}{'─'*55}",
            f"  CORPUS ANALYSIS",
            f"{'─'*55}{C['R']}",
            f"  Characters    : {C['G']}{self.total_chars:>10,}{C['R']}",
            f"  Tokens        : {C['G']}{self.total_tokens:>10,}{C['R']}",
            f"  Unique tokens : {C['G']}{self.unique_tokens:>10,}{C['R']}",
            f"  Vocab (fitted): {C['G']}{self.vocab_size:>10,}{C['R']}",
            f"  Avg token len : {C['Y']}{self.avg_token_len:>10.2f}{C['R']}",
            f"  Hapax legomena: {C['Y']}{self.hapax_legomena:>10,}{C['R']}",
            f"  Type-token TTR: {C['Y']}{self.type_token_ratio:>10.4f}{C['R']}",
            f"  Shannon entropy:{C['B']}{self.entropy_bits:>9.4f} bits{C['R']}",
            f"  Index of Coinc:{C['B']}{self.index_coincid:>10.6f}{C['R']}",
            f"    (English ≈ 0.065 | Random ≈ 0.038 | Voynich ≈ 0.057)",
            f"",
            f"  Top-10 tokens : {', '.join(f'{t}({n})' for t,n in self.top_tokens[:10])}",
            f"  Top-10 chars  : {', '.join(f'{repr(c)}({n})' for c,n in self.top_chars[:10])}",
            f"  Top-10 bigrams: {', '.join(f'{b}({n})' for b,n in self.top_bigrams[:10])}",
            f"{'─'*55}",
        ]
        return "\n".join(lines)


# ─── Tokenizer ────────────────────────────────────────────────────────────────

class VoynichTokenizer:
    """
    Tokenizer for Voynich / cipher-text corpora.

    Modes
    ─────
    char   — character-level (best for unknown scripts)
    word   — whole-word tokens (for word-pattern analysis)
    bigram — overlapping 2-char tokens within words (captures sub-word patterns)

    Usage
    ─────
    tok = VoynichTokenizer(mode="char")
    clean = tok.limpiar_texto(raw_text)
    tok.ajustar(clean)
    ids = tok.codificar(clean)
    text = tok.decodificar(ids)
    stats = tok.obtener_estadisticas(clean)
    print(stats.render())
    """

    SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    PAD_ID  = 0
    UNK_ID  = 1
    BOS_ID  = 2
    EOS_ID  = 3

    VALID_MODES = {"char", "word", "bigram"}

    def __init__(self, mode: str = "char"):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode:    str           = mode
        self.char2i:  Dict[str,int] = dict(self.SPECIAL)
        self.i2char:  Dict[int,str] = {v: k for k, v in self.SPECIAL.items()}
        self.vocab_size: int        = len(self.SPECIAL)
        self._fitted: bool          = False

    # ── Corpus cleaning ───────────────────────────────────────────────────────

    def limpiar_texto(self, texto: str) -> str:
        """
        Deep cleaning pipeline for Voynich transcription files:
          1. Strip HTML / XML tags
          2. Strip Voynich folio markers  (f1r, f32v2, ...)
          3. Strip curly-brace annotations and comments
          4. Remove digits, punctuation, special characters
          5. Lowercase + collapse whitespace
        """
        texto = re.sub(r"<[^>]+>", "", texto)                   # HTML/XML tags
        texto = re.sub(r"\bf\d+[rv][\d.]*\b", "", texto)        # folio markers
        texto = re.sub(r"\{[^}]*\}", "", texto)                  # {annotations}
        texto = re.sub(r"[0-9!?,;:\-\[\]\(\)_=+#@$%^&*|\\/<>~`\"\'\.]+", "", texto)
        texto = texto.lower().strip()
        return " ".join(texto.split())

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _tokenise(self, texto: str) -> List[str]:
        if self.mode == "char":
            return list(texto)
        elif self.mode == "word":
            return texto.split()
        elif self.mode == "bigram":
            tokens: List[str] = []
            for word in texto.split():
                for i in range(len(word) - 1):
                    tokens.append(word[i : i + 2])
                tokens.append(" ")
            return tokens

    def ajustar(self, texto_limpio: str) -> "VoynichTokenizer":
        """Build vocabulary from cleaned corpus text. Call once before encoding."""
        for token in sorted(set(self._tokenise(texto_limpio))):
            if token not in self.char2i:
                self.char2i[token] = self.vocab_size
                self.i2char[self.vocab_size] = token
                self.vocab_size += 1
        self._fitted = True
        return self

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def codificar(self, texto: str) -> List[int]:
        """Text → list of integer IDs. Unknown tokens map to UNK_ID."""
        if not self._fitted:
            raise RuntimeError("Tokenizer not fitted — call ajustar() first.")
        return [self.char2i.get(t, self.UNK_ID) for t in self._tokenise(texto)]

    def decodificar(self, ids: List[int], skip_special: bool = True) -> str:
        """List of integer IDs → text string."""
        special_ids = set(self.SPECIAL.values()) if skip_special else set()
        tokens = [self.i2char.get(i, "<UNK>") for i in ids if i not in special_ids]
        sep = "" if self.mode == "char" else " "
        return sep.join(tokens)

    def codificar_con_bos_eos(self, texto: str) -> List[int]:
        return [self.BOS_ID] + self.codificar(texto) + [self.EOS_ID]

    # ── Cryptanalytic utilities ───────────────────────────────────────────────

    def calcular_entropia(self, texto: str) -> float:
        """
        Shannon entropy H = -Σ p(c) log₂ p(c)
        Higher entropy → more uniform distribution → more random-looking.
        English ≈ 4.0 bits, Voynich ≈ 3.5–4.0 bits, random ≈ log₂(N) bits.
        """
        chars = texto.replace(" ", "")
        total = len(chars)
        if total == 0:
            return 0.0
        freq = Counter(chars)
        return -sum((n / total) * math.log2(n / total) for n in freq.values())

    def calcular_ic(self, texto: str) -> float:
        """
        Index of Coincidence — probability two randomly chosen letters are equal.
          IC = Σ f(c)(f(c)-1) / N(N-1)
        Reference values: English ≈ 0.065 | Random ≈ 0.038 | Voynich ≈ 0.057
        """
        chars = texto.replace(" ", "")
        n = len(chars)
        if n < 2:
            return 0.0
        freq = Counter(chars)
        return sum(f * (f - 1) for f in freq.values()) / (n * (n - 1))

    def analizar_ngramas(self, texto: str, n: int = 2, top_k: int = 20) -> List[Tuple[str, int]]:
        """
        Character n-gram frequency analysis.
        n=2 (bigrams) and n=3 (trigrams) are most informative for Voynich.
        """
        words = texto.split()
        ngrams: List[str] = []
        for word in words:
            for i in range(len(word) - n + 1):
                ngrams.append(word[i : i + n])
        return Counter(ngrams).most_common(top_k)

    def calcular_kasiski(self, texto: str, min_len: int = 3) -> List[Tuple[str, List[int]]]:
        """
        Kasiski examination: find repeated sequences and their distances.
        Useful for estimating key length in Vigenère-like ciphers.
        Returns [(sequence, [positions, ...]), ...] sorted by occurrence count.
        """
        clean = texto.replace(" ", "")
        results: Dict[str, List[int]] = {}
        for length in range(min_len, min(8, len(clean) // 2)):
            for i in range(len(clean) - length):
                seq = clean[i : i + length]
                if seq not in results:
                    results[seq] = []
                results[seq].append(i)
        # Keep only repeated sequences
        repeated = {seq: pos for seq, pos in results.items() if len(pos) > 1}
        return sorted(repeated.items(), key=lambda x: -len(x[1]))[:15]

    # ── Statistics ────────────────────────────────────────────────────────────

    def obtener_estadisticas(self, texto: str) -> CorpusStats:
        """Full corpus analysis. Returns a CorpusStats dataclass with .render() support."""
        tokens = texto.split()
        chars  = texto.replace(" ", "")
        freq_t = Counter(tokens)
        freq_c = Counter(chars)

        return CorpusStats(
            total_chars      = len(chars),
            total_tokens     = len(tokens),
            unique_tokens    = len(freq_t),
            vocab_size       = self.vocab_size,
            entropy_bits     = round(self.calcular_entropia(texto), 4),
            index_coincid    = round(self.calcular_ic(texto), 6),
            avg_token_len    = round(sum(len(t) for t in tokens) / max(1, len(tokens)), 2),
            hapax_legomena   = sum(1 for v in freq_t.values() if v == 1),
            type_token_ratio = round(len(freq_t) / max(1, len(tokens)), 4),
            top_tokens       = freq_t.most_common(10),
            top_chars        = freq_c.most_common(10),
            top_bigrams      = self.analizar_ngramas(texto, n=2, top_k=10),
        )

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"VoynichTokenizer(mode='{self.mode}', vocab_size={self.vocab_size}, {status})"
