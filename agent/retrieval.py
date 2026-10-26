"""Retrieval: chunk docs -> embed chunks -> store in a NumPy array -> cosine top-k.

Three interchangeable embedders (set with RAG_EMBEDDER or --embedder):
  local   sentence-transformers all-MiniLM-L6-v2. Free, runs on your laptop,
          downloads ~90 MB from Hugging Face the first time.          (default)
  voyage  Voyage AI API (voyage-4-lite by default). Needs VOYAGE_API_KEY.
  tfidf   TF-IDF vectors built from the corpus itself. No download, no key.
          A sparse baseline; useful offline, in tests and in CI.

Embeddings are cached in data/index/ and rebuilt when docs or settings change.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "data" / "index"

DEFAULT_EMBEDDER = os.environ.get("RAG_EMBEDDER", "local")
DEFAULT_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", 500))
DEFAULT_OVERLAP = int(os.environ.get("RAG_OVERLAP", 50))


# ---------------------------------------------------------------- chunking

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Approximate token count (words + punctuation). Within ~15% of real BPE
    tokenizers on English prose, and needs no tokenizer download."""
    return len(_TOKEN_RE.findall(text))


@dataclass
class Chunk:
    id: str      # "expense-policy#2"
    doc: str     # file stem, "expense-policy"
    title: str   # document title from the first "# " line
    text: str


def _units(markdown: str) -> list[str]:
    """Split a doc into small units: headings, list items and sentences."""
    units = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("#", "-", "*")) or re.match(r"\d+\.", line):
            units.append(line)
        else:
            units += [s for s in re.split(r"(?<=[.!?])\s+", line) if s]
    return units


def chunk_doc(doc: str, markdown: str, chunk_size: int, overlap: int,
              heading_aware: bool = True) -> list[Chunk]:
    """Greedily pack units into chunks of <= chunk_size tokens. Each new chunk
    starts with the trailing units of the previous one (up to `overlap` tokens)
    so a fact near a boundary is not cut off from its context.

    heading_aware=True moves a heading that would end a chunk into the next one,
    so a section's text is never separated from its heading. (Turning it off is
    the "before" setting of the experiment recorded in the README.)"""
    title = markdown.strip().splitlines()[0].lstrip("# ").strip()
    units = _units(markdown)
    chunks, current, size = [], [], 0
    for unit in units:
        n = count_tokens(unit)
        if current and size + n > chunk_size:
            # Never end a chunk on a heading; it belongs with the text below it.
            heading = []
            while heading_aware and current and current[-1].startswith("#"):
                heading.insert(0, current.pop())
            if not current:  # chunk was only headings; keep them together
                current, heading = heading, []
            chunks.append(current)
            carry, carried = [], 0
            for prev in reversed(current):
                t = count_tokens(prev)
                if carried + t > overlap:
                    break
                carry.insert(0, prev)
                carried += t
            current = carry + heading
            size = sum(count_tokens(u) for u in current)
        current.append(unit)
        size += n
    if current:
        chunks.append(current)
    return [Chunk(f"{doc}#{i}", doc, title, "\n".join(c)) for i, c in enumerate(chunks)]


def load_chunks(chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP,
                heading_aware: bool = True, doc_dir: Path = DOC_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(doc_dir).glob("*.md")):
        chunks += chunk_doc(path.stem, path.read_text(encoding="utf-8"), chunk_size, overlap, heading_aware)
    return chunks


# ---------------------------------------------------------------- embedders

class LocalEmbedder:
    name = "local"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.model = SentenceTransformer(model)

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


class VoyageEmbedder:
    name = "voyage"

    def __init__(self, model: str = os.environ.get("VOYAGE_MODEL", "voyage-4-lite")):
        import voyageai
        self.model_name = model
        self.client = voyageai.Client()  # reads VOYAGE_API_KEY

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 128):
            out += self.client.embed(texts[i:i + 128], model=self.model_name, input_type=kind).embeddings
        return np.asarray(out, dtype=np.float32)


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an and are as at be by can do does for from has have how i if in is it its many much my "
    "of on or our per should so that the their there this to was we what when where which who "
    "will with you your".split()
)


def _stem(t: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            return t[: -len(suffix)]
    return t


def _words(text: str) -> list[str]:
    return [_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOP]


class TfidfEmbedder:
    """Fits a vocabulary on the corpus, then encodes texts as L2-normalized
    TF-IDF vectors, so cosine similarity works exactly as with dense vectors."""
    name = "tfidf"
    model_name = "tfidf"

    def fit(self, texts: list[str]) -> None:
        docs = [set(_words(t)) for t in texts]
        df = Counter(w for d in docs for w in d)
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in sorted(df)], dtype=np.float32)

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for w, c in Counter(_words(text)).items():
                if w in self.vocab:
                    mat[row, self.vocab[w]] = 1 + math.log(c)
        mat *= self.idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1, norms)


def make_embedder(name: str):
    if name == "local":
        return LocalEmbedder()
    if name == "voyage":
        return VoyageEmbedder()
    if name == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"unknown embedder {name!r}; use local, voyage or tfidf")


# ---------------------------------------------------------------- index

class VectorIndex:
    def __init__(self, embedder: str = DEFAULT_EMBEDDER, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 overlap: int = DEFAULT_OVERLAP, title_prefix: bool = False,
                 heading_aware: bool = True, use_cache: bool = True):
        self.chunk_size, self.overlap, self.title_prefix = chunk_size, overlap, title_prefix
        self.chunks = load_chunks(chunk_size, overlap, heading_aware)
        self.embedder = make_embedder(embedder)
        texts = [self._embed_text(c) for c in self.chunks]

        if isinstance(self.embedder, TfidfEmbedder):
            self.embedder.fit(texts)  # cheap; never cached
            self.vectors = self.embedder.embed(texts, "document")
            return

        key = hashlib.sha256(json.dumps(
            [self.embedder.model_name, texts]).encode()).hexdigest()[:16]
        cache = INDEX_DIR / f"{self.embedder.name}-{chunk_size}-{overlap}-{key}.npz"
        if use_cache and cache.exists():
            self.vectors = np.load(cache)["vectors"]
        else:
            self.vectors = self.embedder.embed(texts, "document").astype(np.float32)
            if use_cache:
                INDEX_DIR.mkdir(parents=True, exist_ok=True)
                np.savez(cache, vectors=self.vectors)
                (cache.with_suffix(".json")).write_text(json.dumps([asdict(c) for c in self.chunks]))

    def _embed_text(self, c: Chunk) -> str:
        # Optional: prepend the doc title so a chunk that never names its topic
        # ("Up to 5 days carry over...") still embeds near "PTO".
        return f"{c.title}\n{c.text}" if self.title_prefix else c.text

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([query], "query")[0]
        scores = self.vectors @ q  # vectors are unit length, so dot product = cosine similarity
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]
