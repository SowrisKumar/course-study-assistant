"""Step 2b – Retrieval: find the chunks most relevant to a question.

Four retrievers with the same interface, so they can be compared directly:

  tfidf   – TF-IDF + cosine (the RAG_primer_fast notebook from class). Keyword match.
  bm25    – BM25, the standard keyword-search ranking (better length normalisation than TF-IDF).
  dense   – sentence embeddings (all-MiniLM-L6-v2, as in the class RAG notebooks) + cosine.
            Matches meaning, e.g. "how do GANs learn" ~ "generator and discriminator training".
  hybrid  – BM25 + dense merged with Reciprocal Rank Fusion (RRF): each list votes
            1 / (60 + rank) for its results. Keyword search catches exact terms
            ("KL divergence", "Bellman"); dense search catches paraphrases.

Every hit carries `dense_score` (cosine similarity to the question) whatever the method,
so the assistant can use one calibrated number to decide "this isn't in the course".

Usage:
    from study_assistant.retrieval import load_chunks, build_retriever
    retriever = build_retriever("hybrid", load_chunks())
    for hit in retriever.search("what is the reparameterization trick?", k=5):
        print(hit.rank, round(hit.score, 3), hit.chunk.citation)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .chunking import EMBED_MODEL, Chunk, chunk_documents
from .config import CACHE_DIR
from .ingest import load_documents

RRF_K = 60  # standard constant from the RRF paper (Cormack et al., 2009)


@dataclass
class Hit:
    chunk: Chunk
    score: float  # the method's own score (higher = more relevant)
    rank: int  # 1-based
    dense_score: float  # cosine similarity between question and chunk embeddings


def load_chunks() -> list[Chunk]:
    """Chunks of the shipped knowledge base."""
    return chunk_documents(load_documents())


# --- dense embeddings (shared) --------------------------------------------------------


class Embedder:
    """Encodes texts with the sentence-transformer; chunk embeddings are cached on disk."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def encode_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        texts = [c.embed_text for c in chunks]
        key = hashlib.sha1((self.model_name + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
        path = CACHE_DIR / "embeddings" / f"{self.model_name.split('/')[-1]}.{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.encode(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs


# --- keyword tokenisation ----------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def _stopwords() -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return ENGLISH_STOP_WORDS


def keyword_tokens(text: str) -> list[str]:
    stop = _stopwords()
    return [t for t in _TOKEN.findall(text.lower()) if t not in stop]


# --- retrievers ---------------------------------------------------------------------------


class Retriever:
    name = "base"

    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.embedder = embedder or Embedder()

    @cached_property
    def chunk_vectors(self) -> np.ndarray:
        return self.embedder.encode_chunks(self.chunks)

    def dense_scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query])[0]
        return self.chunk_vectors @ q

    def scores(self, query: str) -> np.ndarray:  # one score per chunk
        raise NotImplementedError

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.scores(query)
        dense = self.dense_scores(query)
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(self.chunks[i], float(scores[i]), rank, float(dense[i]))
            for rank, i in enumerate(top, start=1)
        ]


class TfidfRetriever(Retriever):
    name = "tfidf"

    @cached_property
    def _index(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(tokenizer=keyword_tokens, lowercase=False, token_pattern=None,
                              ngram_range=(1, 2), sublinear_tf=True)
        return vec, vec.fit_transform([c.embed_text for c in self.chunks])

    def scores(self, query: str) -> np.ndarray:
        vec, matrix = self._index
        return (matrix @ vec.transform([query]).T).toarray().ravel()


class BM25Retriever(Retriever):
    name = "bm25"

    @cached_property
    def _bm25(self):
        from rank_bm25 import BM25Okapi

        return BM25Okapi([keyword_tokens(c.embed_text) for c in self.chunks])

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(keyword_tokens(query)))


class DenseRetriever(Retriever):
    name = "dense"

    def scores(self, query: str) -> np.ndarray:
        return self.dense_scores(query)


class HybridRetriever(Retriever):
    """Reciprocal Rank Fusion of BM25 and dense rankings."""

    name = "hybrid"
    depth = 50  # how far down each ranking contributes votes

    @cached_property
    def _bm25(self) -> BM25Retriever:
        return BM25Retriever(self.chunks, self.embedder)

    def scores(self, query: str) -> np.ndarray:
        fused = np.zeros(len(self.chunks))
        for s in (self._bm25.scores(query), self.dense_scores(query)):
            top = np.argsort(-s, kind="stable")[: self.depth]
            fused[top] += 1.0 / (RRF_K + np.arange(1, len(top) + 1))
        return fused


RETRIEVERS = {r.name: r for r in (TfidfRetriever, BM25Retriever, DenseRetriever, HybridRetriever)}


def build_retriever(name: str, chunks: list[Chunk] | None = None, embedder: Embedder | None = None) -> Retriever:
    if name not in RETRIEVERS:
        raise ValueError(f"Unknown retriever {name!r}; choose from {sorted(RETRIEVERS)}")
    return RETRIEVERS[name](chunks if chunks is not None else load_chunks(), embedder)
