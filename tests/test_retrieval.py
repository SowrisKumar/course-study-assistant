from __future__ import annotations

import pytest

from study_assistant.chunking import chunk_documents
from study_assistant.ingest.schema import Document
from study_assistant.retrieval import RETRIEVERS, build_retriever, keyword_tokens

TOPICS = {
    "vae": "The variational autoencoder encoder outputs a mean and log variance; the KL divergence regularises the latent space.",
    "gan": "A generator network and a discriminator network compete: the discriminator learns to tell real images from fakes.",
    "rl": "The Bellman equation expresses the value of a state as reward plus the discounted value of the next state.",
    "pool": "Max pooling keeps the largest value in each window, halving the width and height of the feature map.",
}


@pytest.fixture(scope="module")
def chunks():
    docs = [
        Document(doc_id=f"{k}::slide-1", text=t, source_file=f"{k}.pptx", source_type="slides",
                 title=k.upper(), section=k, locator="slide 1", lecture=i, track="lecture")
        for i, (k, t) in enumerate(TOPICS.items(), start=1)
    ]
    return chunk_documents(docs)


def test_keyword_tokens_drop_stopwords_and_punctuation():
    assert keyword_tokens("What is the KL-divergence of a VAE?") == ["kl", "divergence", "vae"]


@pytest.mark.parametrize("method", sorted(RETRIEVERS))
@pytest.mark.parametrize(
    "query,expected",
    [("What is the Bellman equation?", "rl"), ("How does the discriminator work?", "gan"), ("KL divergence in a VAE", "vae")],
)
def test_each_method_finds_the_matching_topic(chunks, method, query, expected):
    hits = build_retriever(method, chunks).search(query, k=2)
    assert hits[0].chunk.source_file == f"{expected}.pptx"
    assert [h.rank for h in hits] == [1, 2]
    assert -1.0 <= hits[0].dense_score <= 1.0


def test_dense_matches_meaning_without_shared_words(chunks):
    # no keyword overlap with the pooling text, but the same meaning
    hits = build_retriever("dense", chunks).search("shrinking an image while keeping the strongest features", k=1)
    assert hits[0].chunk.source_file == "pool.pptx"


def test_unknown_method_raises(chunks):
    with pytest.raises(ValueError):
        build_retriever("nope", chunks)
