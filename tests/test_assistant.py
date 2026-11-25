"""The assistant graph, tested with a scripted fake LLM (deterministic, no Ollama/keys needed)."""

from __future__ import annotations

import pytest

from study_assistant import prompts
from study_assistant.assistant import StudyAssistant, parse_relevant, route_question
from study_assistant.chunking import chunk_documents
from study_assistant.ingest.schema import Document
from study_assistant.llm import LLM, LLMResponse
from study_assistant.retrieval import build_retriever

TOPICS = {
    "rag": ("RAG retrieves relevant documents and adds them to the prompt so the model answers from them.", 3),
    "pool": ("Max pooling keeps the largest value in each window, halving the width and height of the feature map.", 4),
    "gan": ("A generator and a discriminator compete: the discriminator learns to tell real images from fakes.", 6),
}


class ScriptedLLM(LLM):
    """Answers the relevance check with `grader_reply` and everything else with `answer_reply`."""

    name = "scripted"

    def __init__(self, grader_reply="RELEVANT: 1", answer_reply="An answer [1]."):
        super().__init__("fake")
        self.grader_reply, self.answer_reply = grader_reply, answer_reply
        self.calls: list[tuple[str, str]] = []  # (kind, prompt)

    def _generate(self, prompt, system, max_tokens, temperature):
        kind = "grader" if system == prompts.GRADER_SYSTEM else "generate"
        self.calls.append((kind, prompt))
        return LLMResponse(self.grader_reply if kind == "grader" else self.answer_reply, self.name, self.model, 0.0)


@pytest.fixture(scope="module")
def retriever():
    docs = [Document(doc_id=f"{k}.pptx::slide-1", text=t, source_file=f"{k}.pptx", source_type="slides",
                     title=f"Deck {k}", section=k, locator="slide 1", lecture=lec, track="lecture")
            for k, (t, lec) in TOPICS.items()]
    return build_retriever("dense", chunk_documents(docs))


def make(retriever, llm, **kw):
    # floor -1 = off (cosine can be negative for unrelated text); tests that need it pass floor=...
    return StudyAssistant(llm, retriever, k=3, max_passages=3, off_topic_floor=kw.pop("floor", -1.0), **kw)


# --- routing / parsing -------------------------------------------------------------------


@pytest.mark.parametrize("q,mode,query", [
    ("How do LSTMs differ from simple RNNs?", "answer", "How do LSTMs differ from simple RNNs?"),
    ("Explain VAEs more simply", "simplify", "VAEs"),
    ("Can you explain diffusion models in simple terms?", "simplify", "diffusion models"),
    ("Quiz me on GANs", "quiz", "GANs"),
    ("Give me a short practice quiz about the Bellman equation", "quiz", "Bellman equation"),
    ("hello!", "chitchat", ""),
    ("Hi, what is a GAN?", "answer", "Hi, what is a GAN?"),
])
def test_route_question(q, mode, query):
    assert route_question(q) == (mode, query)


@pytest.mark.parametrize("reply,expected", [
    ("RELEVANT: 1, 3", [1, 3]), ("relevant: none", []), ("RELEVANT: [2]", [2]),
    ("RELEVANT: 9", None), ("I think passage 2", None),
])
def test_parse_relevant(reply, expected):
    assert parse_relevant(reply, n=3) == expected


# --- graph paths -------------------------------------------------------------------------------


def test_answer_with_citation(retriever):
    llm = ScriptedLLM("RELEVANT: 1", "Pooling keeps the largest value [1].")
    r = make(retriever, llm).ask("What does max pooling do?")
    assert not r.abstained and r.mode == "answer"
    assert r.answer == "Pooling keeps the largest value [1]."
    assert [s.label for s in r.sources] == ["Lecture 4 · Deck pool, slide 1"]
    assert [k for k, _ in llm.calls] == ["grader", "generate"]


def test_citations_are_validated_and_renumbered(retriever):
    # passage 2 cited first, passage 7 doesn't exist
    llm = ScriptedLLM("RELEVANT: 1, 2, 3", "First fact [2]. Second fact [1][7]. Both [1, 2].")
    r = make(retriever, llm).ask("What does max pooling do?")
    assert r.answer == "First fact [1]. Second fact [2]. Both [2][1]."
    assert [s.number for s in r.sources] == [1, 2]
    assert len(r.sources) == 2


def test_grader_none_abstains_without_generating(retriever):
    llm = ScriptedLLM("RELEVANT: none")
    r = make(retriever, llm).ask("What is the capital of France?")
    assert r.abstained and r.answer == prompts.NOT_COVERED and r.sources == []
    assert [k for k, _ in llm.calls] == ["grader"]


def test_generator_saying_not_covered_abstains(retriever):
    r = make(retriever, ScriptedLLM("RELEVANT: 1", prompts.NOT_COVERED)).ask("What is mode collapse?")
    assert r.abstained and "insufficient" in r.abstain_reason


def test_off_topic_floor_abstains_without_any_llm_call(retriever):
    llm = ScriptedLLM()
    r = make(retriever, llm, floor=0.99).ask("What is the capital of France?")
    assert r.abstained and "off-topic" in r.abstain_reason and llm.calls == []


def test_uncited_answer_is_flagged(retriever):
    r = make(retriever, ScriptedLLM("RELEVANT: 1", "An answer with no citations.")).ask("What does max pooling do?")
    assert r.uncited and not r.abstained and "did not cite" in r.format()


def test_unparseable_grader_reply_keeps_all_passages(retriever):
    llm = ScriptedLLM("I'm not sure.", "Answer [3].")
    r = make(retriever, llm).ask("What does max pooling do?")
    assert not r.abstained and len(r.sources) == 1
    assert "[3]" in llm.calls[-1][1]  # all three passages were shown to the generator


def test_threshold_grading_makes_no_grader_call(retriever):
    llm = ScriptedLLM(answer_reply="Answer [1].")
    r = make(retriever, llm, grading="threshold", threshold=0.0).ask("What does max pooling do?")
    assert not r.abstained and [k for k, _ in llm.calls] == ["generate"]


def test_quiz_uses_topic_and_quiz_prompt(retriever):
    llm = ScriptedLLM("RELEVANT: 1", "Q1. ... Answer: B [1]")
    r = make(retriever, llm).ask("Quiz me on GANs")
    assert r.mode == "quiz" and not r.abstained
    assert "Write a 3-question practice quiz about: GANs" in llm.calls[-1][1]


def test_chitchat_makes_no_llm_call(retriever):
    llm = ScriptedLLM()
    r = make(retriever, llm).ask("hello")
    assert r.mode == "chitchat" and r.answer == prompts.CHITCHAT_REPLY and llm.calls == []


# --- support check (Step 6) --------------------------------------------------------------------


class VerifyingLLM(ScriptedLLM):
    """Like ScriptedLLM, but answers the support check with `verify_reply`."""

    def __init__(self, verify_reply, **kw):
        super().__init__(**kw)
        self.verify_reply = verify_reply

    def _generate(self, prompt, system, max_tokens, temperature):
        if system == prompts.VERIFY_SYSTEM:
            self.calls.append(("verify", prompt))
            return LLMResponse(self.verify_reply, self.name, self.model, 0.0)
        return super()._generate(prompt, system, max_tokens, temperature)


def test_verify_unsupported_answer_abstains(retriever):
    llm = VerifyingLLM("SUPPORTED: no", grader_reply="RELEVANT: 1", answer_reply="Stable Diffusion uses a VAE [1].")
    r = make(retriever, llm, verify=True).ask("What does max pooling do?")
    assert r.abstained and r.answer == prompts.NOT_COVERED and r.sources == []
    assert "support check" in r.abstain_reason
    assert [k for k, _ in llm.calls] == ["grader", "generate", "verify"]


def test_verify_sees_all_shown_passages_without_citation_numbers(retriever):
    llm = VerifyingLLM("SUPPORTED: yes", grader_reply="RELEVANT: 1, 2, 3", answer_reply="Fact [2].")
    r = make(retriever, llm, verify=True).ask("What does max pooling do?")
    assert not r.abstained and r.answer == "Fact [1]."
    verify_prompt = llm.calls[-1][1]
    for deck in ("Deck pool", "Deck gan", "Deck rag"):  # all three shown passages, not just the cited one
        assert deck in verify_prompt
    assert "Answer to check:\nFact." in verify_prompt  # citation markers stripped


@pytest.mark.parametrize("answer,expected", [
    (prompts.NOT_COVERED, True),
    ("The reward model is not explicitly mentioned. However, typically ... \n\nThis isn't covered in the course materials.", True),
    ("This isn’t covered in the course materials.", True),  # curly apostrophe
    ("", True),
    ("Pooling keeps the largest value [1].", False),
])
def test_is_not_covered(answer, expected):
    from study_assistant.assistant import is_not_covered

    assert is_not_covered(answer) is expected


def test_verify_unparseable_reply_keeps_answer(retriever):
    r = make(retriever, VerifyingLLM("hmm", answer_reply="Fact [1]."), verify=True).ask("What does max pooling do?")
    assert not r.abstained


def test_verify_skipped_for_quiz_and_when_disabled(retriever):
    llm = VerifyingLLM("SUPPORTED: no", answer_reply="Q1 ... [1]")
    assert not make(retriever, llm, verify=True).ask("Quiz me on GANs").abstained
    llm2 = VerifyingLLM("SUPPORTED: no", answer_reply="Fact [1].")
    assert not make(retriever, llm2, verify=False).ask("What does max pooling do?").abstained
    assert "verify" not in [k for k, _ in llm.calls + llm2.calls]
