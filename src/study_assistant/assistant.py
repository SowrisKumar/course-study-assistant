"""Step 4 – The study assistant: a LangGraph workflow (same pattern as campus_help_desk_rag.ipynb).

    question
       │
     route ──────────────► chitchat ─────────────────────────────► END
       │ answer / simplify / quiz
    retrieve      (top chunks, grouped into passages per document)
       │
     check        (is the material relevant? off-topic floor + LLM relevance check)
       ├── nothing relevant ─► abstain ("This isn't covered...") ──► END
       │
    generate      (answer / simple explanation / quiz, citing passages as [n])
       │
    cite          (keep only real citations, map [n] -> lecture & slide/section)
       │
      END

Why the relevance check: Step 2 showed that similarity scores can't tell covered from
not-covered questions ("capital of France" scored like a real question, "mode collapse"
beat seven real ones), so the LLM reads the passages and decides.

    from study_assistant.assistant import StudyAssistant
    from study_assistant.llm import get_llm
    bot = StudyAssistant(get_llm("ollama"))
    result = bot.ask("Why do transformers need positional encodings?")
    print(result.format())
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from . import prompts
from .llm import LLM
from .refusal import is_refusal
from .retrieval import Hit, Retriever, build_retriever

Mode = Literal["answer", "simplify", "quiz", "chitchat"]

# Step 5 evaluation (end to end, Llama 3.1): hybrid won 4 questions and lost 1 vs dense after removing
# judge noise, and abstained on one more near-miss. Chosen on the evaluation set, so its scores there
# are slightly optimistic (see docs/design.md, Step 5).
DEFAULT_RETRIEVER = "hybrid"
# Support check (Step 6): None (off), "strict" (is the answer stated in the passages?) or "concept" (do the
# passages discuss the specific thing asked about?). On the hand-labelled development set strict rejected
# 2 correct answers and 1 wrong one; concept rejected the 3 wrong answers and no correct ones. On the
# evaluation set the main system already declined every not-covered question: concept changed nothing and
# strict rejected 1 borderline answer (q02), each adding ~4 s per answer. So it is off by default (docs/report.md, §7).
# Available as StudyAssistant(..., verify="concept") / `chat --verify`.
DEFAULT_VERIFY: str | None = None
VERIFY_VARIANTS = ("strict", "concept")
OFF_TOPIC_FLOOR = 0.25  # below this best cosine score the question is clearly unrelated (Step 2: in-scope min 0.37)


# --- data passed between steps -------------------------------------------------------------


@dataclass
class Passage:
    """One or more retrieved chunks from the same document, shown to the LLM as [number]."""

    number: int
    doc_id: str
    citation: str
    lecture: int | None
    track: str
    text: str
    score: float  # best dense score among its chunks

    @property
    def label(self) -> str:
        where = f"Lecture {self.lecture}" if self.lecture is not None else "Add-on lab"
        return f"{where} · {self.citation}"


@dataclass
class AssistantResult:
    question: str
    mode: Mode
    answer: str
    sources: list[Passage] = field(default_factory=list)  # passages actually cited, renumbered [1..n]
    abstained: bool = False
    abstain_reason: str = ""
    uncited: bool = False  # an answer was given but cited nothing (a grounding warning)
    passages: list[Passage] = field(default_factory=list)  # everything retrieved (for inspection/eval)
    trace: list[str] = field(default_factory=list)

    def format(self) -> str:
        out = self.answer
        if self.sources:
            out += "\n\nSources:\n" + "\n".join(f"[{p.number}] {p.label}" for p in self.sources)
        if self.uncited:
            out += "\n\n(Note: this answer did not cite the course materials; treat it with caution.)"
        return out


class State(TypedDict, total=False):
    question: str
    mode: Mode
    search_query: str
    passages: list[Passage]
    relevant: list[Passage]
    answer: str
    sources: list[Passage]
    abstained: bool
    abstain_reason: str
    uncited: bool
    trace: Annotated[list[str], operator.add]  # each step appends a line


# --- routing (rule-based, like the class router: cheap, predictable, no LLM call) -----------

_QUIZ = re.compile(r"\b(quiz|test me|practice (questions?|test)|mcqs?|multiple[- ]choice)\b", re.I)
# Explicit phrases only: a bare "simple" would misroute questions like "How do LSTMs differ from simple RNNs?"
_SIMPLIFY = re.compile(
    r"\b(simpler|simply|simplify|in simple (terms|words|language)|eli5|explain like i'?m (5|five)|"
    r"plain (english|words|language)|for a beginner|dumb(ed)? (it )?down|easier (way|terms)|layman'?s?)\b", re.I)
_CHITCHAT = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|thx|bye|good (morning|evening))\b[\s!.?]*$", re.I)
_FILLER = re.compile(
    r"\b(can you|could you|please|give me|make( me)?|create|write|generate|i want|a|an|short|quick|"
    r"quiz|test me|practice|questions?|mcqs?|multiple[- ]choice|explain|in|simple|simpler|simply|simplify|"
    r"terms|words|language|plain|english|eli5|like i'?m (5|five)|for a beginner|on|about|of|the|topic|me|more|to)\b", re.I)


def route_question(question: str) -> tuple[Mode, str]:
    """Return (mode, search query). The search query drops mode words like "quiz me on"."""
    if _CHITCHAT.match(question):
        return "chitchat", ""
    mode: Mode = "quiz" if _QUIZ.search(question) else "simplify" if _SIMPLIFY.search(question) else "answer"
    if mode == "answer":
        return mode, question
    topic = re.sub(r"\s+", " ", _FILLER.sub(" ", question)).strip(" ?.!,:")
    return mode, topic if len(topic) >= 3 else question


# --- helpers ------------------------------------------------------------------------------------


def group_into_passages(hits: list[Hit], max_passages: int) -> list[Passage]:
    """Merge chunks of the same document (in reading order) so each source is cited once."""
    by_doc: dict[str, list[Hit]] = {}
    for h in hits:  # hits are in rank order, so dict order = rank of each document's best chunk
        by_doc.setdefault(h.chunk.doc_id, []).append(h)
    passages = []
    for n, (doc_id, doc_hits) in enumerate(list(by_doc.items())[:max_passages], start=1):
        c = doc_hits[0].chunk
        text = "\n\n".join(h.chunk.text for h in sorted(doc_hits, key=lambda h: h.chunk.part))
        passages.append(Passage(n, doc_id, c.citation, c.lecture, c.track, text,
                                max(h.dense_score for h in doc_hits)))
    return passages


def format_passages(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{p.number}] ({p.label})\n{p.text}" for p in passages)


def parse_relevant(reply: str, n: int) -> list[int] | None:
    """Passage numbers from 'RELEVANT: 1, 3' (or [] for 'none'); None if the reply is unusable."""
    m = re.search(r"RELEVANT\s*:\s*(.*)", reply, re.I)
    if not m:
        return None
    line = m.group(1)
    if re.match(r"\s*none\b", line, re.I):
        return []
    nums = sorted({int(x) for x in re.findall(r"\d+", line) if 1 <= int(x) <= n})
    return nums if nums else None


# A citation is [n] or [n, m] NOT directly after an identifier or ")" (so `x[1]`, `f(x)[0]` are code),
# and never inside inline code or a fenced code block. "]" is allowed before it: chained citations [1][2].
_CITE = re.compile(r"(?<![\w)])\[(\d+(?:\s*,\s*\d+)*)\]")
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


def sub_citations(text: str, fn) -> str:
    """Apply `fn(match)` to every citation marker outside code spans."""
    out, pos = [], 0
    for m in _CODE.finditer(text):
        out += [_CITE.sub(fn, text[pos:m.start()]), m.group(0)]
        pos = m.end()
    out.append(_CITE.sub(fn, text[pos:]))
    return "".join(out)


is_not_covered = is_refusal  # shared with the evaluation (refusal.py)


# --- the assistant ----------------------------------------------------------------------------


class StudyAssistant:
    """
    llm:          writes the answers (any backend from study_assistant.llm)
    retriever:    Step 2 retriever (default: hybrid)
    grader_llm:   LLM for the relevance check (default: same as llm)
    grading:      "llm" (default) | "threshold" (score >= threshold; for weak models like flan-t5) | "none"
    verify:       support check after answering (Step 6): None | "strict" | "concept" (True = "strict");
                  abstain if it fails
    k:            chunks to retrieve; max_passages: documents shown to the LLM
    """

    def __init__(self, llm: LLM, retriever: Retriever | None = None, *, grader_llm: LLM | None = None,
                 grading: str = "llm", threshold: float = 0.45, k: int = 8, max_passages: int = 5,
                 off_topic_floor: float = OFF_TOPIC_FLOOR, verify: str | bool | None = DEFAULT_VERIFY):
        if grading not in ("llm", "threshold", "none"):
            raise ValueError("grading must be 'llm', 'threshold' or 'none'")
        verify = "strict" if verify is True else (verify or None)
        if verify not in (None, *VERIFY_VARIANTS):
            raise ValueError(f"verify must be None or one of {VERIFY_VARIANTS}")
        self.llm = llm
        self.retriever = retriever or build_retriever(DEFAULT_RETRIEVER)
        self.grader_llm = grader_llm or llm
        self.grading, self.threshold, self.verify = grading, threshold, verify
        self.k, self.max_passages, self.off_topic_floor = k, max_passages, off_topic_floor
        self.graph = self._build_graph()

    # ---- nodes ----

    def _route(self, state: State) -> State:
        mode, query = route_question(state["question"])
        return {"mode": mode, "search_query": query, "trace": [f"route: mode={mode}, search={query!r}"]}

    def _chitchat(self, state: State) -> State:
        return {"answer": prompts.CHITCHAT_REPLY, "abstained": False, "trace": ["chitchat: canned reply"]}

    def _retrieve(self, state: State) -> State:
        hits = self.retriever.search(state["search_query"], k=self.k)
        passages = group_into_passages(hits, self.max_passages)
        lines = [f"retrieve ({self.retriever.name}): {len(hits)} chunks -> {len(passages)} passages"]
        lines += [f"  [{p.number}] {p.score:.2f} {p.label}" for p in passages]
        return {"passages": passages, "trace": lines}

    def _check(self, state: State) -> State:
        passages = state["passages"]
        best = max((p.score for p in passages), default=0.0)
        if best < self.off_topic_floor:
            return {"relevant": [], "abstain_reason": f"off-topic (best similarity {best:.2f} < {self.off_topic_floor})",
                    "trace": [f"check: best similarity {best:.2f} below floor -> abstain"]}

        if self.grading == "none":
            return {"relevant": passages, "trace": ["check: skipped (grading='none')"]}
        if self.grading == "threshold":
            keep = [p for p in passages if p.score >= self.threshold]
            return {"relevant": keep, "abstain_reason": "" if keep else f"no passage scored >= {self.threshold}",
                    "trace": [f"check: threshold {self.threshold} keeps {[p.number for p in keep]}"]}

        # LLM relevance check: one call that grades all passages
        reply = self.grader_llm.generate(
            prompts.GRADER_PROMPT.format(question=state["question"], passages=format_passages(passages)),
            system=prompts.GRADER_SYSTEM, max_tokens=30,
        ).text
        nums = parse_relevant(reply, len(passages))
        if nums is None:  # unusable reply: fail open, the generator can still say "not covered"
            return {"relevant": passages, "trace": [f"check: unparseable grader reply {reply!r} -> keep all"]}
        keep = [p for p in passages if p.number in nums]
        return {"relevant": keep, "abstain_reason": "" if keep else "relevance check found no relevant passage",
                "trace": [f"check: grader says relevant={nums or 'none'}"]}

    def _abstain(self, state: State) -> State:
        return {"answer": prompts.NOT_COVERED, "abstained": True, "sources": [], "trace": ["abstain"]}

    def _generate(self, state: State) -> State:
        # renumber the relevant passages 1..n so the model sees a clean list
        shown = [Passage(i, p.doc_id, p.citation, p.lecture, p.track, p.text, p.score)
                 for i, p in enumerate(state["relevant"], start=1)]
        mode = state["mode"]
        task = prompts.TASKS[mode].format(question=state["search_query"] if mode == "quiz" else state["question"])
        reply = self.llm.generate(
            prompts.GENERATE_PROMPT.format(passages=format_passages(shown), task=task),
            system=prompts.SYSTEMS[mode], max_tokens=900 if mode == "quiz" else 400,
        )
        return {"answer": reply.text, "relevant": shown,
                "trace": [f"generate ({reply.backend}/{reply.model}, {reply.latency_s:.1f}s)"]}

    def _cite(self, state: State) -> State:
        answer, shown = state["answer"], state["relevant"]
        if is_not_covered(answer):
            return {"answer": prompts.NOT_COVERED, "abstained": True, "sources": [],
                    "abstain_reason": "the model found the passages insufficient",
                    "trace": ["cite: model answered 'not covered' -> abstain"]}

        valid = {p.number for p in shown}
        used: list[int] = []

        def fix(m: re.Match) -> str:
            nums = [int(x) for x in m.group(1).split(",")]
            good = [n for n in nums if n in valid]
            used.extend(n for n in good if n not in used)
            return "".join(f"[{n}]" for n in good)  # drop citations to passages that don't exist

        answer = sub_citations(answer, fix)
        # renumber cited passages in order of first use: [1], [2], ... matching the Sources list
        remap = {old: new for new, old in enumerate(used, start=1)}
        answer = sub_citations(answer, lambda m: f"[{remap[int(m.group(1))]}]")
        sources = [Passage(remap[p.number], p.doc_id, p.citation, p.lecture, p.track, p.text, p.score)
                   for p in sorted(shown, key=lambda p: remap.get(p.number, 99)) if p.number in remap]
        return {"answer": answer.strip(), "sources": sources, "abstained": False, "uncited": not sources,
                "trace": [f"cite: {len(sources)} source(s) cited" + ("" if sources else " -> WARNING: uncited answer")]}

    def _verify(self, state: State) -> State:
        """Support check: is the answer's main content actually in the passages? If not, abstain.

        Only for plain answers: simple explanations add analogies and quizzes add distractors on
        purpose, which a strict support check would wrongly reject.
        """
        if not self.verify or state.get("abstained") or state["mode"] != "answer":
            return {"trace": []}
        # Check against ALL passages the model was shown, not only the cited ones: on the development set a
        # correct answer drew on slide 5 but cited only slide 6 and was wrongly rejected. Citation numbers
        # are removed from both sides so the checker judges content, not numbering.
        passages = "\n\n".join(f"({p.label})\n{p.text}" for p in state["relevant"])
        answer = re.sub(r"[ \t]+([.,;:!?])", r"\1", sub_citations(state["answer"], lambda m: "")).strip()
        if self.verify == "concept":
            prompt = prompts.CONCEPT_PROMPT.format(question=state["question"], passages=passages)
            system, field = prompts.CONCEPT_SYSTEM, "DISCUSSED"
        else:
            prompt = prompts.VERIFY_PROMPT.format(question=state["question"], passages=passages, answer=answer)
            system, field = prompts.VERIFY_SYSTEM, "SUPPORTED"
        reply = self.grader_llm.generate(prompt, system=system, max_tokens=10).text
        m = re.search(field + r"\s*:\s*(yes|no)", reply, re.I)
        if m and m.group(1).lower() == "no":
            return {"answer": prompts.NOT_COVERED, "abstained": True, "sources": [], "uncited": False,
                    "abstain_reason": f"support check ({self.verify}): answer not supported by the passages",
                    "trace": [f"verify ({self.verify}): answer NOT supported by the passages -> abstain"]}
        verdict = "supported" if m else f"unparseable reply {reply!r} -> keep answer"
        return {"trace": [f"verify ({self.verify}): {verdict}"]}

    # ---- graph ----

    def _build_graph(self):
        g = StateGraph(State)
        g.add_node("route", self._route)
        g.add_node("chitchat", self._chitchat)
        g.add_node("retrieve", self._retrieve)
        g.add_node("check", self._check)
        g.add_node("abstain", self._abstain)
        g.add_node("generate", self._generate)
        g.add_node("cite", self._cite)
        g.add_node("verify", self._verify)

        g.add_edge(START, "route")
        g.add_conditional_edges("route", lambda s: "chitchat" if s["mode"] == "chitchat" else "retrieve",
                                {"chitchat": "chitchat", "retrieve": "retrieve"})
        g.add_edge("retrieve", "check")
        g.add_conditional_edges("check", lambda s: "generate" if s["relevant"] else "abstain",
                                {"generate": "generate", "abstain": "abstain"})
        g.add_edge("generate", "cite")
        g.add_edge("cite", "verify")
        g.add_edge("verify", END)
        g.add_edge("abstain", END)
        g.add_edge("chitchat", END)
        return g.compile()

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def ask(self, question: str) -> AssistantResult:
        s = self.graph.invoke({"question": question, "trace": []})
        return AssistantResult(
            question=question,
            mode=s["mode"],
            answer=s.get("answer", ""),
            sources=s.get("sources", []),
            abstained=s.get("abstained", False),
            abstain_reason=s.get("abstain_reason", ""),
            uncited=s.get("uncited", False),
            passages=s.get("passages", []),
            trace=s.get("trace", []),
        )
