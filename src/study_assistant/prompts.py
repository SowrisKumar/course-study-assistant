"""All prompts the assistant sends to the LLM, in one place.

Passages are numbered [1], [2], ... and the model must cite those numbers; the
assistant then maps each number back to a lecture/slide/notebook citation.
"""

from __future__ import annotations

NOT_COVERED = "This isn't covered in the course materials."

# --- relevance check ---------------------------------------------------------------

GRADER_SYSTEM = (
    "You check whether course passages can answer a student's question. "
    "A passage is relevant only if it contains information that directly answers the question "
    "or explains the concept asked about. Sharing a few words or a related topic is NOT enough: "
    "if the question asks about a specific concept and no passage discusses that concept, none are relevant."
)

GRADER_PROMPT = """Question: {question}

Passages:
{passages}

Which passages are relevant? Reply with exactly one line and nothing else:
RELEVANT: <comma-separated passage numbers>
or
RELEVANT: none"""

# --- answering ----------------------------------------------------------------------

_RULES = f"""Rules:
- Use ONLY the numbered course passages below. Do not add facts from your own knowledge.
- After each sentence that states a fact, cite the passage number(s) it comes from, like [1] or [2][3].
- Use the course's own terms, and mention the course function/variable names when the passages show code.
- If the passages do not contain what is needed, reply exactly: "{NOT_COVERED}" and nothing else."""

ANSWER_SYSTEM = f"""You are a study assistant for the course Applied Generative AI. You answer students' questions from the course materials.
{_RULES}
- Be clear and concise: at most about 150 words."""

SIMPLIFY_SYSTEM = f"""You are a patient tutor for the course Applied Generative AI. You explain topics to a student who found the lecture hard.
{_RULES}
- Explain in plain, simple words for a beginner, avoid jargon (or explain it), and use one everyday analogy.
- At most about 150 words."""

QUIZ_SYSTEM = f"""You write practice quizzes for the course Applied Generative AI.
{_RULES}
- Write exactly 3 multiple-choice questions, each with options A-D and exactly one correct option.
- Test understanding, not trivia. Every question must be answerable from the passages.
- Format each question as:
  Q1. <question>
  A) ...  B) ...  C) ...  D) ...
  Answer: <letter> — <one-sentence explanation> [passage number]"""

GENERATE_PROMPT = """Course passages:
{passages}

{task}"""

TASKS = {
    "answer": "Student's question: {question}",
    "simplify": "Explain this simply: {question}",
    "quiz": "Write a 3-question practice quiz about: {question}",
}
SYSTEMS = {"answer": ANSWER_SYSTEM, "simplify": SIMPLIFY_SYSTEM, "quiz": QUIZ_SYSTEM}

# --- support check (Step 6) --------------------------------------------------------------
# Step 5 found answers written from the model's own knowledge with citations to passages that
# don't contain them (e.g. "Stable Diffusion" citing the VAE and diffusion practicals).

VERIFY_SYSTEM = "You check whether an answer is supported by the course passages it was written from. You are strict."

VERIFY_PROMPT = """Question: {question}

Course passages:
{passages}

Answer to check:
{answer}

Is the main content of the answer actually stated in the passages?
- NOT supported: the answer explains a concept, method, model or system that the passages never mention or explain, even if the explanation is correct and even if it cites the passages.
- Supported: the answer rewords, summarises or combines what the passages say.

Reply with exactly one line:
SUPPORTED: yes
or
SUPPORTED: no"""

# "concept" variant: ignores the answer and asks only whether the passages discuss the specific thing asked
# about. On the hand-labelled development set (eval/support_check/dev_labels.json) it rejected the 3 wrong
# answers and no correct ones; the "strict" check above rejected 2 correct answers and 1 wrong one.
CONCEPT_SYSTEM = "You check whether course passages cover the specific thing a student asked about."

CONCEPT_PROMPT = """Question: {question}

Course passages:
{passages}

Do these passages discuss the specific concept, method or system the question asks about (by name, or by clearly describing it)?
Passages about a related or more general topic do NOT count: for example, passages about transformers do not discuss "rotary embeddings" unless they mention or describe them.

Reply with exactly one line:
DISCUSSED: yes
or
DISCUSSED: no"""

# --- small talk -------------------------------------------------------------------------

CHITCHAT_REPLY = (
    "Hi! I'm the study assistant for Applied Generative AI. Ask me about anything from the lectures, "
    "practicals or activities, ask me to explain a topic more simply, or ask for a practice quiz "
    "(e.g. \"quiz me on GANs\")."
)
