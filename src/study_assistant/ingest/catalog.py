"""Which course files we ingest, and what each one *is*.

Why an explicit catalog?
  The practical notebooks are numbered differently from the lectures
  (e.g. `Module_4_Practical_VAE.ipynb` belongs to the **Module 5** VAE lecture,
  `Module_8_Practical_GPT.ipynb` to the **Module 9** Transformer lecture).
  If we trusted the file names, citations would send students to the wrong lecture.
  The mapping below was made by comparing each notebook's content with the
  slide titles of every deck (see docs/design.md, "Lecture mapping").

Files not listed here still get ingested through `classify()`'s fallback rules,
so new course materials are picked up automatically (just with less precise metadata).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pptx", ".ipynb", ".pdf"}

# Folders inside the source folder that are never course content.
SKIP_PARTS = {".ipynb_checkpoints", "__pycache__", "shared"}


@dataclass(frozen=True)
class FileInfo:
    source_type: str  # "slides" | "notebook" | "activity" | "guide"
    lecture: int | None
    track: str  # "lecture" | "add-on lab"
    title: str | None = None  # None -> derived by the loader (deck title / first H1)


# Lecture each practical notebook belongs to, by topic (not by its file-name number).
NOTEBOOK_LECTURE: dict[str, int] = {
    "Module_2_Practical_1_Probability-1.ipynb": 2,
    "Module_2_Practical_2_Word_Sampling-1.ipynb": 2,
    "Module_2_Practical_3_Word_Embeddings-2.ipynb": 2,
    "M3_Prompting_Strategies_colab.ipynb": 3,
    "Module_3_Practical_1_FCNN.ipynb": 4,  # Lecture 4: FCNN review, tensors, training
    "Module_3_Practical_2_Convolutions.ipynb": 4,  # Lecture 4: convolutions, pooling, CNNs
    "Module_3_Practical_3_CNN.ipynb": 4,
    "Module_4_Practical_VAE.ipynb": 5,  # Lecture 5: autoencoders, VAE
    "Module_6_Practical_1_GAN_(1).ipynb": 6,  # Lecture 6: GANs
    "Module_6_Practical_2_RNN_(1).ipynb": 6,  # Lecture 6: autoregressive models, RNNs
    "Module_7_Practical_1_EnergyBasedMethods.ipynb": 8,  # Lecture 8: energy-based models
    "Module_7_Practical_2_DiffusionMethods.ipynb": 8,  # Lecture 8: diffusion
    "Module_8_Practical_GPT.ipynb": 9,  # Lecture 9: transformers, LLMs
    "Module_9_Practical_ReinforcementLearning.ipynb": 10,  # Lecture 10: RL basics
    "Module_10_ReinforcementLearningAndLLM.ipynb": 11,  # Lecture 11: RL applied to LLMs
}

# PDFs whose lecture differs from (or is missing in) the file name.
PDF_OVERRIDES: dict[str, FileInfo] = {
    "module10_rl_slides.pdf": FileInfo(
        "slides", 10, "lecture", "Module 10 – Basics of Reinforcement Learning (extended deck)"
    ),
    "Module 1 Custom Setup Guide-3.pdf": FileInfo(
        "guide", 1, "lecture", "Module 1 – Custom Setup Guide"
    ),
    # Activities are numbered by week; the LSTM activity builds on the Lecture 6 RNN material.
    "Module 7 Activity - Adding LSTM to Text Generation API.pdf": FileInfo(
        "activity", 6, "lecture", "Module 7 Activity – Adding LSTM to Text Generation API"
    ),
}

_MODULE_NUM = re.compile(r"module[\s_]*(\d+)", re.IGNORECASE)


def clean_stem(name: str) -> str:
    """'Module 4 - Modern Deep Learning Architectures Part 1.pptx' -> 'Module 4 – Modern ... Part 1'.

    Also strips download-duplicate suffixes like '-1', ' (1) (2)', '(3)'.
    """
    stem = Path(name).stem
    stem = re.sub(r"(\s*\(\d+\))+$", "", stem)  # ' (1) (2)'
    stem = re.sub(r"_\(\d+\)$", "", stem)  # '_(1)'
    stem = re.sub(r"-\d+$", "", stem)  # '-1', '-3'
    stem = stem.replace(" - ", " – ").replace("_", " ")
    return stem.strip()


def classify(path: Path) -> FileInfo:
    """Return what kind of course material `path` is."""
    name = path.name
    suffix = path.suffix.lower()
    m = _MODULE_NUM.search(name)
    file_module = int(m.group(1)) if m else None

    if suffix == ".pptx":
        return FileInfo("slides", file_module, "lecture", clean_stem(name))

    if suffix == ".pdf":
        if name in PDF_OVERRIDES:
            return PDF_OVERRIDES[name]
        kind = "activity" if "activity" in name.lower() else "guide"
        return FileInfo(kind, file_module, "lecture", clean_stem(name))

    if suffix == ".ipynb":
        if name in NOTEBOOK_LECTURE:
            return FileInfo("notebook", NOTEBOOK_LECTURE[name], "lecture", None)
        # Everything else is from the agents / LLM add-on labs (RAG, LangGraph, evals, fine-tuning...).
        return FileInfo("notebook", None, "add-on lab", None)

    raise ValueError(f"Unsupported file type: {name}")


def discover_files(course_dir: Path) -> list[Path]:
    """All supported course files under `course_dir`, sorted for deterministic output."""
    files = []
    for p in sorted(course_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in p.relative_to(course_dir).parts):
            continue
        if p.name.startswith((".", "~$")):  # hidden files, Office lock files
            continue
        files.append(p)
    return files
