"""Guards that keep course materials out of the (public) repository."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from study_assistant.config import PROJECT_ROOT

NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_have_no_outputs(path: Path):
    """Notebook outputs print course text, so notebooks are committed without them.

    Fix with:  uv run --with nbconvert jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    with_output = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and c.get("outputs")]
    assert not with_output, f"{path.name}: clear outputs of cells {with_output} before committing"


def test_course_files_are_gitignored():
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/", "*.pptx", "*.pdf", ".env"):
        assert pattern in ignore.splitlines(), f"{pattern} missing from .gitignore"
