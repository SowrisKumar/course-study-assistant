"""Draw the assistant's LangGraph workflow as a PNG (for the report; mermaid doesn't survive PDF export).

    uv run python docs/figures/make_workflow.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, LINE, BG = "#0b0b0b", "#52514e", "#8a897f", "#fcfcfb"
NODE, DECIDE, STOP, DONE = "#e8f0fb", "#fdf3e7", "#eeebfb", "#e6f6ef"  # light fills; text carries the meaning

# name: (x, y, label, fill)
NODES = {
    "q": (0.5, 9.0, "question", BG),
    "route": (0.5, 7.9, "route\nanswer · simplify · quiz · chitchat", DECIDE),
    "chit": (4.1, 7.9, "chitchat reply\n(no LLM call)", DONE),
    "retrieve": (0.5, 6.7, "retrieve (hybrid: BM25 + dense)\ntop 8 chunks → ≤ 5 passages", NODE),
    "check": (0.5, 5.5, "check\nsimilarity floor + LLM relevance", DECIDE),
    "generate": (0.5, 4.3, "generate\nanswer / explanation / quiz,\nciting passages as [n]", NODE),
    "cite": (0.5, 3.1, "cite\nvalidate [n], map to lecture + slide", NODE),
    "verify": (0.5, 1.9, "verify (optional, --verify)\nsupport check", DECIDE),
    "answer": (0.5, 0.8, "answer + sources", DONE),
    "abstain": (4.1, 3.1, "abstain\n\"This isn't covered in\nthe course materials.\"", STOP),
}
EDGES = [  # (from, to, label)
    ("q", "route", ""), ("route", "chit", "hi / thanks"), ("route", "retrieve", ""),
    ("retrieve", "check", ""), ("check", "generate", "relevant"), ("check", "abstain", "off-topic / none relevant"),
    ("generate", "cite", ""), ("cite", "abstain", "model says\nnot covered"), ("cite", "verify", ""),
    ("verify", "abstain", "not supported"), ("verify", "answer", "ok / off"),
]
W, H = 2.6, 0.78


def main(out: Path = Path(__file__).with_name("workflow.png")) -> None:
    fig, ax = plt.subplots(figsize=(8, 8.2))
    ax.set_xlim(-0.9, 5.5); ax.set_ylim(0.2, 9.5); ax.axis("off")
    for x, y, text, fill in NODES.values():
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fill, ec=LINE, lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=INK, linespacing=1.3)
    for a, b, label in EDGES:
        (xa, ya, *_), (xb, yb, *_) = NODES[a], NODES[b]
        if xa == xb:  # vertical: bottom of a -> top of b
            start, end = (xa, ya - H / 2), (xb, yb + H / 2)
        elif ya == yb:  # horizontal
            start, end = (xa + W / 2, ya), (xb - W / 2, yb)
        else:  # diagonal into the side of the abstain box
            start, end = (xa + W / 2, ya), (xb - W / 2, yb + (0.25 if ya > yb else -0.25))
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=LINE, lw=1.1,
                                     connectionstyle="arc3,rad=0" if xa == xb or ya == yb else "arc3,rad=-0.12"))
        if label:
            if xa == xb:  # vertical: to the right of the arrow
                x, y, ha = start[0] + 0.06, (start[1] + end[1]) / 2, "left"
            elif ya == yb:  # horizontal: above the arrow
                x, y, ha = (start[0] + end[0]) / 2, start[1] + (0.26 if "\n" in label else 0.16), "center"
            else:  # diagonal: beside the arrow, near its end
                t = 0.55
                x, y, ha = start[0] + t * (end[0] - start[0]) + 0.08, start[1] + t * (end[1] - start[1]), "left"
            ax.text(x, y, label, fontsize=7.5, color=MUTED, ha=ha, va="center", bbox=dict(fc=BG, ec="none", pad=1))
    ax.set_title("The study assistant's workflow (LangGraph StateGraph)", loc="left", fontsize=11, color=INK)
    fig.patch.set_facecolor(BG)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
