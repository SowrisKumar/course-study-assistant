"""Build submission/report.pdf from docs/report.md (figures embedded, repo links pointing to GitHub).

    uv run --with markdown==3.7 python scripts/make_report_pdf.py

Needs Google Chrome (used headless to print the HTML to PDF). Output goes to submission/, which is not
committed: rebuild it whenever the report changes.
"""

import base64
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/SowrisKumar/course-study-assistant/blob/main/"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """body{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.45;color:#111;max-width:46em;margin:auto}
h1{font-size:20pt}h2{font-size:14pt;margin-top:1.6em;border-bottom:1px solid #ccc}h3{font-size:12pt}
table{border-collapse:collapse;margin:.8em 0;font-size:9pt}th,td{border:1px solid #bbb;padding:3px 6px;vertical-align:top}
th{background:#f2f2f2}code{font-size:9pt;background:#f5f5f5;padding:0 2px}pre{background:#f5f5f5;padding:8px;font-size:8.5pt;white-space:pre-wrap}
img{max-width:100%}a{color:#1a5fb4}li{margin:.15em 0}@page{margin:16mm}"""


def for_python_markdown(src: str) -> str:
    """GitHub-flavoured details Python-Markdown needs spelled out: a blank line before a list,
    4-space indents for nested lists, and links for bare URLs."""
    out, in_code = [], False
    for line in src.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code:
            m = re.match(r"^( *)([-*] |\d+\. )", line)
            if m:
                line = " " * (len(m.group(1)) * 2) + line.lstrip()  # 2-space nesting -> 4-space
                prev = out[-1] if out else ""
                if prev.strip() and not re.match(r"^ *([-*] |\d+\. )", prev) and not prev.startswith(" "):
                    out.append("")
            line = re.sub(r"(?<![(<\[])(https?://[^\s)<]+)", r"<\1>", line)
        out.append(line)
    return "\n".join(out)


def repo_links(src: str) -> str:
    """Relative links to repo files -> GitHub links, so they work in the PDF."""
    def fix(m):
        text, link = m.group(1), m.group(2)
        if link.startswith(("http", "#", "figures/")):
            return m.group(0)
        path = (ROOT / "docs" / link).resolve().relative_to(ROOT)
        return f"[{text}]({GITHUB}{path.as_posix()})"
    return re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", fix, src)


def main() -> None:
    src = for_python_markdown(repo_links((ROOT / "docs" / "report.md").read_text()))
    html = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists"])
    html = re.sub(r'src="(figures/[^"]+\.png)"',
                  lambda m: f'src="data:image/png;base64,{base64.b64encode((ROOT / "docs" / m.group(1)).read_bytes()).decode()}"',
                  html)
    out = ROOT / "submission"
    out.mkdir(exist_ok=True)
    page = out / "report.html"
    page.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Course Study Assistant – Final Report"
                    f"</title><style>{CSS}</style></head><body>{html}</body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out / 'report.pdf'}", page.as_uri()], check=True, capture_output=True)
    page.unlink()
    print(f"wrote {out / 'report.pdf'}")


if __name__ == "__main__":
    main()
