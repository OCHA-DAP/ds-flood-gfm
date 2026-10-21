"""Export the review doc (Claude Docs) into the Quarto book at docs/book/: one checkpoint = one run + commit + tag.

The doc is the development surface; this book is the durable record. Input: the tabs' XML as the Docs
`read` returns it (saved to files; the API is reached through the assistant, not from here). The Analysis
tab is split into one chapter per level-2 heading; Glossary and Events become appendices. Figures: doc
blobs map to the PNGs under docs/figures/ and results/cv/ (BLOBS below); embedded diagrams map to
screenshots saved under docs/book/figures/.

Run: uv run python scripts/fusion/doc_to_book.py --analysis analysis.xml --events events.xml [--glossary glossary.xml] --date YYYY-MM-DD
Then: cd docs/book && quarto render
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "docs" / "book"
BLOBS = {  # doc blob id -> source PNG (relative to repo root)
    "blob/c37bcc59-9d08": "docs/figures/event_timeline_example.png",
    "blob/8a880d9e-1913": "docs/figures/strata_example.png",
    "blob/c6aac6ae-abb0": "docs/figures/water_vs_flood_example.png",
    "blob/5e7b3273-bf77": "docs/figures/scoring_mechanics.png",
    "blob/548ef8c7-fa6a": "results/cv/f1_by_continent.png",
    "blob/80b7e6b6-3be5": "results/cv/f1_by_landtype.png",
    "blob/b9f60417-5ed7": "results/cv/reliability_open.png",
    "blob/eeef5d9a-887a": "docs/figures/water_vs_flood_cells.png",
    "blob/c79ad1d5-c167": "docs/figures/tradeoff_f1_false_events.png",
}
EMBEDS = {  # widget node -> screenshot already under docs/book/figures/
    "node/bc79694b-770a": "diagram_who_counts_water.png",
    "node/1238a4e0-935b": "diagram_water_first.png",
}


def inline(el) -> str:
    """Markdown for the inline content of a paragraph."""
    out = []
    for node in el:
        tag = node.tag
        if tag == "text":
            s = "".join(inline_marks(node))
            out.append(s)
        elif tag == "mention":
            out.append(node.get("label") or node.get("name") or "")
        elif tag == "date":
            out.append(node.get("value", ""))
        elif tag == "media":
            src = BLOBS.get(node.get("blob"))
            if src is None:
                raise KeyError(f"unmapped blob {node.get('blob')}")
            name = Path(src).name
            shutil.copy(ROOT / src, BOOK / "figures" / name)
            out.append(f"![{node.get('alt', '')}](figures/{name})")
        elif tag == "hardBreak":
            out.append("\n")
        else:
            out.append(inline(node))
    return "".join(out)


def inline_marks(text_el) -> list[str]:
    s = text_el.text or ""
    parts = []
    if len(text_el):
        for m in text_el:
            inner = (m.text or "") + "".join(inline_marks(c) for c in m)
            if m.tag == "bold":
                parts.append(f"**{inner}**")
            elif m.tag == "italic":
                parts.append(f"*{inner}*")
            elif m.tag == "code":
                parts.append(f"`{inner}`")
            elif m.tag == "link":
                parts.append(f"[{inner}]({m.get('href', '')})")
            else:
                parts.append(inner)
            parts.append(m.tail or "")
    return [s, *parts]


def block(el, depth=0) -> str:
    tag = el.tag
    if tag == "paragraph":
        h = el.get("heading")
        txt = inline(el).strip()
        return ("#" * int(h) + " " + txt) if h else txt
    if tag == "list":
        kind = el.get("kind", "bullet")
        items = []
        for i, li in enumerate(el.findall("listItem"), 1):
            prefix = f"{i}. " if kind == "ordered" else ("- [x] " if kind == "check" and li.get("checked") == "true" else "- [ ] " if kind == "check" else "- ")
            body = [block(c, depth + 1) for c in li]
            first, rest = body[0], body[1:]
            ind = " " * len(prefix)
            txt = prefix + first.replace("\n", "\n" + ind)
            for r in rest:
                txt += "\n" + ind + r.replace("\n", "\n" + ind)
            items.append(txt)
        return "\n".join(items)
    if tag == "table":
        rows = []
        for r in el.findall("row"):
            cells = [" ".join(block(c, depth + 1) for c in cell).replace("\n", " ").replace("|", "\\|").strip() for cell in r.findall("cell")]
            rows.append("| " + " | ".join(cells) + " |")
        if not rows:
            return ""
        n = rows[0].count("|") - 1
        return rows[0] + "\n|" + "---|" * n + "\n" + "\n".join(rows[1:])
    if tag == "embed":
        name = EMBEDS.get(el.get("ref"))
        if name is None:
            raise KeyError(f"unmapped embed {el.get('ref')}")
        return f"![{el.get('caption', '')}](figures/{name})"
    if tag == "media":
        return inline(ET.Element("p", {}) if False else _wrap(el))
    if tag in ("codeBlock", "code"):
        return "```\n" + (el.text or "") + "\n```"
    return "\n\n".join(block(c, depth) for c in el)


def _wrap(el):
    p = ET.Element("paragraph")
    p.append(el)
    return p


def doc_to_blocks(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    return [block(c) for c in root if c.tag != "gap"]


def split_chapters(blocks: list[str]) -> list[tuple[str, list[str]]]:
    """(title, blocks) per level-2 heading; the part before the first is the front matter."""
    chapters, cur, title = [], [], "front"
    for b in blocks:
        if b.startswith("## "):
            chapters.append((title, cur))
            title, cur = b[3:].strip(), []
        else:
            cur.append(b)
    chapters.append((title, cur))
    return chapters


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--glossary", default=None, help="Glossary tab XML; without it the existing glossary.qmd is kept")
    ap.add_argument("--events", required=True)
    ap.add_argument("--append-md", default=None, help="markdown continuing the Analysis tab, when its read was capped and the rest came back inline")
    ap.add_argument("--date", required=True, help="checkpoint date, e.g. 2026-10-21")
    a = ap.parse_args()
    (BOOK / "figures").mkdir(parents=True, exist_ok=True)
    for old in BOOK.glob("[0-9][0-9]-*.qmd"):
        old.unlink()
    blocks = doc_to_blocks(Path(a.analysis).read_text())
    if a.append_md:  # one block per blank-line-separated chunk; tables and lists stay whole
        blocks += [b for b in re.split(r"\n\s*\n", Path(a.append_md).read_text()) if b.strip()]
    chapters = split_chapters(blocks)
    front = chapters[0][1]
    h1 = next((b[2:] for b in front if b.startswith("# ")), "Flood fusion")
    intro = [b for b in front if not b.startswith("# ")]
    files = []
    for i, (title, body) in enumerate(chapters[1:], 1):
        name = f"{i:02d}-{slug(title)}.qmd"
        (BOOK / name).write_text(f"# {title}\n\n" + "\n\n".join(body) + "\n")
        files.append(name)
    for src, name in ((a.glossary, "glossary.qmd"), (a.events, "events.qmd")):
        if src is None:
            continue
        bl = doc_to_blocks(Path(src).read_text())
        (BOOK / name).write_text("\n\n".join(b for b in bl) + "\n")
    shutil.copy(ROOT / "docs" / "log.md", BOOK / "log.qmd")
    (BOOK / "index.qmd").write_text(
        f"# {h1} {{.unnumbered}}\n\n*Checkpoint {a.date}, exported from the review doc "
        "(https://claude.ai/artifact/EzpeTJ8mnRVPaw6rAE7Xop) by `scripts/fusion/doc_to_book.py`. The doc is where the analysis "
        "is developed; this book is the record. Every number comes from a file under `results/`; `log.md` lists the command "
        "behind each one.*\n\n" + "\n\n".join(intro) + "\n")
    (BOOK / "_quarto.yml").write_text(
        "project:\n  type: book\n  output-dir: _book\n\nbook:\n  title: \"Flood fusion\"\n"
        f"  subtitle: \"Analysis and results, checkpoint {a.date}\"\n  author: \"OCHA Centre for Humanitarian Data\"\n"
        "  date: today\n  sidebar:\n    collapse-level: 1\n  chapters:\n    - index.qmd\n"
        + "".join(f"    - {f}\n" for f in files)
        + "  appendices:\n    - glossary.qmd\n    - events.qmd\n    - log.qmd\n\nformat:\n  html:\n    theme: cosmo\n    toc: true\n    number-sections: true\n")
    print("chapters:", *files, sep="\n  ")


if __name__ == "__main__":
    main()
