"""Export the review doc (Claude Docs) into the Quarto book at docs/book/: one checkpoint = one run + commit + tag.

The doc is the development surface; this book is the durable record. Input: the tabs' XML as the Docs
`read` returns it (saved to files; the API is reached through the assistant, not from here). The Analysis
tab is split into one chapter per level-2 heading; Glossary and Events become appendices. Figures: doc
blobs map to the PNGs under docs/figures/ and results/cv/ (BLOBS below); embedded diagrams map to
screenshots saved under docs/book/figures/.

Run: uv run python scripts/fusion/doc_to_book.py --analysis analysis.xml --events events.xml [--glossary glossary.xml] --date YYYY-MM-DD
 or: uv run python scripts/fusion/doc_to_book.py --analysis-md analysis.md --events events.xml --date YYYY-MM-DD
Then: cd docs/book && quarto render

--analysis-md takes the Docs `export` of the Analysis tab (format markdown, decoded). That export drops every
figure (blob images and embedded diagrams), so figures are re-inserted where the previous book had them: for
each figure in the existing chapters, the non-empty line before it is the anchor; the figure goes after the
new block that starts with the same text. An anchor with no match raises (the paragraph was reworded: edit the
anchor by hand or export the XML instead). Chapters that contain executable Quarto cells are never overwritten
(the book stops being a pure export once numbers are read from results/; see docs/book/README.md).
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


def md_blocks(text: str) -> list[str]:
    """One block per blank-line-separated chunk; tables and lists stay whole. The markdown export leaves an
    `&#91;image: caption\\]` placeholder where each figure was; those are dropped (figures are re-inserted)."""
    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", text) if b.strip()]
    return [b for b in blocks if not b.startswith("&#91;image:")]


def figure_anchors() -> list[tuple[str, str]]:
    """(anchor text, figure markdown) for every figure in the existing chapters, in reading order."""
    out = []
    for qmd in sorted(BOOK.glob("[0-9][0-9]-*.qmd")):
        lines = qmd.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith("!["):
                prev = next((l for l in reversed(lines[:i]) if l.strip()), None)
                if prev is None:
                    raise ValueError(f"{qmd.name}: figure with nothing before it: {line[:60]}")
                out.append((prev.strip(), line))
    return out


def reinsert_figures(blocks: list[str], anchors: list[tuple[str, str]]) -> list[str]:
    """Put each figure after the block whose first line matches its anchor's first 60 characters."""
    missing = []
    for anchor, fig in anchors:
        key = anchor[:60]
        hits = [i for i, b in enumerate(blocks) if not b.startswith("![") and any(l.strip()[:60] == key for l in b.splitlines())]
        if len(hits) != 1:
            missing.append((len(hits), anchor[:80]))
            continue
        blocks.insert(hits[0] + 1, fig)
    if missing:
        raise ValueError("figure anchors not matched exactly once (hits, anchor):\n  " + "\n  ".join(f"{n}: {t}" for n, t in missing))
    return blocks


def converted_chapters() -> dict[str, str]:
    """title -> text of every chapter that holds a {python} cell (numbers read from results/ at render time).
    These are kept as they are and only renumbered; the doc no longer owns their numbers (docs/book/README.md)."""
    out = {}
    for q in sorted(BOOK.glob("[0-9][0-9]-*.qmd")):
        text = q.read_text()
        if "{python}" in text:
            out[text.splitlines()[0].removeprefix("# ").strip()] = text
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=None, help="Analysis tab XML (Docs read)")
    ap.add_argument("--analysis-md", default=None, help="Analysis tab markdown (Docs export, decoded); figures re-inserted by anchor")
    ap.add_argument("--glossary", default=None, help="Glossary tab XML; without it the existing glossary.qmd is kept")
    ap.add_argument("--events", required=True)
    ap.add_argument("--append-md", default=None, help="markdown continuing the Analysis tab, when its read was capped and the rest came back inline")
    ap.add_argument("--date", required=True, help="checkpoint date, e.g. 2026-10-21")
    a = ap.parse_args()
    if (a.analysis is None) == (a.analysis_md is None):
        raise SystemExit("give exactly one of --analysis (XML) or --analysis-md (markdown export)")
    converted = converted_chapters()
    (BOOK / "figures").mkdir(parents=True, exist_ok=True)
    if a.analysis_md:
        anchors = figure_anchors()
        for src in BLOBS.values():  # refresh the PNGs the figures point at
            shutil.copy(ROOT / src, BOOK / "figures" / Path(src).name)
        blocks = reinsert_figures(md_blocks(Path(a.analysis_md).read_text()), anchors)
        print(f"{len(anchors)} figures re-inserted")
    for old in BOOK.glob("[0-9][0-9]-*.qmd"):
        old.unlink()
    if a.analysis:
        blocks = doc_to_blocks(Path(a.analysis).read_text())
    if a.append_md:
        blocks += md_blocks(Path(a.append_md).read_text())
    chapters = split_chapters(blocks)
    front = chapters[0][1]
    h1 = next((b[2:] for b in front if b.startswith("# ")), "Flood fusion")
    intro = [b for b in front if not b.startswith("# ")]
    files = []
    kept = []
    for i, (title, body) in enumerate(chapters[1:], 1):
        name = f"{i:02d}-{slug(title)}.qmd"
        if title in converted:
            (BOOK / name).write_text(converted.pop(title))
            kept.append(name)
        else:
            (BOOK / name).write_text(f"# {title}\n\n" + "\n\n".join(body) + "\n")
        files.append(name)
    if converted:
        raise SystemExit(f"converted chapters whose title no longer exists in the doc (renamed or removed?): {list(converted)}. "
                         "Rename the chapter's first line to the doc's new title, or delete it, then run again.")
    if kept:
        print("kept as converted (numbers from results/):", *kept, sep="\n  ")
    for src, name in ((a.glossary, "glossary.qmd"), (a.events, "events.qmd")):
        if src is None:
            continue
        bl = doc_to_blocks(Path(src).read_text())
        (BOOK / name).write_text("\n\n".join(b for b in bl) + "\n")
    (BOOK / "log.qmd").write_text("{{< include ../log.md >}}\n")  # the live ledger, never a copy
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
        + "  appendices:\n    - glossary.qmd\n    - events.qmd\n    - log.qmd\n\nformat:\n  html:\n    theme: cosmo\n"
        "    include-before-body:\n      - home-link.html  # back-link to the landing page (team convention)\n    toc: true\n    number-sections: true\n")
    print("chapters:", *files, sep="\n  ")


if __name__ == "__main__":
    main()
