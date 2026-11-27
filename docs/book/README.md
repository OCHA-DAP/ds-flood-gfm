# The flood fusion book (`docs/book/`)

The durable record of the analysis: rendered to https://ocha-dap.github.io/ds-flood-gfm/fusion/ by
`scripts/publish_pages.sh`, tagged `checkpoint-YYYY-MM-DD` in git.

## Two kinds of chapter

- **Exported prose.** Most chapters are the Analysis tab of the review doc
  (https://claude.ai/artifact/EzpeTJ8mnRVPaw6rAE7Xop) as `scripts/fusion/doc_to_book.py` exported it at the
  last checkpoint; the raw export is kept under `_source/<date>/`. The doc is where the analysis is developed.
- **Data-driven chapters.** A chapter that holds a `{python}` cell reads its numbers and tables from
  `results/` at render time through `_numbers.py` (`make book`). Nothing in such a chapter is typed in: a
  missing file, column or row fails the render. `doc_to_book.py` refuses to overwrite these chapters, so
  once a chapter is converted the book, not the doc, is the source of its numbers, and prose changes made in
  the doc have to be carried over by hand.

Converted so far: `15-results.qmd` (the "In short" headline numbers, the all-cells table and the
without-the-largest-basin table).

## Rendering

`make book` (sets `QUARTO_PYTHON` to the project venv, which has the kernel and pandas). Plain
`quarto render` picks whatever Python Quarto finds first and may fail on the imports.

## Checkpoint

1. Export the Analysis tab (Docs `export`, markdown), decode it, run
   `uv run python scripts/fusion/doc_to_book.py --analysis-md analysis.md --events events.xml --date YYYY-MM-DD`.
   Figures are re-inserted after the paragraph that preceded them last time; a reworded anchor raises.
2. `make book`; read the render.
3. Commit, tag `checkpoint-YYYY-MM-DD`, `scripts/publish_pages.sh`.
