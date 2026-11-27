# The flood fusion book (`docs/book/`)

The durable record of the analysis: rendered to https://ocha-dap.github.io/ds-flood-gfm/fusion/ by
`scripts/publish_pages.sh`, tagged `checkpoint-YYYY-MM-DD` in git.

## Two kinds of chapter

- **Exported prose.** Most chapters are the Analysis tab of the review doc
  (https://claude.ai/artifact/EzpeTJ8mnRVPaw6rAE7Xop) as `scripts/fusion/doc_to_book.py` exported it at the
  last checkpoint; the raw export is kept under `_source/<date>/`. The doc is where the analysis is developed.
- **Converted chapters.** A chapter that holds a `{python}` cell reads the converted passages from
  `results/` at render time through `_numbers.py` (`make book`): a missing file, column or row fails the
  render. The rest of such a chapter is still exported prose until it is converted too. `doc_to_book.py`
  keeps a converted chapter as it is (matched by title; renumbered if a section was inserted before it) and
  re-exports the others, so prose changes made in the doc to a converted chapter have to be carried over by
  hand, and the book, not the doc, is the source of its numbers.

Converted so far (keep this list current):

| chapter | converted passages | still hand-typed |
|---|---|---|
| `15-results.qmd` | "In short" headline numbers (overall, Asia, open, vegetated, microwave vote), event and cell-day counts, the all-cells table, the without-the-largest-basin table | every other table and number in the chapter |

## Rendering

`make book` (sets `QUARTO_PYTHON` to the project venv, which has the kernel and pandas). Plain
`quarto render` picks whatever Python Quarto finds first and may fail on the imports.

## Checkpoint

1. Export the Analysis tab (Docs `export`, markdown), decode it, run
   `uv run python scripts/fusion/doc_to_book.py --analysis-md analysis.md --events events.xml --date YYYY-MM-DD`.
   Figures are re-inserted after the paragraph that preceded them last time; a reworded anchor raises.
2. `make book`; read the render.
3. Commit, tag `checkpoint-YYYY-MM-DD`, `scripts/publish_pages.sh`.
