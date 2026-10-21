#!/usr/bin/env bash
# Publish this repo's GitHub Pages site (gh-pages branch, legacy branch mode) following the team's
# landing-page convention (KB methods/static-data-apps.md): a landing page at /, each product under
# its own path. Never `quarto publish gh-pages`: it runs `git rm -r .` on the branch.
#
# Layout on gh-pages:
#   /                  landing page (pages/index.html, pages/assets/) + redirect stubs for the old
#                      root pages (pages/_redirects/*.html -> /gfm-exploration/*.html)
#   /gfm-exploration/  the 2024 GFM exploration book (rendered html; moved from / on first run)
#   /fusion/           the flood fusion book, docs/book/_book/
#
# Usage: scripts/publish_pages.sh [--no-render] [--dry-run]
set -euo pipefail
render=1; dry_run=0
for arg in "$@"; do case "$arg" in --no-render) render=0;; --dry-run) dry_run=1;; *) echo "usage: $0 [--no-render] [--dry-run]" >&2; exit 2;; esac; done
root="$(git rev-parse --show-toplevel)"; cd "$root"
rev="$(git describe --always --dirty --tags)"
[ "$render" = 1 ] && quarto render docs/book
test -f docs/book/_book/index.html || { echo "error: docs/book/_book/index.html missing" >&2; exit 1; }

git fetch origin gh-pages
wt="$(mktemp -d)/gh-pages"
git worktree add --detach "$wt" origin/gh-pages
trap 'cd "$root" && git worktree remove --force "$wt"' EXIT
cd "$wt"

# one-time move of the old root book into gfm-exploration/ (idempotent: skipped once moved)
if [ ! -d gfm-exploration ] && [ -f 01_querying_gfm.html ]; then
  mkdir gfm-exploration
  for f in *.html *_files site_libs search.json; do [ -e "$f" ] && git mv -k "$f" gfm-exploration/ 2>/dev/null || true; done
  echo "moved the old root book to gfm-exploration/"
fi
rsync -a --exclude .DS_Store "$root/pages/index.html" "$root/pages/assets" ./
rsync -a --exclude .DS_Store "$root/pages/_redirects/" ./
rsync -a --delete --exclude .DS_Store "$root/docs/book/_book/" fusion/
touch .nojekyll
git add -A -f -- index.html assets fusion gfm-exploration .nojekyll
git add -A -f -- $(ls "$root/pages/_redirects") 2>/dev/null || true
outside="$(git status --porcelain --untracked-files=all | grep -v -E '^.. (index\.html|assets/|fusion/|gfm-exploration/|\.nojekyll|[0-9A-Z][0-9A-Za-z_]*\.html)' || true)"
if [ -n "$outside" ]; then echo "error: refusing to publish, unexpected changes:" >&2; echo "$outside" >&2; exit 1; fi
if git diff --cached --quiet; then echo "Nothing to publish."; exit 0; fi
git diff --cached --stat | tail -1
[ "$dry_run" = 1 ] && { echo "Dry run: not committing or pushing."; exit 0; }
git commit -q -m "Publish pages site from $rev"
git push origin HEAD:gh-pages
echo "Published: https://ocha-dap.github.io/ds-flood-gfm/"
