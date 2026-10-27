# Flood fusion v1: one entry point per phase. See README.md "Flood fusion v1" and CLAUDE.md.
# Every step is resumable: re-running a target skips work whose outputs exist.
# Long steps are network-bound; on a laptop run them with `nice` (the default here).

PY := nice -n 19 uv run python -u
P1 := scripts/fusion/phase1_feasibility.py
P2 := scripts/fusion/phase2_cube.py

.PHONY: help setup test phase0 phase1 phase2 phase3 phase4 phase5 phase6 daily folds crossval

help:
	@grep -E '^[a-z0-9]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup: ## install the environment (uv) incl. test extras
	uv sync --extra dev

test: ## unit tests (synthetic data; no network)
	uv run --extra dev pytest -q tests/

phase0: ## inventory inputs: snow climatology (CDS), static-layer tile indexes
	$(PY) scripts/fusion/fetch_snow_climatology.py
	$(PY) -c "from ds_flood_gfm.fusion.config import load_config; from ds_flood_gfm.fusion.static import tile_index; c=load_config(); [tile_index(c, l) for l in ('hand','gsw','worldcover')]"

phase1: ## gold-only feasibility: select events, gold cube, sources, metrics, gate
	$(PY) $(P1) select
	$(PY) $(P1) gold --workers 2
	$(PY) $(P1) tally
	$(PY) $(P1) gfds
	$(PY) $(P1) sources --workers 2 --continue-on-error
	$(PY) $(P1) metrics

phase2: ## training cube: stratified random sample + static rasters (prior v1 is written by phase3)
	$(PY) $(P2) design --workers 2
	$(PY) $(P2) gfds
	$(PY) $(P2) sources --workers 2
	$(PY) $(P2) static

phase3: ## accuracy tables: gold (+ EM for thin strata) and prior v1; then the agreement table and its comparison (ADR 0002)
	$(PY) scripts/fusion/phase3_accuracy.py
	$(PY) scripts/fusion/phase3_agreement.py

phase4: ## engine tests (must pass), then a one-day live run on a small AOI as a check
	$(PY) scripts/fusion/phase4_engine.py test

phase5: ## held-out validation, daily outputs for every held-out window, ablation
	$(PY) scripts/fusion/phase5_validate.py

folds: ## five folds by river basin over every in-scope event (ADR 0004); refuses until all are built
	$(PY) scripts/fusion/make_folds.py

crossval: ## cross-validated re-run: fit on four folds, score the fifth (ADRs 0003-0007) -> results/cv/
	$(PY) scripts/fusion/phase5_crossval.py

phase6: ## mirror outputs to blob (dev), write the dataset manifest
	$(PY) scripts/fusion/phase6_package.py

daily: ## operational daily run for the configured AOIs (DATE=YYYY-MM-DD, default yesterday UTC)
	$(PY) scripts/fusion/daily_job.py $(if $(DATE),--date $(DATE),)
