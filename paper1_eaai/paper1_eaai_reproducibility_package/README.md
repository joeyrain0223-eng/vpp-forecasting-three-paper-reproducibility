# Paper 1 EAAI Reproducibility Package

Generated: 2026-07-06 10:09:30 UTC

Purpose: provide a compact, public-data-only supplement for the Paper 1 EAAI submission candidate, "Sequence-Anchored GraphPatch Residual Learning for Uncertainty-Aware Electricity Price Forecasting in Virtual Power Plant Market Operation".

This package deliberately excludes local/private Hunan and Shandong data. The manuscript's core empirical claims are reproducible on the public Open Power System Data hourly benchmark.

## Contents

- `scripts/`: Python scripts used to download/prepare public data and reproduce the Paper 1 OPSD experiments.
- `data/processed/`: processed OPSD tidy dataset used by the current results. The large raw OPSD CSV is not bundled; use the public download command below if full raw provenance is needed.
- `results/`: Paper 1 public result CSV files.
- `figures/`: Paper 1 rendered figures used in the manuscript.
- `references/`: verified BibTeX, reference block, and reference audit/register files.
- `manuscript/`: current anonymous manuscript, title/declarations file, highlights, and checklist.
- `repository_release/`: data/code availability text, release notes, `CITATION.cff`, and `.zenodo.json` templates for DOI-backed public deposit.
- `requirements-paper1.txt`: minimal Python package requirements for the public-data experiments.
- `REPRODUCE_RESULTS.sh`: offline core-result rerun entry point using the bundled processed public data.
- `RUN_COMMANDS.sh`: extended historical/project command list including raw download and manuscript support steps.
- `MANIFEST.csv`: SHA-256 checksums for bundled files, excluding the manifest file itself.

Bundled file count excluding `MANIFEST.csv`: 98

Core artifact payload size before generated package docs: 63.45 MiB

## Public Data Source

Primary public source:

`https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv`

The bundled processed tidy file is:

`data/processed/opsd_hourly_price_load_renewables_tidy.csv`

To redownload the raw OPSD data in the original workspace layout:

```bash
python scripts/public_data_download_templates.py --download opsd_time_series_60min_singleindex
```

## Reproduction Order

Run from the root of this supplement package after extracting it. The bundled scripts are sanitized to use the package root as their working artifact root; recheck paths after any manual repository rearrangement.

For a reviewer-side rerun without downloading raw source files or rebuilding manuscript support artifacts, use:

```bash
bash REPRODUCE_RESULTS.sh
```

The extended project command sequence is:

```bash
python scripts/public_data_download_templates.py --download opsd_time_series_60min_singleindex
python scripts/run_public_opsd_baselines.py
python scripts/run_opsd_probabilistic_price_model.py
python scripts/run_opsd_graph_temporal_price_ablation.py
python scripts/run_opsd_nonlinear_price_baselines.py
python scripts/run_opsd_deep_graph_patch_price_model.py
python scripts/run_opsd_graphpatch_robustness.py
python scripts/run_opsd_modern_sequence_price_baselines.py
python scripts/run_opsd_sequence_anchor_graphpatch_price_model.py
python scripts/run_opsd_tdconv_sequence_anchor_graphpatch_price_model.py
python scripts/run_opsd_patch_attention_price_baseline.py
python scripts/run_opsd_price_sensitivity_checks.py
python scripts/build_paper1_eaai_reference_assets.py
python scripts/build_eaai_paper1_submission_pack.py
```

## Software Notes

The current scripts use Python standard-library modules plus `numpy`, `pandas`, `Pillow`, and `python-docx` for manuscript/package generation. The manuscript DOCX render QA was performed separately with the local document-rendering workflow.

## Key Result Anchors

- Sequence-anchored GraphPatch improves spike-regime RMSE in 4/4 OPSD zones versus the selected DLinear/NLinear anchor, with 3.69% mean improvement.
- TDConv-inclusive selected-anchor GraphPatch improves all-hour RMSE in 4/4 OPSD zones and spike-regime RMSE in 3/4 zones, with Great Britain exposed as the stricter strong-anchor limitation case.
- Patch-attention reviewer baseline is included as CPU-only patch-family evidence: it improves spike RMSE numerically in DE-LU, DK1 and DK2 versus TDConv, but only DK2 has positive paired evidence; Great Britain remains a negative-control case.
- Spike-threshold sensitivity remains positive in 4/4 OPSD zones at the 85th, 90th, and 95th percentile training-set volatility thresholds, with mean RMSE gains between 3.68% and 3.74%.
- Conformal calibration-window sensitivity shows that 15-20% pre-test calibration windows stay closest to the 90% coverage target on average.
- Rolling-origin GraphPatch robustness improves spike-regime RMSE in 12/12 zone-window cases, with 11.25% mean improvement.
- Leave-one-zone-out transfer improves spike-regime RMSE in 3/4 held-out zones; DE-LU remains the limitation case.
- The verified reference register has 26 DOI-confirmed entries, 9 stable URL/dataset entries, and zero review-required entries.

## Publication Boundary

Do not add local customer identifiers, settlement sheets, contract files, or non-public trading details to this supplement. Local data should remain a separate application case unless written publication permission is obtained.
