#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" scripts/public_data_download_templates.py --download opsd_time_series_60min_singleindex
"${PYTHON_BIN}" scripts/run_public_opsd_baselines.py
"${PYTHON_BIN}" scripts/run_opsd_probabilistic_price_model.py
"${PYTHON_BIN}" scripts/run_opsd_graph_temporal_price_ablation.py
"${PYTHON_BIN}" scripts/run_opsd_nonlinear_price_baselines.py
"${PYTHON_BIN}" scripts/run_opsd_deep_graph_patch_price_model.py
"${PYTHON_BIN}" scripts/run_opsd_graphpatch_robustness.py
"${PYTHON_BIN}" scripts/run_opsd_modern_sequence_price_baselines.py
"${PYTHON_BIN}" scripts/run_opsd_sequence_anchor_graphpatch_price_model.py
"${PYTHON_BIN}" scripts/run_opsd_tdconv_sequence_anchor_graphpatch_price_model.py
"${PYTHON_BIN}" scripts/run_opsd_patch_attention_price_baseline.py
"${PYTHON_BIN}" scripts/run_opsd_price_sensitivity_checks.py
"${PYTHON_BIN}" scripts/build_paper1_eaai_reference_assets.py
"${PYTHON_BIN}" scripts/build_eaai_paper1_submission_pack.py
