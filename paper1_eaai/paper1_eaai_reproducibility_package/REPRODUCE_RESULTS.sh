#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Core public-data rerun using bundled processed data and package-local output paths.
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
