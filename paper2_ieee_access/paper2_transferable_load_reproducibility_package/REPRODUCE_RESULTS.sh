#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Core public-data rerun using bundled processed data and package-local output paths.
"${PYTHON_BIN}" scripts/run_uci_ssl_representation_prototype.py
"${PYTHON_BIN}" scripts/run_uci_ssl_cold_start_diagnostics.py
"${PYTHON_BIN}" scripts/run_uci_client_statistical_tests.py
"${PYTHON_BIN}" scripts/run_uci_random_conv_representation.py
"${PYTHON_BIN}" scripts/run_uci_trainable_tdconv_baseline.py
"${PYTHON_BIN}" scripts/run_uci_tdconv_multiseed_stability.py
"${PYTHON_BIN}" scripts/run_uci_patch_attention_transfer_baseline.py
"${PYTHON_BIN}" scripts/run_uci_source_mlp_transfer_baseline.py
"${PYTHON_BIN}" scripts/run_uci_neural_tdconv_residual_check.py
"${PYTHON_BIN}" scripts/run_uci_appliances_energy_baselines.py
"${PYTHON_BIN}" scripts/run_uci_appliances_multihorizon_robustness.py
