#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" scripts/public_data_download_templates.py --download opsd_time_series_60min_singleindex
"${PYTHON_BIN}" scripts/public_data_download_templates.py --download uci_electricity_load_diagrams
"${PYTHON_BIN}" scripts/run_public_opsd_baselines.py
"${PYTHON_BIN}" scripts/run_uci_load_transfer_baselines.py
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
"${PYTHON_BIN}" scripts/build_paper2_reference_assets.py
"${PYTHON_BIN}" scripts/integrate_uci_load_transfer_results.py
"${PYTHON_BIN}" scripts/integrate_uci_client_stat_tests.py
"${PYTHON_BIN}" scripts/integrate_uci_random_conv_results.py
"${PYTHON_BIN}" scripts/integrate_uci_trainable_tdconv_results.py
"${PYTHON_BIN}" scripts/integrate_uci_neural_tdconv_residual_results.py
"${PYTHON_BIN}" scripts/integrate_uci_appliances_results.py
"${PYTHON_BIN}" scripts/build_paper2_ieee_access_submission_pack.py
