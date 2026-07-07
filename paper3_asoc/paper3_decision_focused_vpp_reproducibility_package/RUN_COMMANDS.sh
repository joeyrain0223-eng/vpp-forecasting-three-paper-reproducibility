#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" scripts/public_data_download_templates.py --download opsd_time_series_60min_singleindex
"${PYTHON_BIN}" scripts/run_public_opsd_baselines.py
"${PYTHON_BIN}" scripts/run_opsd_vpp_risk_simulator.py
"${PYTHON_BIN}" scripts/run_opsd_decision_focused_policy_search.py
"${PYTHON_BIN}" scripts/run_opsd_forecast_coupled_vpp.py
"${PYTHON_BIN}" scripts/run_opsd_risk_aversion_sensitivity.py
"${PYTHON_BIN}" scripts/run_opsd_vpp_reviewer_robustness.py
"${PYTHON_BIN}" scripts/run_opsd_fuzzy_risk_vpp_baseline.py
"${PYTHON_BIN}" scripts/run_opsd_constrained_q_learning_vpp_baseline.py
"${PYTHON_BIN}" scripts/run_opsd_genetic_policy_search_vpp_baseline.py
"${PYTHON_BIN}" scripts/run_opsd_genetic_policy_multiseed_stability.py
"${PYTHON_BIN}" scripts/run_opsd_pso_policy_search_vpp_baseline.py
"${PYTHON_BIN}" scripts/run_opsd_surrogate_policy_gradient_vpp_baseline.py
"${PYTHON_BIN}" scripts/integrate_paper3_verified_references.py
"${PYTHON_BIN}" scripts/integrate_paper3_notation_appendix.py
