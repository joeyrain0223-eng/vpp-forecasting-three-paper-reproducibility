#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Core public-data rerun using bundled processed data and package-local output paths.
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
