# Paper 3 Decision-Focused VPP Reproducibility Package

Generated: 2026-07-06 11:18:08 UTC

Purpose: provide a compact, public-data-only supplement for the Paper 3 manuscript, "Decision-Focused Learning for Virtual Power Plant Bidding under Electricity Price and Load Uncertainty".

The package supports the public OPSD decision benchmark, extended VPP risk simulator, held-out decision-focused policy search, forecast-coupled VPP experiment, risk-aversion sensitivity, reviewer-facing robustness checks, fuzzy risk-aware soft-computing comparator, genetic soft-computing policy search, multi-seed genetic-search stability diagnostics, particle-swarm policy-search checking, constrained tabular Q-learning policy-learning boundary check, and a bounded GIS infrastructure-context figure. It deliberately excludes raw local/private Hunan and Shandong data and raw OSM/GEM/GIS source files.

## Contents

- `scripts/`: Python scripts used to reproduce the public OPSD decision experiments and reference/appendix assets.
- `data/processed/`: processed OPSD tidy dataset used by the current public results.
- `results/`: daily-level, summary, aggregate, and coefficient CSV files for Paper 3 public experiments.
- `figures/`: public-data figures used in the Paper 3 manuscript, plus a GIS context figure generated from summary-level infrastructure metadata.
- `context/`: derived GIS summary tables and audit files. Raw OSM/GEM/GIS source files are excluded pending attribution and redistribution gates.
- `references/`: verified BibTeX, reference block, and reference audit/register files.
- `manuscript/`: current ASOC submission-candidate manuscript and standalone simulator appendix.
- `requirements-paper3.txt`: minimal Python package requirements for the public-data experiments.
- `REPRODUCE_RESULTS.sh`: offline core-result rerun entry point using the bundled processed public data.
- `RUN_COMMANDS.sh`: extended historical/project command list including reference and appendix integration steps.
- `MANIFEST.csv`: SHA-256 checksums for bundled files, excluding the manifest file itself.

Bundled file count excluding `MANIFEST.csv`: 117

Core artifact payload size before generated package docs: 110.03 MiB

## Public Data Source

Primary public source:

`https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv`

The bundled processed tidy file is:

`data/processed/opsd_hourly_price_load_renewables_tidy.csv`

The large raw OPSD CSV is not bundled. The processed file is sufficient to reproduce the public Paper 3 decision experiments from the extracted supplement package.

## Reproduction Order

Run from the root of this supplement package after extracting it. The bundled scripts are sanitized to use the package root as their working artifact root; recheck paths after any manual repository rearrangement.

For a reviewer-side rerun without rebuilding reference or appendix support artifacts, use:

```bash
bash REPRODUCE_RESULTS.sh
```

The extended project command sequence is:

```bash
python scripts/run_public_opsd_baselines.py
python scripts/run_opsd_vpp_risk_simulator.py
python scripts/run_opsd_decision_focused_policy_search.py
python scripts/run_opsd_forecast_coupled_vpp.py
python scripts/run_opsd_risk_aversion_sensitivity.py
python scripts/run_opsd_vpp_reviewer_robustness.py
python scripts/run_opsd_fuzzy_risk_vpp_baseline.py
python scripts/run_opsd_constrained_q_learning_vpp_baseline.py
python scripts/run_opsd_genetic_policy_search_vpp_baseline.py
python scripts/run_opsd_genetic_policy_multiseed_stability.py
python scripts/run_opsd_pso_policy_search_vpp_baseline.py
python scripts/run_opsd_surrogate_policy_gradient_vpp_baseline.py
python scripts/build_paper3_reference_assets.py
python scripts/integrate_paper3_verified_references.py
python scripts/integrate_paper3_notation_appendix.py
```

## Key Result Anchors

- Extended VPP simulator: rolling-28d mean FTO reaches 20.88 EUR/day proxy mean revenue; robust quantile FTO reaches 20.10; prev-day FTO reaches 17.39.
- Held-out decision-focused policy search: revenue-selected policy reaches 21.64 EUR/day proxy mean revenue on the final 20% public test split.
- Forecast-coupled VPP experiment: forecast-coupled DF revenue policy reaches 31.43 EUR/day proxy mean revenue and 4.67 regret on the held-out coupled split.
- Risk-aversion sensitivity: lambda_r=0.25/0.50 improves held-out CVaR10 from -1.85 to -1.28 relative to revenue-only selection.
- Robustness check: under high imbalance penalty plus transaction cost, the revenue-selected decision-focused policy keeps the highest average revenue, while robust quantile FTO has fewer loss days and less negative CVaR10.
- Fuzzy risk-aware soft-computing comparator: the interpretable fuzzy rule reaches 11.34 EUR/day proxy revenue and -4.97 CVaR10, making it useful as a transparent but not dominant comparator.
- Genetic soft-computing policy search: the risk-adjusted genetic search reaches 21.65 EUR/day proxy revenue and -1.39 CVaR10, giving a small continuous-search improvement over the coarse revenue-selected policy grid while preserving the chronological held-out boundary.
- Multi-seed genetic stability: eight risk-adjusted genetic seeds reach 21.543 +/- 0.078 EUR/day proxy revenue and -1.530 +/- 0.163 CVaR10, showing stable risk-oriented refinement but not a universal revenue advantage over the coarse grid.
- Particle-swarm policy search: the risk-adjusted PSO check reaches 21.60 EUR/day proxy revenue and -1.74 CVaR10, staying close to the decision-focused and genetic policies while confirming that the swarm-search result is competitive but not a new best policy over the genetic risk-adjusted variant.
- Surrogate policy-gradient baseline: the risk-adjusted surrogate reaches 21.26 EUR/day proxy revenue and -2.56 CVaR10, below GA/PSO but above fuzzy and Q-learning; this closes a differentiable-policy reviewer check without claiming neural-RL superiority.
- Constrained tabular Q-learning boundary check: the Q-learning policy reaches -5.65 EUR/day proxy revenue and -24.53 CVaR10, losing to decision-focused policy search on 1280 of 1378 paired test days; this negative result documents the policy-learning boundary rather than supporting a deep-RL claim.
- GIS infrastructure context: the summary-level audit records 12,839 2025 transmission-line records, 2,041 substation records, 9,444 grid-link records, 4.87 million OSM mainland power records, 4,274 WRI China power-plant records, and 3,108.9 GW of GEM operating capacity represented in the local China table. This supports external motivation only; it is not a model-training dataset for the OPSD decision benchmark.

## Publication Boundary

Do not add raw local customer identifiers, contract files, account numbers, settlement sheets, non-public trading details, or raw GIS/OSM/GEM extracts to this supplement. Local and GIS source data should remain separate application-context material unless written publication permission, source attribution, and redistribution handling are confirmed.
