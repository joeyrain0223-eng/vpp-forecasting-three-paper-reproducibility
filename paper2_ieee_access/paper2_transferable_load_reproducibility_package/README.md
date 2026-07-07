# Paper 2 Transferable Load Forecasting Reproducibility Package

Generated: 2026-07-06 11:03:14 UTC

Purpose: provide a public-data-only supplement for the Paper 2 manuscript, "Transferable Short-Term Load Forecasting for Aggregated Virtual Power Plant Resources via Source-Pooled Time-Series Representation Learning".

The package supports the UCI multi-client transfer baseline, masked-reconstruction representation prototype, label-scarce/cold-start diagnostics, client-level sign tests, deterministic random-convolution representation check, trainable dilated-convolution ridge encoder check, multi-seed TDConv source-subsampling stability check, CPU-only patch-attention transfer check, source-trained MLP transfer check, neural TDConv residual-head check, the OPSD public load baseline, and the UCI Appliances Energy Prediction one-hour-ahead plus multi-horizon external load sanity checks. It deliberately excludes local/private Hunan and Shandong operational files.

## Contents

- `scripts/`: Python scripts used to reproduce public Paper 2 experiments, integrate manuscript results, build reference assets, and generate the IEEE Access candidate pack.
- `data/processed/`: processed public UCI and OPSD datasets used by the current result files.
- `results/`: public result CSV files for OPSD and UCI baselines, SSL diagnostics, client-level tests, random-convolution representation checks, trainable dilated-convolution ridge checks, multi-seed TDConv stability checks, patch-attention checks, source-trained MLP checks, neural TDConv residual-head checks, and UCI Appliances one-hour/multi-horizon robustness checks.
- `figures/`: public-data figures used by the Paper 2 manuscript.
- `references/`: verified BibTeX, reference block, and reference audit/register files.
- `manuscript/`: current Paper 2 manuscript variants and IEEE Access data/code availability pack.
- `requirements-paper2.txt`: minimal Python package requirements for the public-data experiments.
- `REPRODUCE_RESULTS.sh`: offline core-result rerun entry point using the bundled processed public data.
- `RUN_COMMANDS.sh`: extended historical/project command list including integration and submission-pack steps.
- `MANIFEST.csv`: SHA-256 checksums for bundled files, excluding the manifest file itself.

Bundled file count excluding `MANIFEST.csv`: 111

Core artifact payload size before generated package docs: 55.33 MiB

## Public Data Sources

- UCI Electricity Load Diagrams 2011-2014: `https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014`
- Open Power System Data time-series package: `https://data.open-power-system-data.org/time_series/2020-10-06/`
- UCI Appliances Energy Prediction: `https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction`

Bundled processed files:

- `data/processed/uci_electricity_hourly_selected_clients.csv`
- `data/processed/uci_electricity_hourly_selected_clients_wide.csv`
- `data/processed/opsd_hourly_price_load_renewables_tidy.csv`
- `data/processed/uci_appliances_energy_supervised_1h.csv`

## Reproduction Order

Run from the root of this supplement package after extracting it. The bundled scripts are sanitized to use the package root as their working artifact root; recheck paths after any manual repository rearrangement.

```bash
python scripts/run_public_opsd_baselines.py
python scripts/run_uci_load_transfer_baselines.py
python scripts/run_uci_ssl_representation_prototype.py
python scripts/run_uci_ssl_cold_start_diagnostics.py
python scripts/run_uci_client_statistical_tests.py
python scripts/run_uci_random_conv_representation.py
python scripts/run_uci_trainable_tdconv_baseline.py
python scripts/run_uci_tdconv_multiseed_stability.py
python scripts/run_uci_patch_attention_transfer_baseline.py
python scripts/run_uci_source_mlp_transfer_baseline.py
python scripts/run_uci_neural_tdconv_residual_check.py
python scripts/build_paper2_reference_assets.py
python scripts/integrate_uci_load_transfer_results.py
python scripts/integrate_uci_client_stat_tests.py
python scripts/integrate_uci_random_conv_results.py
python scripts/integrate_uci_trainable_tdconv_results.py
python scripts/integrate_uci_neural_tdconv_residual_results.py
python scripts/build_paper2_ieee_access_submission_pack.py
```

## Key Result Anchors

- UCI transfer baseline: 28-day target-only ridge reaches mean RMSE 82.78 on 10 target clients.
- Masked-reconstruction representation: zero-label source head reaches mean RMSE 75.69 and beats the 28-day target-only ridge on 9/10 clients.
- Masked-reconstruction 28-day adapter reaches mean RMSE 74.96 and beats the 28-day target-only ridge on 9/10 clients.
- Random-convolution 28-day adapter reaches mean RMSE 67.50, beats the masked-reconstruction 28-day adapter on 10/10 clients, and improves mean RMSE by 11.33 percent relative to that adapter.
- Random-convolution 7-day adapter beats the 7-day target ridge on 10/10 clients.
- Trainable dilated-convolution ridge 28-day adapter reaches mean RMSE 65.29, beats the random-convolution 28-day adapter on 10/10 clients, and improves mean RMSE by 3.32 percent relative to that adapter.
- Multi-seed TDConv stability: across eight source-window subsampling seeds, the 28-day TDConv adapter reaches 65.338 +/- 0.056 mean RMSE, with every seed beating the random-convolution 28-day adapter on 10/10 clients and the 28-day target ridge on 9/10 clients.
- CPU-only patch-attention 28-day adapter reaches mean RMSE 64.76, beats the TDConv 28-day adapter on 9/10 clients, and exposes target-only patch fitting as a negative control.
- Source-trained MLP 28-day adapter reaches mean RMSE 86.06, loses against patch-attention and TDConv on 10/10 clients, and is retained as a neural-capacity negative-control boundary.
- Neural TDConv residual-head 28-day adapter reaches mean RMSE 65.58, is neutral against the trainable TDConv 28-day adapter at 5/10 wins, but still beats the random-convolution 28-day adapter on 10/10 clients and the 28-day target ridge on 9/10 clients.

## Publication Boundary

Do not add raw local customer identifiers, account numbers, contract files, settlement sheets, or non-public load records to this supplement. Local Hunan and Shandong operational records are application-context evidence only unless written publication permission is obtained.
