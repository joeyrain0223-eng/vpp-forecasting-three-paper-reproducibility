# IEEE Access Paper 2 Data and Code Availability Pack

## Public Reproducibility Layer

Primary public datasets:

- UCI Electricity Load Diagrams 2011-2014: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
- Open Power System Data time-series package: https://data.open-power-system-data.org/time_series/2020-10-06/
- UCI Appliances Energy Prediction: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

Main reproducibility scripts:

- `work/run_uci_load_transfer_baselines.py`
- `work/run_uci_ssl_representation_prototype.py`
- `work/run_uci_ssl_cold_start_diagnostics.py`
- `work/run_uci_client_statistical_tests.py`
- `work/run_uci_random_conv_representation.py`
- `work/run_uci_trainable_tdconv_baseline.py`
- `work/run_uci_tdconv_multiseed_stability.py`
- `work/run_uci_patch_attention_transfer_baseline.py`
- `work/run_uci_source_mlp_transfer_baseline.py`
- `work/run_uci_neural_tdconv_residual_check.py`
- `work/run_uci_appliances_energy_baselines.py`
- `work/run_public_opsd_baselines.py`

Primary result files:

- `public_experiment_results/uci_load_transfer_summary.csv`
- `public_experiment_results/uci_ssl_representation_summary.csv`
- `public_experiment_results/uci_ssl_cold_start_summary.csv`
- `public_experiment_results/uci_ssl_client_level_stat_tests.csv`
- `public_experiment_results/uci_random_conv_representation_summary.csv`
- `public_experiment_results/uci_random_conv_client_level_tests.csv`
- `public_experiment_results/uci_trainable_tdconv_baseline_summary.csv`
- `public_experiment_results/uci_trainable_tdconv_client_level_tests.csv`
- `public_experiment_results/uci_tdconv_multiseed_stability_results.csv`
- `public_experiment_results/uci_tdconv_multiseed_stability_by_seed.csv`
- `public_experiment_results/uci_tdconv_multiseed_stability_summary.csv`
- `public_experiment_results/uci_tdconv_multiseed_stability_tests.csv`
- `public_experiment_results/uci_patch_attention_transfer_summary.csv`
- `public_experiment_results/uci_patch_attention_transfer_client_level_tests.csv`
- `public_experiment_results/uci_source_mlp_transfer_summary.csv`
- `public_experiment_results/uci_source_mlp_transfer_client_level_tests.csv`
- `public_experiment_results/uci_source_mlp_transfer_training_diagnostics.csv`
- `public_experiment_results/uci_neural_tdconv_residual_summary.csv`
- `public_experiment_results/uci_neural_tdconv_residual_client_level_tests.csv`
- `public_experiment_results/uci_neural_tdconv_residual_training_diagnostics.csv`
- `public_experiment_results/uci_appliances_energy_dataset_stats.csv`
- `public_experiment_results/uci_appliances_energy_baselines.csv`

Generated supplement:

- Package directory: `submission_supplements/paper2_transferable_load_reproducibility_package/`
- Zip archive: `submission_supplements/paper2_transferable_load_reproducibility_package.zip`
- Audit report: `submission_supplements/paper2_transferable_load_reproducibility_package_audit.md`
- Current zip status: Present

## Non-Public Boundary

Local Hunan and Shandong files are not included in a public release unless the author obtains explicit permission. They are treated only as application-context evidence and must not be uploaded as raw supplementary data.

## Recommended First-Submission Wording

The public-data layer of this study is reproducible from the UCI Electricity Load Diagrams 2011-2014 dataset, the Open Power System Data time-series package, and the UCI Appliances Energy Prediction dataset. The public supplement contains processed public data, preprocessing scripts, result tables, generated figures, verified references, manuscript files, manifest hashes, and an audit report for the main transfer-learning claims, including masked-reconstruction, random-convolution, trainable dilated-convolution ridge, multi-seed TDConv stability, CPU-only patch-attention transfer, source-trained MLP transfer, neural TDConv residual-head, and external UCI Appliances load checks. Local Hunan and Shandong operational records are used only as non-public application-context evidence and are not redistributed.

## Portal-Ready Data and Code Wording

### Data Availability Statement

The OPSD public benchmark [18] is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark [17] is reproducible from `run_uci_load_transfer_baselines.py`; the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`; the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`; the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`; the multi-seed TDConv stability check is reproducible from `run_uci_tdconv_multiseed_stability.py`; the CPU-only patch-attention transfer check is reproducible from `run_uci_patch_attention_transfer_baseline.py`; the source-trained MLP transfer check is reproducible from `run_uci_source_mlp_transfer_baseline.py`; and the neural TDConv residual-head check is reproducible from `run_uci_neural_tdconv_residual_check.py`. The second public load-dataset check on UCI Appliances Energy Prediction [21] is reproducible from `run_uci_appliances_energy_baselines.py`, and its 1-hour, 3-hour, 6-hour, and 12-hour robustness extension is reproducible from `run_uci_appliances_multihorizon_robustness.py`. Main claims are reproducible without relying solely on private data.

### Data and Code Availability

The reproducible public-data layer uses the UCI Electricity Load Diagrams 2011-2014 dataset, the Open Power System Data time-series package, and the UCI Appliances Energy Prediction dataset. The public reproducibility supplement contains processed public data, preprocessing scripts, result tables, generated figures, verified references, manuscript files, manifest hashes, and an audit report for the UCI multi-client transfer, source-pooled representation, masked-reconstruction representation, cold-start, client-level statistical-test, random-convolution representation checks, trainable dilated-convolution ridge checks, multi-seed TDConv stability checks, CPU-only patch-attention transfer checks, source-trained MLP transfer checks, neural TDConv residual-head checks, OPSD public load baseline, and external UCI Appliances one-hour-ahead and multi-horizon load sanity checks. Local Hunan and Shandong operational records are used only as non-public application-context evidence and are not redistributed.

## Repository Route

For the first IEEE Access submission, upload `paper2_transferable_load_reproducibility_package.zip` as supplementary material if the portal allows. After author approval, mirror the same public package to GitHub and Zenodo or another citable repository before final publication metadata is locked.

## Current Audit Summary

```text
# Paper 2 Supplement Package Audit

Status: PASS

## Manifest

- Manifest rows: 104
- Missing files: 0
- Hash/size mismatches: 0

## Zip Integrity

- Zip path: `./submission_supplements/paper2_transferable_load_reproducibility_package.zip`
- Zip members: 105
- Uncompressed bytes: 57605427
- Compressed bytes: 13173543
- First bad member from `ZipFile.testzip()`: ``

## Key Result Checks

- UCI 28-day target-only ridge RMSE: 82.783638.
- Zero-label SSL source-head RMSE/wins/p: 75.688904 / 9 / 0.021484375.
- 28-day SSL adapter RMSE/wins/p: 74.961866 / 9 / 0.021484375.
- Random-convolution 28-day adapter RMSE: 67.500061.
- Random-convolution 28-day adapter vs MR 28-day adapter wins/gain/p: 10 / 11.330348% / 0.001953125.
- Random-convolution 7-day adapter vs 7-day target ridge wins: 10.
- Trainable dilated-convolution 28-day adapter RMSE: 65.290727.
- Trainable dilated-convolution source-head RMSE: 65.471364.
- Trainable dilated-convolution 28-day adapter vs random-convolution 28-day adapter wins/gain/p: 10 / 3.322311% / 0.001953125.
- Trainable dilated-convolution source-head vs random-convolution source-head wins: 10.
- Trainable dilated-convolution 28-day adapter vs 28-day target ridge wins/gain/p: 9 / 23.778909% / 0.021484375.
- Target-only trainable dilated-convolution 28-day head RMSE/gain vs target ridge: 91.787933 / -7.492262%.
- Multi-seed TDConv 28-day adapter RMSE mean/std/min/max: 65.338206 / 0.055629 / 65.283000 / 65.465458.
- Multi-seed TDConv minimum wins vs target ridge and random-convolution adapter: 9 / 10.
- CPU-only patch-attention 28-day adapter/source-head RMSE: 64.760508 / 65.118562.
- CPU-only patch-attention 28-day adapter vs TDConv 28-day adapter wins/gain/p: 9 / 1.899956% / 0.021484375.
- CPU-only patch-attention 28-day adapter vs random-convolution and target ridge wins: 9 / 9.
- Target-only patch-attention 28-day head RMSE/gain vs target ridge: 130.567929 / -54.377953%.
- Source-trained MLP 28-day adapter/source-head/hidden-head RMSE: 86.056321 / 88.212496 / 85.882583.
- Source-trained MLP validation RMSE and selected epoch: 0.091722 / 32.
- Source-trained MLP 28-day adapter vs patch-attention wins-losses/gain/p: 0-10 / -36.581375% / 0.001953125.
- Source-trained MLP 28-day adapter vs target ridge wins-losses/p: 5-5 / 1.000000000.
- Neural TDConv residual 28-day adapter RMSE / shrinkage: 65.581836 / 0.25.
- Neural TDConv residual source-head / 7-day adapter RMSE: 65.592087 / 66.566272.
- Neural TDConv residual 28-day adapter vs trainable TDConv 28-day adapter wins-losses/p: 5-5 / 1.000000000.
- Neural TDConv residual 28-day adapter vs random-convolution 28-day adapter wins/gain/p: 10 / 2.976753% / 0.001953125.
- Neural TDConv residual 28-day adapter vs 28-day target ridge wins/gain/p: 9 / 23.471433% / 0.021484375.
- Neural TDConv residual 7-day adapter vs 7-day target ridge wins: 10.
- Neural residual final train/validation residual RMSE: 0.088782 / 0.095402.
- UCI Appliances lag-weather / random-window / persistence / seasonal RMSE: 78.588546 / 78.612194 / 98.976376 / 107.036809.
- UCI Appliances multi-horizon best models and RMSE: 1h Lag-weather ridge 78.595971; 3h Random-window ridge 83.543521; 6h Random-window ridge 85.445262; 12h Random-window ridge 84.899684.
- OPSD public load baseline zones represented: 4.
- Numeric/result checks pass: True.
- Reference register: 20 DOI-verified, 10 URL entries, 0 review-required.

## Privacy Boundary

- Local/private-data filename flags: 0
- Local/private-data content mentions in text-like files: 11

Flagged paths:

- None

Text files mentioning local-data terms:

- `README.md`
- `DATA_LICENSE_AND_REDISTRIBUTION.md`
- `scripts/integrate_public_opsd_results.py`
- `scripts/integrate_uci_load_transfer_results.py`
- `scripts/verify_paper2_supplement_package.py`
- `scripts/build_paper2_ieee_access_submission_pack.py`
- `data/PUBLIC_DATA_SOURCE_NOTES.md`
- `manuscript/ieee_access/paper_2_ieee_access_manuscript_candidate.md`
- `manuscript/ieee_access/paper_2_ieee_access_data_code_availability_pack.md`
- `manuscript/submission_candidate/paper_2_transferable_load_forecasting.md`
- `manuscript/main/paper_2_transferable_load_forecasting.md`

## Interpretation

This audit verifies package structure, checksums, zip readability, key public result consistency, reference-register status, and absence of local/private-data filenames in bundled paths. Text mentions of local application boundaries are reported for human review but do not fail the package unless raw/private filenames are bundled as file paths.
```
