# Paper 1 EAAI Manual Reference Audit

Audit date: 2026-06-30

Purpose: separate safe DOI/reference updates from noisy automated Crossref first-hit matches. The automated table is saved as `paper1_eaai_crossref_reference_audit.md`, but several first hits are false positives and must not be inserted automatically.

## High-Confidence DOI Candidates

These entries have title-level agreement between the manuscript reference and the Crossref hit, but should still be checked in Zotero, EndNote, Crossref, or the publisher page before final insertion.

| # | Manuscript item | DOI candidate | Status |
|---|---|---|---|
| 2 | Temporal Fusion Transformers for interpretable multi-horizon time series forecasting | 10.1016/j.ijforecast.2021.03.012 | High confidence |
| 3 | Informer: Beyond efficient Transformer for long sequence time-series forecasting | 10.1609/aaai.v35i12.17325 | High confidence |
| 6 | Are Transformers effective for time series forecasting? | 10.1609/aaai.v37i9.26317 | High confidence |
| 8 | Graph WaveNet for deep spatial-temporal graph modeling | 10.24963/ijcai.2019/264 | High confidence |
| 9 | A gentle introduction to conformal prediction and distribution-free uncertainty quantification | 10.1561/2200000101 | High confidence |
| 10 | From predictive to prescriptive analytics | 10.1287/mnsc.2018.3253 | High confidence |
| 11 | Smart predict, then optimize | 10.1287/mnsc.2020.3922 | High confidence |
| 14 | Global energy forecasting competition 2012 | 10.1016/j.ijforecast.2013.07.001 | High confidence |
| 15 | Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond | 10.1016/j.ijforecast.2016.02.001 | High confidence |
| 16 | Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms | 10.1016/j.apenergy.2018.02.069 | High confidence |
| 17 | Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark | 10.1016/j.apenergy.2021.116983 | High confidence |
| 18 | Electricity price forecasting: A review of the state-of-the-art with a look into the future | 10.1016/j.ijforecast.2014.08.008 | High confidence |
| 20 | Recent advances in electricity price forecasting: A review of probabilistic forecasting | 10.1016/j.rser.2017.05.234 | High confidence |
| 26 | Convex Optimization | 10.1017/cbo9780511804441 | High confidence |

## Likely False Crossref Hits

Do not insert these automated DOI candidates without manual correction.

| # | Manuscript item | Automated hit problem |
|---|---|---|
| 1 | Attention is all you need | Crossref returned a nonstandard DOI candidate; NeurIPS proceedings entries commonly have no DOI. Use the NeurIPS URL or omit DOI after reference-manager check. |
| 4 | Autoformer: Decomposition Transformers with auto-correlation for long-term series forecasting | Automated hit returned Non-Stationary Transformers, not Autoformer. |
| 5 | PatchTST / A time series is worth 64 words | Automated hit returned TSMixer, not PatchTST. Use OpenReview/ICLR metadata unless a verified DOI exists. |
| 7 | N-BEATS | Automated hit returned a different time-series paper. |
| 12 | OptNet | Automated hit returned an unrelated 2008 ICMLC paper. |
| 13 | Task-based end-to-end model learning in stochastic optimization | Automated hit returned an unrelated energy-based model paper. |
| 19 | Beating the Naive: Using transfer learning to improve day-ahead electricity price forecasting | Automated hit returned a different Energy Economics paper. |
| 21 | Coherent probabilistic forecasts for hierarchical time series | Automated hit returned an unrelated Sensors transfer-learning paper. |
| 22 | Forecasting: Principles and Practice | Automated hit returned the R package record, not the book metadata. |
| 23 | Open Power System Data, Data Package Time Series | Automated hit returned an unrelated compression paper. Cite the dataset URL and version instead. |
| 24 | Virtual power plant: A review | Automated hit returned a hydrogen production review, not the VPP review. |
| 25 | Virtual power plant management for sustainable power systems: A review | Automated hit returned a microgrid/VPP review with different title; verify manually before deciding whether to replace the current reference. |

## Next Reference Work

1. Export the high-confidence rows to Zotero/EndNote or manually add DOI fields in a CSL/BibTeX file.
2. Manually repair the likely false-hit references using publisher pages, DBLP, OpenReview, NeurIPS, PMLR, or dataset landing pages.
3. Rebuild the EAAI reference section from the confirmed reference manager export rather than continuing to hand-edit plain text.

