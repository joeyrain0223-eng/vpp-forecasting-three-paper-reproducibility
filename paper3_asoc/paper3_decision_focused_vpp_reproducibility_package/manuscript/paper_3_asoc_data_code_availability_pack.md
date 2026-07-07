# Applied Soft Computing Paper 3 Data and Code Availability Pack

## Public Dataset

The reproducible experimental layer uses the Open Power System Data time-series package, version 2020-10-06:

https://data.open-power-system-data.org/time_series/2020-10-06/

## Supplementary Package

- Package directory: `submission_supplements/paper3_decision_focused_vpp_reproducibility_package/`
- Zip archive: `submission_supplements/paper3_decision_focused_vpp_reproducibility_package.zip`
- Audit report: `submission_supplements/paper3_decision_focused_vpp_reproducibility_package_audit.md`
- Public GitHub repository: https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility
- Public GitHub release: https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/tag/v0.1.0-pre-doi
- Release asset: https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/download/v0.1.0-pre-doi/three_paper_public_repository_staging_bundle.zip
- Commit: 69a3fc05b02ef27686bd9bd1ca422096c21f21a2
- DOI status: DOI pending: no Zenodo/Figshare/OSF DOI has been issued yet.

## Included Reproducibility Assets

- Processed public OPSD data.
- Daily and aggregate result CSV files.
- Figure PNG files used by the manuscript.
- Verified reference assets and BibTeX.
- Requirements file and reproduction README.
- Scripts for VPP risk simulation, decision-focused policy search, genetic soft-computing policy search, multi-seed genetic stability checking, particle-swarm policy search, surrogate policy-gradient baseline testing, forecast-coupled VPP evaluation, risk-aversion sensitivity, reviewer robustness checks, constrained Q-learning boundary testing, and package verification.

## Recommended Data Statement for the Submission Portal

The public reproducibility layer is based on the Open Power System Data time-series package and is bundled as a public-data-only supplementary package containing processed public data, scripts, result CSV files, figures, reference assets, requirements, manifest hashes, and an audit report. The first submission may upload this package as supplementary material. A public GitHub pre-DOI release is available at https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/tag/v0.1.0-pre-doi; the same release should be archived through Zenodo, Figshare, OSF, or an equivalent citable repository before a DOI is inserted in the final article. Local Hunan and Shandong operational records are not redistributed because they are non-public application-context data. The GIS infrastructure-context audit uses local copies of open or externally sourced grid/resource metadata only as background evidence; raw OSM/GEM/GIS files are not included in the journal supplement until source-page, attribution, and redistribution gates are confirmed.

## Private-Data Boundary

Local Hunan and Shandong operational records are not included in the public supplement. They are not necessary to reproduce the main paper claims, and they should remain local unless the author obtains explicit redistribution permission.

## Current Audit Summary

```text
# Paper 3 Supplement Package Audit

Status: PASS

## Manifest

- Manifest rows: 104
- Missing files: 0
- Hash/size mismatches: 0

## Zip Integrity

- Zip path: `./submission_supplements/paper3_decision_focused_vpp_reproducibility_package.zip`
- Zip members: 105
- Uncompressed bytes: 113981765
- Compressed bytes: 26447074
- First bad member from `ZipFile.testzip()`: ``

## Key Result Checks

- Extended VPP risk simulator mean revenues: rolling-28d=20.880618, robust=20.100934, prev-day=17.389398.
- Held-out DF policy revenue: 21.638521.
- Forecast-coupled DF revenue/regret: 31.434681 / 4.669494.
- Risk sweep CVaR10 lambda=0.00 / 0.25: -1.852859 / -1.275977.
- High-stress best revenue method: `DF policy search (revenue)`.
- Fine-grid best revenue method: `Coarse revenue grid`.
- Fine-grid best CVaR method: `Coarse risk-adjusted grid`.
- Constrained Q-learning revenue/CVaR10: -5.654693 / -24.534076.
- Constrained Q-learning losses versus DF revenue policy: 1280 paired test days.
- Genetic risk-adjusted policy revenue/CVaR10/loss days: 21.652815 / -1.390343 / 111.
- Genetic risk-adjusted versus DF revenue wins/losses: 237 / 240 paired test days.
- Multi-seed genetic revenue mean/std: 21.543300 / 0.078236.
- Multi-seed genetic CVaR10 mean/std: -1.530009 / 0.162559.
- Multi-seed genetic seeds at or above DF revenue baseline: 1 of 8.
- PSO risk-adjusted policy revenue/CVaR10/loss days: 21.596237 / -1.741398 / 113.
- PSO risk-adjusted versus genetic risk-adjusted wins/losses: 173 / 194 paired test days.
- Numeric/result checks pass: True.
- Reference register: 32 DOI-verified, 3 URL entries, 0 review-required.

## Privacy Boundary

- Local/private-data filename flags: 0
- Local/private-data content mentions in text-like files: 7

## Manuscript Claim Boundary

- Forbidden old implementation claims found: 0
- Required auditable-policy-search terms missing: 0

Flagged paths:

- None

Text files mentioning local-data terms:

- `README.md`
- `DATA_LICENSE_AND_REDISTRIBUTION.md`
- `manuscript/paper_3_asoc_manuscript_candidate.md`
- `context/gis_energy_infrastructure_evidence.md`
- `scripts/verify_paper3_supplement_package.py`
- `scripts/build_gis_energy_infrastructure_evidence.py`
- `data/PUBLIC_DATA_SOURCE_NOTES.md`

Forbidden claim findings:

- None

Missing required claim terms:

- None

## Interpretation

This audit verifies package structure, checksums, zip readability, key public result consistency, reference-register status, absence of local/private-data filenames in bundled paths, and consistency between the bundled manuscript copy and the current auditable policy-search claim boundary. Text mentions of local case-study boundaries are reported for human review but do not fail the package unless raw/private filenames are bundled as file paths.
```
