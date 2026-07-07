# Paper 1 Data and Code Availability Pack

Generated: 2026-06-30

Manuscript: Sequence-Anchored GraphPatch Residual Learning for Uncertainty-Aware Electricity Price Forecasting in Virtual Power Plant Market Operation.

Author and contact: zhijie REN, College of Computer Science, Hunan University, Lushan South Road, Yuelu District, Changsha, Hunan 410082, China, 471062741@qq.com.

Purpose: settle the data/code availability wording and public-release route for the Paper 1 Engineering Applications of Artificial Intelligence submission candidate.

## Recommended Route

Use a staged route:

1. Submit the current public-data-only reproducibility package as journal supplementary material during review.
2. A public GitHub repository and pre-DOI release now exist for the unified three-paper reproducibility package.
3. Archive the tagged release through Zenodo, Figshare, OSF, or an equivalent citable repository before inserting a DOI.

This route keeps review fast, avoids exposing non-public Hunan/Shandong material, and still gives the final article a durable repository identifier.

Current public repository: https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility

Current pre-DOI release: https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/tag/v0.1.0-pre-doi

Current release asset: https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/download/v0.1.0-pre-doi/three_paper_public_repository_staging_bundle.zip

Commit: 69a3fc05b02ef27686bd9bd1ca422096c21f21a2

DOI status: DOI pending: no Zenodo/Figshare/OSF DOI has been issued yet.

## Public and Non-Public Boundary

Publicly shareable:

- OPSD processed hourly benchmark used for the reproducible main experiments.
- Scripts for OPSD download/preprocessing, baseline execution, GraphPatch experiments, figures, reference assets, and package verification.
- Derived result CSV files and rendered manuscript figures.
- Anonymous manuscript, title/declarations file, highlights, submission checklist, verified BibTeX, and reference audit files.

Not shareable without written permission:

- Hunan local user-load files, private electricity-contract material, settlement sheets, account-level identifiers, and customer-specific operational records.
- Shandong disclosure or node-price files that cannot be redistributed under a clear public license.
- Any non-public virtual-power-plant bidding details, resource-owner identifiers, or contractual pricing terms.

## Manuscript Data Availability Statement

Use this version if uploading the zip as journal supplementary material only:

The public-data reproducibility package is supplied as supplementary material. It contains the processed public OPSD benchmark data, experiment scripts, result tables, rendered figures, verified references, and a manifest with SHA-256 checksums. The original OPSD time-series source can be retrieved from the public Open Power System Data repository. Local Hunan and Shandong operational records were used only as non-public application-context evidence and are not redistributed because they may contain confidential market-operation or customer-related information.

Use this version after a Zenodo DOI is available:

The data and code supporting the public empirical claims are archived at [repository DOI to be inserted]. The archive contains the processed public OPSD benchmark data, experiment scripts, result tables, rendered figures, verified references, and a checksum manifest. The original OPSD source is publicly available from Open Power System Data. Local Hunan and Shandong operational records are not redistributed because they may contain confidential market-operation or customer-related information; these records are not required to reproduce the public benchmark claims.

Use this version for a non-anonymous submission or title-page/declarations/supporting-file route before DOI assignment:

The public-data reproducibility package is available in the public GitHub repository https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility, release https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/tag/v0.1.0-pre-doi. The release asset contains the unified public-data-only reproducibility package for the three related manuscripts and excludes non-public Hunan/Shandong operational records. A persistent DOI has not yet been issued; the DOI field should be added only after a Zenodo, Figshare, OSF, or equivalent archive page exists.

Anonymous-review boundary for EAAI:

Do not place the GitHub repository URL in the anonymous main manuscript if the review workflow treats repository ownership as identifying information. For double-anonymous review, use the supplementary ZIP route or an anonymized repository mechanism approved by the journal portal, and keep the author-owned GitHub URL in the title-page/declarations or post-acceptance data-availability update.

Use this short portal-box version:

Public OPSD data, code, derived results, figures, and checksum manifest are provided in the supplementary reproducibility package. A public GitHub pre-DOI release is available at https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/tag/v0.1.0-pre-doi; no persistent DOI has been issued yet. Non-public Hunan/Shandong application records are excluded and are not required to reproduce the main public benchmark claims.

## Code Availability Statement

Use this version in the manuscript or submission portal:

The experiment code used for the public OPSD benchmark is included in the reproducibility package, together with the command order, package requirements, generated results, figures, and manifest checksums. A public GitHub pre-DOI release exists at https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/tag/v0.1.0-pre-doi. Before inserting a DOI into the manuscript, archive the tagged release through a DOI-backed service and verify that the DOI page points to the same repository state or release asset.

## Repository Deposit Metadata

Recommended repository title:

Sequence-Anchored GraphPatch Residual Learning for Uncertainty-Aware Electricity Price Forecasting in Virtual Power Plant Market Operation: Reproducibility Package

Repository description:

Public-data code and derived artifacts for a Paper 1 submission candidate on uncertainty-aware spatio-temporal electricity price forecasting for virtual power plant market operation.

Creator:

- zhijie REN; College of Computer Science, Hunan University, Lushan South Road, Yuelu District, Changsha, Hunan 410082, China; email: 471062741@qq.com; ORCID: 0009-0006-1048-6640.

Suggested licenses:

- Code: MIT License, if the author approves permissive software reuse.
- Text/figures/derived public-data artifacts: CC BY 4.0, if the author approves attribution-based reuse.
- Do not apply these licenses to non-public local data; those files must remain outside the public repository.

Recommended version tag:

v1.0.0-paper1-eaai-submission

Recommended repository top-level structure:

- README.md
- CITATION.cff
- .zenodo.json
- requirements-paper1.txt
- RUN_COMMANDS.sh
- scripts/
- data/processed/
- results/
- figures/
- references/
- manuscript/
- repository_release/
- MANIFEST.csv

## Release Checklist

- Verify no private/local-data filenames appear in the public package.
- Replace local absolute path constants with repository-relative paths.
- Run the full OPSD reproduction sequence from a clean checkout.
- Rebuild MANIFEST.csv and verify all SHA-256 checksums.
- Confirm the manuscript data availability statement matches the actual repository URL, DOI, or supplementary-file route.
- Confirm author metadata, funding statement, competing-interest declaration, and AI-use declaration are consistent across the portal, title page, and manuscript.
- Keep acknowledgements absent unless the author explicitly adds one.

## Operational Decision

For the first real submission, the safest upload path is:

- Anonymous manuscript DOCX.
- Title page/declarations DOCX.
- Highlights DOCX.
- Cover letter DOCX.
- Public reproducibility zip: submission_supplements/paper1_eaai_reproducibility_package.zip.
- Data/code availability text copied from this pack into the portal.

For final accepted publication, add:

- DOI-backed repository link.
- Final CITATION.cff.
- Final .zenodo.json.
- Final license files after author approval.
