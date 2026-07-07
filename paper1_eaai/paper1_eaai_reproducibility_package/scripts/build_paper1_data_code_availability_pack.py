from pathlib import Path

from create_submission_candidate_manuscripts import rebuild_docx


BASE = Path("/Users/joey/Documents/paper")
ROOT = BASE / "outputs" / "019f1500-3356-7a02-b657-097bf5e23528"
EAAI = ROOT / "paper_package" / "target_journal_eaai_paper1"


TITLE = "Paper 1 Data and Code Availability Pack"
MANUSCRIPT_TITLE = (
    "Sequence-Anchored GraphPatch Residual Learning for Uncertainty-Aware "
    "Electricity Price Forecasting in Virtual Power Plant Market Operation"
)
AUTHOR = "zhijie REN"
AFFILIATION = "College of Computer Science, Hunan University, Lushan South Road, Yuelu District, Changsha, Hunan 410082, China"
EMAIL = "471062741@qq.com"
ORCID = "0009-0006-1048-6640"
SUPP_ZIP = "submission_supplements/paper1_eaai_reproducibility_package.zip"
GITHUB_REPO = "https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility"
GITHUB_RELEASE = "https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/tag/v0.1.0-pre-doi"
GITHUB_ASSET = "https://github.com/joeyrain0223-eng/vpp-forecasting-three-paper-reproducibility/releases/download/v0.1.0-pre-doi/three_paper_public_repository_staging_bundle.zip"
GITHUB_COMMIT = "69a3fc05b02ef27686bd9bd1ca422096c21f21a2"
DOI_STATUS = "DOI pending: no Zenodo/Figshare/OSF DOI has been issued yet."


AVAILABILITY_MD = f"""# {TITLE}

Generated: 2026-06-30

Manuscript: {MANUSCRIPT_TITLE}.

Author and contact: {AUTHOR}, {AFFILIATION}, {EMAIL}.

Purpose: settle the data/code availability wording and public-release route for the Paper 1 Engineering Applications of Artificial Intelligence submission candidate.

## Recommended Route

Use a staged route:

1. Submit the current public-data-only reproducibility package as journal supplementary material during review.
2. A public GitHub repository and pre-DOI release now exist for the unified three-paper reproducibility package.
3. Archive the tagged release through Zenodo, Figshare, OSF, or an equivalent citable repository before inserting a DOI.

This route keeps review fast, avoids exposing non-public Hunan/Shandong material, and still gives the final article a durable repository identifier.

Current public repository: {GITHUB_REPO}

Current pre-DOI release: {GITHUB_RELEASE}

Current release asset: {GITHUB_ASSET}

Commit: {GITHUB_COMMIT}

DOI status: {DOI_STATUS}

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

The public-data reproducibility package is available in the public GitHub repository {GITHUB_REPO}, release {GITHUB_RELEASE}. The release asset contains the unified public-data-only reproducibility package for the three related manuscripts and excludes non-public Hunan/Shandong operational records. A persistent DOI has not yet been issued; the DOI field should be added only after a Zenodo, Figshare, OSF, or equivalent archive page exists.

Anonymous-review boundary for EAAI:

Do not place the GitHub repository URL in the anonymous main manuscript if the review workflow treats repository ownership as identifying information. For double-anonymous review, use the supplementary ZIP route or an anonymized repository mechanism approved by the journal portal, and keep the author-owned GitHub URL in the title-page/declarations or post-acceptance data-availability update.

Use this short portal-box version:

Public OPSD data, code, derived results, figures, and checksum manifest are provided in the supplementary reproducibility package. A public GitHub pre-DOI release is available at {GITHUB_RELEASE}; no persistent DOI has been issued yet. Non-public Hunan/Shandong application records are excluded and are not required to reproduce the main public benchmark claims.

## Code Availability Statement

Use this version in the manuscript or submission portal:

The experiment code used for the public OPSD benchmark is included in the reproducibility package, together with the command order, package requirements, generated results, figures, and manifest checksums. A public GitHub pre-DOI release exists at {GITHUB_RELEASE}. Before inserting a DOI into the manuscript, archive the tagged release through a DOI-backed service and verify that the DOI page points to the same repository state or release asset.

## Repository Deposit Metadata

Recommended repository title:

Sequence-Anchored GraphPatch Residual Learning for Uncertainty-Aware Electricity Price Forecasting in Virtual Power Plant Market Operation: Reproducibility Package

Repository description:

Public-data code and derived artifacts for a Paper 1 submission candidate on uncertainty-aware spatio-temporal electricity price forecasting for virtual power plant market operation.

Creator:

- {AUTHOR}; {AFFILIATION}; email: {EMAIL}; ORCID: {ORCID}.

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
- Public reproducibility zip: {SUPP_ZIP}.
- Data/code availability text copied from this pack into the portal.

For final accepted publication, add:

- DOI-backed repository link.
- Final CITATION.cff.
- Final .zenodo.json.
- Final license files after author approval.
"""


DEPOSIT_JSON = """{
  "title": "Sequence-Anchored GraphPatch Residual Learning for Uncertainty-Aware Electricity Price Forecasting in Virtual Power Plant Market Operation: Reproducibility Package",
  "upload_type": "software",
  "description": "Public-data code and derived artifacts for a Paper 1 submission candidate on uncertainty-aware spatio-temporal electricity price forecasting for virtual power plant market operation.",
  "creators": [
    {
      "name": "Ren, Zhijie",
      "affiliation": "College of Computer Science, Hunan University"
    }
  ],
  "access_right": "open",
  "license": "MIT",
  "keywords": [
    "electricity price forecasting",
    "virtual power plant",
    "spatio-temporal learning",
    "conformal prediction",
    "Open Power System Data"
  ],
  "related_identifiers": [
    {
      "identifier": "manuscript DOI to be added after publication",
      "relation": "isSupplementTo",
      "scheme": "doi"
    }
  ]
}
"""


CITATION_CFF = f"""cff-version: 1.2.0
message: "If you use this reproducibility package, please cite the associated manuscript and this archived software record."
title: "Sequence-Anchored GraphPatch Residual Learning for Uncertainty-Aware Electricity Price Forecasting in Virtual Power Plant Market Operation: Reproducibility Package"
authors:
  - family-names: "Ren"
    given-names: "Zhijie"
    affiliation: "College of Computer Science, Hunan University"
version: "v1.0.0-paper1-eaai-submission"
date-released: "2026-06-30"
repository-code: "{GITHUB_REPO}"
doi: "DOI pending; insert only after DOI-backed archive release"
keywords:
  - "electricity price forecasting"
  - "virtual power plant"
  - "spatio-temporal learning"
  - "conformal prediction"
  - "Open Power System Data"
license: "MIT"
"""


def main() -> None:
    EAAI.mkdir(parents=True, exist_ok=True)
    md_path = EAAI / "paper_1_data_code_availability_pack.md"
    md_path.write_text(AVAILABILITY_MD, encoding="utf-8")
    rebuild_docx(md_path, AVAILABILITY_MD)
    (EAAI / "paper_1_repository_deposit_metadata.json").write_text(DEPOSIT_JSON, encoding="utf-8")
    (EAAI / "paper_1_repository_citation.cff").write_text(CITATION_CFF, encoding="utf-8")
    print(md_path)
    print(md_path.with_suffix(".docx"))
    print(EAAI / "paper_1_repository_deposit_metadata.json")
    print(EAAI / "paper_1_repository_citation.cff")


if __name__ == "__main__":
    main()
