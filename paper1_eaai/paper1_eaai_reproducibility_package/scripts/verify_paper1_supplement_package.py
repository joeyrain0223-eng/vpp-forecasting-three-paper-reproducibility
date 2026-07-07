from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
SUPP = ROOT
ZIP_PATH = ROOT.parent / "paper1_eaai_reproducibility_package.zip"
AUDIT = ROOT.parent / "paper1_eaai_reproducibility_package_audit.md"

FORBIDDEN_MANUSCRIPT_TERMS = [
    "B-class",
    "C-class",
    "B 类",
    "C 类",
    "school-recognition",
    "graduation rule",
    "doctoral graduation",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest() -> tuple[list[dict], list[str], list[str]]:
    rows = list(csv.DictReader((SUPP / "MANIFEST.csv").open(encoding="utf-8")))
    missing = []
    mismatched = []
    for row in rows:
        path = SUPP / row["relative_path"]
        if not path.exists():
            missing.append(row["relative_path"])
            continue
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            mismatched.append(row["relative_path"])
    return rows, missing, mismatched


def verify_zip() -> tuple[int, int, int, str]:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        bad = zf.testzip()
        infos = zf.infolist()
        return len(infos), sum(i.file_size for i in infos), sum(i.compress_size for i in infos), bad or ""


def verify_key_results() -> dict:
    result_dir = SUPP / "results"
    seq = pd.read_csv(result_dir / "opsd_sequence_anchor_graphpatch_price_summary.csv")
    seq_spike = seq[(seq["regime"] == "spike") & (seq["model"] == "Sequence-anchored GraphPatch residual")]
    tdconv = pd.read_csv(result_dir / "opsd_tdconv_sequence_anchor_graphpatch_price_summary.csv")
    tdconv_all = tdconv[(tdconv["regime"] == "all") & (tdconv["model"] == "TDConv-inclusive GraphPatch residual")]
    tdconv_spike = tdconv[(tdconv["regime"] == "spike") & (tdconv["model"] == "TDConv-inclusive GraphPatch residual")]
    patch = pd.read_csv(result_dir / "opsd_patch_attention_price_baseline_summary.csv")
    patch_pair = pd.read_csv(result_dir / "opsd_patch_attention_price_baseline_paired_tests.csv")
    patch_spike = patch[(patch["regime"] == "spike") & (patch["model"] == "Patch-attention sequence ridge")]
    tdconv_seq_spike = patch[(patch["regime"] == "spike") & (patch["model"] == "TDConv-style sequence ridge")]
    patch_pair_spike = patch_pair[(patch_pair["regime"] == "spike") & (patch_pair["model"] == "Patch-attention sequence ridge")]
    spike_sens = pd.read_csv(result_dir / "opsd_sequence_anchor_spike_threshold_sensitivity_aggregate.csv")
    cal_sens = pd.read_csv(result_dir / "opsd_conformal_calibration_window_sensitivity_aggregate.csv")
    rob = pd.read_csv(result_dir / "opsd_graphpatch_robustness_aggregate.csv")
    rolling_spike = rob[(rob["protocol"] == "rolling_origin") & (rob["regime"] == "spike")].iloc[0]
    zone_spike = rob[(rob["protocol"] == "zone_holdout") & (rob["regime"] == "spike")].iloc[0]
    reg = pd.read_csv(SUPP / "references" / "paper1_eaai_verified_reference_register.csv")
    return {
        "sequence_anchor_positive": f"{int((seq_spike['rmse_improvement_pct_vs_anchor'] > 0).sum())}/{len(seq_spike)}",
        "sequence_anchor_mean_rmse_gain_pct": float(seq_spike["rmse_improvement_pct_vs_anchor"].mean()),
        "tdconv_all_positive": f"{int((tdconv_all['rmse_improvement_pct_vs_anchor'] > 0).sum())}/{len(tdconv_all)}",
        "tdconv_spike_positive": f"{int((tdconv_spike['rmse_improvement_pct_vs_anchor'] > 0).sum())}/{len(tdconv_spike)}",
        "tdconv_spike_mean_rmse_gain_pct": float(tdconv_spike["rmse_improvement_pct_vs_anchor"].mean()),
        "patch_attention_rmse_lower_than_tdconv": f"{int((patch_spike['rmse'].to_numpy() < tdconv_seq_spike['rmse'].to_numpy()).sum())}/{len(patch_spike)}",
        "patch_attention_positive_paired": f"{int(((patch_pair_spike['mean_abs_error_delta'] > 0) & (patch_pair_spike['sign_test_p_approx'] < 0.05)).sum())}/{len(patch_pair_spike)}",
        "spike_threshold_positive": f"{int((spike_sens['positive_zones'] == spike_sens['zones']).sum())}/{len(spike_sens)}",
        "spike_threshold_mean_gain_min_pct": float(spike_sens["mean_rmse_gain_pct"].min()),
        "spike_threshold_mean_gain_max_pct": float(spike_sens["mean_rmse_gain_pct"].max()),
        "calibration_window_best_share": float(cal_sens.loc[cal_sens["mean_coverage_error_abs"].idxmin(), "calibration_share"]),
        "calibration_window_best_error": float(cal_sens["mean_coverage_error_abs"].min()),
        "calibration_window_20pct_mean_picp": float(cal_sens.loc[cal_sens["calibration_share"].round(2) == 0.20, "mean_picp"].iloc[0]),
        "rolling_spike_positive": f"{int(rolling_spike['positive_rmse_cases'])}/{int(rolling_spike['cases'])}",
        "rolling_spike_mean_rmse_gain_pct": float(rolling_spike["mean_rmse_improvement_pct"]),
        "zone_holdout_spike_positive": f"{int(zone_spike['positive_rmse_cases'])}/{int(zone_spike['cases'])}",
        "zone_holdout_spike_mean_rmse_gain_pct": float(zone_spike["mean_rmse_improvement_pct"]),
        "doi_confirmed": int((reg["status"] == "doi_confirmed").sum()),
        "url_or_dataset": int(reg["status"].isin(["url_verified_no_doi", "url_verified_no_crossref_doi", "dataset_url_verified"]).sum()),
        "review_required": int((~reg["status"].isin(["doi_confirmed", "url_verified_no_doi", "url_verified_no_crossref_doi", "dataset_url_verified"])).sum()),
    }


def verify_privacy_boundary() -> list[str]:
    forbidden_file_terms = ["hunan", "shandong", "daily_disclosure", "user_load", "node_price"]
    flagged = []
    for path in SUPP.rglob("*"):
        if path.is_file():
            lower = str(path.relative_to(SUPP)).lower()
            if any(term in lower for term in forbidden_file_terms):
                flagged.append(str(path.relative_to(SUPP)))
    return flagged


def verify_manuscript_boundary() -> list[str]:
    flagged = []
    manuscript_dir = SUPP / "manuscript"
    for path in manuscript_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in FORBIDDEN_MANUSCRIPT_TERMS:
            if term in text:
                flagged.append(f"{path.relative_to(SUPP)}::{term}")
    return flagged


def main() -> None:
    rows, missing, mismatched = verify_manifest()
    zip_members, zip_uncompressed, zip_compressed, bad_zip_member = verify_zip()
    key = verify_key_results()
    privacy_flags = verify_privacy_boundary()
    manuscript_flags = verify_manuscript_boundary()
    ok = (
        not missing
        and not mismatched
        and not bad_zip_member
        and not privacy_flags
        and not manuscript_flags
        and key["review_required"] == 0
        and key["tdconv_all_positive"] == "4/4"
        and key["tdconv_spike_positive"] == "3/4"
    )
    audit = f"""# Paper 1 Supplement Package Audit

Status: {'PASS' if ok else 'REVIEW REQUIRED'}

## Manifest

- Manifest rows: {len(rows)}
- Missing files: {len(missing)}
- Hash/size mismatches: {len(mismatched)}

## Zip Integrity

- Zip path: `{ZIP_PATH}`
- Zip members: {zip_members}
- Uncompressed bytes: {zip_uncompressed}
- Compressed bytes: {zip_compressed}
- First bad member from `ZipFile.testzip()`: `{bad_zip_member}`

## Key Result Checks

- Sequence-anchored GraphPatch spike RMSE gains: {key['sequence_anchor_positive']} zones positive; mean gain {key['sequence_anchor_mean_rmse_gain_pct']:.6f}%.
- TDConv-inclusive selected-anchor GraphPatch gains: all-hour {key['tdconv_all_positive']} zones positive; spike-regime {key['tdconv_spike_positive']} zones positive; mean spike gain {key['tdconv_spike_mean_rmse_gain_pct']:.6f}%.
- Patch-attention reviewer baseline versus TDConv on spike hours: RMSE lower in {key['patch_attention_rmse_lower_than_tdconv']} zones; positive significant paired evidence in {key['patch_attention_positive_paired']} zones.
- Spike-threshold sensitivity: {key['spike_threshold_positive']} threshold settings have 4/4 zones positive; mean gains range {key['spike_threshold_mean_gain_min_pct']:.6f}% to {key['spike_threshold_mean_gain_max_pct']:.6f}%.
- Calibration-window sensitivity: best mean coverage-error window is {key['calibration_window_best_share']:.2f} with error {key['calibration_window_best_error']:.6f}; 20% window mean PICP is {key['calibration_window_20pct_mean_picp']:.6f}.
- Rolling-origin GraphPatch spike RMSE gains: {key['rolling_spike_positive']} cases positive; mean gain {key['rolling_spike_mean_rmse_gain_pct']:.6f}%.
- Leave-one-zone-out GraphPatch spike RMSE gains: {key['zone_holdout_spike_positive']} held-out zones positive; mean gain {key['zone_holdout_spike_mean_rmse_gain_pct']:.6f}%.
- Reference register: {key['doi_confirmed']} DOI-confirmed, {key['url_or_dataset']} URL/dataset entries, {key['review_required']} review-required.

## Privacy Boundary

- Local/private-data filename flags: {len(privacy_flags)}

## Manuscript Boundary

- School-recognition/graduation-language flags in bundled manuscript Markdown: {len(manuscript_flags)}

Flagged paths:

{chr(10).join('- `' + item + '`' for item in privacy_flags) if privacy_flags else '- None'}

Flagged manuscript terms:

{chr(10).join('- `' + item + '`' for item in manuscript_flags) if manuscript_flags else '- None'}

## Interpretation

This audit verifies package structure, checksums, zip readability, result-summary consistency, reference-register status, absence of local/private-data filenames in the supplement, and absence of school-recognition/graduation planning language from bundled manuscript Markdown. It does not replace a journal-specific final style check or author/institutional approval.
"""
    AUDIT.write_text(audit, encoding="utf-8")
    print(AUDIT)
    print("PASS" if ok else "REVIEW REQUIRED")


if __name__ == "__main__":
    main()
