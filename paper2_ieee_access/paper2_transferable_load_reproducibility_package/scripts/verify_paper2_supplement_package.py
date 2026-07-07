from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
SUPP_ROOT = ROOT.parent
SUPP = ROOT
ZIP_PATH = ROOT.parent / "paper2_transferable_load_reproducibility_package.zip"
AUDIT = ROOT.parent / "paper2_transferable_load_reproducibility_package_audit.md"


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


def require_close(actual: float, expected: float, tol: float = 0.01) -> bool:
    return abs(float(actual) - expected) <= tol


def verify_key_results() -> dict:
    result_dir = SUPP / "results"
    transfer = pd.read_csv(result_dir / "uci_load_transfer_summary.csv").set_index("model")
    ssl_tests = pd.read_csv(result_dir / "uci_ssl_client_level_stat_tests.csv")
    rc_summary = pd.read_csv(result_dir / "uci_random_conv_representation_summary.csv").set_index("model")
    rc_tests = pd.read_csv(result_dir / "uci_random_conv_client_level_tests.csv")
    td_summary = pd.read_csv(result_dir / "uci_trainable_tdconv_baseline_summary.csv").set_index("model")
    td_tests = pd.read_csv(result_dir / "uci_trainable_tdconv_client_level_tests.csv")
    td_multi = pd.read_csv(result_dir / "uci_tdconv_multiseed_stability_summary.csv").set_index("model")
    td_multi_tests = pd.read_csv(result_dir / "uci_tdconv_multiseed_stability_tests.csv")
    patch_summary = pd.read_csv(result_dir / "uci_patch_attention_transfer_summary.csv").set_index("model")
    patch_tests = pd.read_csv(result_dir / "uci_patch_attention_transfer_client_level_tests.csv")
    source_mlp_summary = pd.read_csv(result_dir / "uci_source_mlp_transfer_summary.csv").set_index("model")
    source_mlp_tests = pd.read_csv(result_dir / "uci_source_mlp_transfer_client_level_tests.csv")
    source_mlp_diag = pd.read_csv(result_dir / "uci_source_mlp_transfer_training_diagnostics.csv")
    neural_summary = pd.read_csv(result_dir / "uci_neural_tdconv_residual_summary.csv").set_index("model")
    neural_tests = pd.read_csv(result_dir / "uci_neural_tdconv_residual_client_level_tests.csv")
    neural_diag = pd.read_csv(result_dir / "uci_neural_tdconv_residual_training_diagnostics.csv")
    appliances = pd.read_csv(result_dir / "uci_appliances_energy_baselines.csv").set_index("model")
    appliances_mh = pd.read_csv(result_dir / "uci_appliances_energy_multihorizon_summary.csv").set_index("horizon")
    opsd = pd.read_csv(result_dir / "opsd_public_baselines.csv")
    reg = pd.read_csv(SUPP / "references" / "paper2_verified_reference_register.csv")

    def row_by_comparison(df: pd.DataFrame, text: str) -> pd.Series:
        rows = df[df["comparison"] == text]
        if rows.empty:
            raise KeyError(text)
        return rows.iloc[0]

    ssl_source = row_by_comparison(ssl_tests, "zero-label SSL source head vs 28d target ridge")
    ssl_adapter = row_by_comparison(ssl_tests, "28d SSL adapter vs 28d target ridge")
    rc_vs_mr = row_by_comparison(rc_tests, "RC adapter 28d vs MR adapter 28d")
    rc7_vs_target = row_by_comparison(rc_tests, "RC adapter 7d vs 7d target ridge")
    td_vs_rc = row_by_comparison(td_tests, "TDConv 28d adapter vs RC 28d adapter")
    td_source_vs_rc = row_by_comparison(td_tests, "TDConv source vs RC source")
    td_vs_target = row_by_comparison(td_tests, "TDConv 28d adapter vs target ridge 28d")
    td_target_head_vs_target = row_by_comparison(td_tests, "TDConv target head 28d vs target ridge 28d")
    td_multi_stability = td_multi.loc["TDConv-ridge+adapter-28d stability tests"]
    td_multi_target = td_multi_tests[td_multi_tests["baseline_model"] == "Target-linear-28d"]
    td_multi_rc = td_multi_tests[td_multi_tests["baseline_model"] == "RC-lag+adapter-28d"]
    patch_vs_td = row_by_comparison(patch_tests, "PatchAttn 28d adapter vs TDConv 28d adapter")
    patch_source_vs_td = row_by_comparison(patch_tests, "PatchAttn source vs TDConv source")
    patch_vs_rc = row_by_comparison(patch_tests, "PatchAttn 28d adapter vs RC 28d adapter")
    patch_vs_target = row_by_comparison(patch_tests, "PatchAttn 28d adapter vs target ridge 28d")
    patch_target_head_vs_target = row_by_comparison(patch_tests, "PatchAttn target head 28d vs target ridge 28d")
    source_mlp_vs_patch = row_by_comparison(source_mlp_tests, "SourceMLP 28d adapter vs patch-attention 28d")
    source_mlp_vs_td = row_by_comparison(source_mlp_tests, "SourceMLP 28d adapter vs TDConv 28d")
    source_mlp_source_vs_td = row_by_comparison(source_mlp_tests, "SourceMLP source vs TDConv source")
    source_mlp_vs_target = row_by_comparison(source_mlp_tests, "SourceMLP 28d adapter vs target ridge 28d")
    source_mlp_hidden_vs_target = row_by_comparison(source_mlp_tests, "SourceMLP hidden target head 28d vs target ridge 28d")
    source_mlp_last_diag = source_mlp_diag.iloc[-1]
    neural_vs_td = row_by_comparison(neural_tests, "Neural residual 28d vs TDConv 28d")
    neural_source_vs_td = row_by_comparison(neural_tests, "Neural residual source vs TDConv source")
    neural_vs_rc = row_by_comparison(neural_tests, "Neural residual 28d vs RC 28d")
    neural_vs_target = row_by_comparison(neural_tests, "Neural residual 28d vs target ridge 28d")
    neural7_vs_target = row_by_comparison(neural_tests, "Neural residual 7d vs target ridge 7d")
    neural_last_diag = neural_diag.iloc[-1]

    opsd_load = opsd[(opsd["task"] == "load_forecasting") & (opsd["model"] == "Linear-lag-calendar-exog")]
    opsd_best_zones = int(opsd_load["zone"].nunique())

    checks = {
        "target_28d_rmse": float(transfer.loc["Target-28d-linear", "mean_rmse"]),
        "ssl_source_rmse": float(ssl_source["mean_candidate_rmse"]),
        "ssl_source_wins": int(ssl_source["wins"]),
        "ssl_source_p": float(ssl_source["sign_test_p_two_sided"]),
        "ssl_adapter_rmse": float(ssl_adapter["mean_candidate_rmse"]),
        "ssl_adapter_wins": int(ssl_adapter["wins"]),
        "ssl_adapter_p": float(ssl_adapter["sign_test_p_two_sided"]),
        "rc_28d_rmse": float(rc_summary.loc["RC-lag+adapter-28d", "mean_rmse"]),
        "rc_vs_mr_wins": int(rc_vs_mr["wins"]),
        "rc_vs_mr_gain_pct": float(rc_vs_mr["mean_rmse_gain_pct"]),
        "rc_vs_mr_p": float(rc_vs_mr["sign_test_p_two_sided"]),
        "rc7_vs_target_wins": int(rc7_vs_target["wins"]),
        "tdconv_28d_rmse": float(td_summary.loc["TDConv-ridge+adapter-28d", "mean_rmse"]),
        "tdconv_source_rmse": float(td_summary.loc["TDConv-ridge-source-head", "mean_rmse"]),
        "tdconv_target_head_28d_rmse": float(td_summary.loc["TDConv-ridge+target-head-28d", "mean_rmse"]),
        "tdconv_vs_rc_wins": int(td_vs_rc["wins"]),
        "tdconv_vs_rc_gain_pct": float(td_vs_rc["mean_rmse_gain_pct"]),
        "tdconv_vs_rc_p": float(td_vs_rc["sign_test_p_two_sided"]),
        "tdconv_source_vs_rc_wins": int(td_source_vs_rc["wins"]),
        "tdconv_vs_target_wins": int(td_vs_target["wins"]),
        "tdconv_vs_target_gain_pct": float(td_vs_target["mean_rmse_gain_pct"]),
        "tdconv_vs_target_p": float(td_vs_target["sign_test_p_two_sided"]),
        "tdconv_target_head_vs_target_gain_pct": float(td_target_head_vs_target["mean_rmse_gain_pct"]),
        "tdconv_multiseed_rmse_mean": float(td_multi_stability["mean_rmse_mean"]),
        "tdconv_multiseed_rmse_std": float(td_multi_stability["mean_rmse_std"]),
        "tdconv_multiseed_rmse_min": float(td_multi_stability["mean_rmse_min"]),
        "tdconv_multiseed_rmse_max": float(td_multi_stability["mean_rmse_max"]),
        "tdconv_multiseed_target_min_wins": int(td_multi_target["wins"].min()),
        "tdconv_multiseed_target_max_losses": int(td_multi_target["losses"].max()),
        "tdconv_multiseed_rc_min_wins": int(td_multi_rc["wins"].min()),
        "tdconv_multiseed_rc_max_losses": int(td_multi_rc["losses"].max()),
        "patch_attention_28d_rmse": float(patch_summary.loc["PatchAttention-ridge+adapter-28d", "mean_rmse"]),
        "patch_attention_source_rmse": float(patch_summary.loc["PatchAttention-ridge-source-head", "mean_rmse"]),
        "patch_attention_target_head_28d_rmse": float(patch_summary.loc["PatchAttention-ridge+target-head-28d", "mean_rmse"]),
        "patch_attention_vs_td_wins": int(patch_vs_td["wins"]),
        "patch_attention_vs_td_gain_pct": float(patch_vs_td["mean_rmse_gain_pct"]),
        "patch_attention_vs_td_p": float(patch_vs_td["sign_test_p_two_sided"]),
        "patch_attention_source_vs_td_wins": int(patch_source_vs_td["wins"]),
        "patch_attention_vs_rc_wins": int(patch_vs_rc["wins"]),
        "patch_attention_vs_target_wins": int(patch_vs_target["wins"]),
        "patch_attention_vs_target_gain_pct": float(patch_vs_target["mean_rmse_gain_pct"]),
        "patch_attention_target_head_vs_target_gain_pct": float(patch_target_head_vs_target["mean_rmse_gain_pct"]),
        "source_mlp_28d_rmse": float(source_mlp_summary.loc["SourceMLP+adapter-28d", "mean_rmse"]),
        "source_mlp_source_rmse": float(source_mlp_summary.loc["SourceMLP-source-head", "mean_rmse"]),
        "source_mlp_hidden_28d_rmse": float(source_mlp_summary.loc["SourceMLP-hidden-target-head-28d", "mean_rmse"]),
        "source_mlp_validation_rmse": float(source_mlp_summary.loc["SourceMLP+adapter-28d", "source_validation_rmse_normalized"]),
        "source_mlp_selected_epoch": int(source_mlp_last_diag["selected_epoch"]),
        "source_mlp_vs_patch_wins": int(source_mlp_vs_patch["wins"]),
        "source_mlp_vs_patch_losses": int(source_mlp_vs_patch["losses"]),
        "source_mlp_vs_patch_gain_pct": float(source_mlp_vs_patch["mean_rmse_gain_pct"]),
        "source_mlp_vs_patch_p": float(source_mlp_vs_patch["sign_test_p_two_sided"]),
        "source_mlp_vs_td_wins": int(source_mlp_vs_td["wins"]),
        "source_mlp_vs_td_losses": int(source_mlp_vs_td["losses"]),
        "source_mlp_source_vs_td_wins": int(source_mlp_source_vs_td["wins"]),
        "source_mlp_source_vs_td_losses": int(source_mlp_source_vs_td["losses"]),
        "source_mlp_vs_target_wins": int(source_mlp_vs_target["wins"]),
        "source_mlp_vs_target_losses": int(source_mlp_vs_target["losses"]),
        "source_mlp_vs_target_p": float(source_mlp_vs_target["sign_test_p_two_sided"]),
        "source_mlp_hidden_vs_target_wins": int(source_mlp_hidden_vs_target["wins"]),
        "source_mlp_hidden_vs_target_losses": int(source_mlp_hidden_vs_target["losses"]),
        "neural_28d_rmse": float(neural_summary.loc["Neural-TDConv-residual+adapter-28d", "mean_rmse"]),
        "neural_source_rmse": float(neural_summary.loc["Neural-TDConv-residual-source-head", "mean_rmse"]),
        "neural_7d_rmse": float(neural_summary.loc["Neural-TDConv-residual+adapter-7d", "mean_rmse"]),
        "neural_residual_shrinkage": float(neural_summary.loc["Neural-TDConv-residual+adapter-28d", "residual_shrinkage"]),
        "neural_vs_td_wins": int(neural_vs_td["wins"]),
        "neural_vs_td_losses": int(neural_vs_td["losses"]),
        "neural_vs_td_p": float(neural_vs_td["sign_test_p_two_sided"]),
        "neural_source_vs_td_wins": int(neural_source_vs_td["wins"]),
        "neural_vs_rc_wins": int(neural_vs_rc["wins"]),
        "neural_vs_rc_gain_pct": float(neural_vs_rc["mean_rmse_gain_pct"]),
        "neural_vs_rc_p": float(neural_vs_rc["sign_test_p_two_sided"]),
        "neural_vs_target_wins": int(neural_vs_target["wins"]),
        "neural_vs_target_gain_pct": float(neural_vs_target["mean_rmse_gain_pct"]),
        "neural_vs_target_p": float(neural_vs_target["sign_test_p_two_sided"]),
        "neural7_vs_target_wins": int(neural7_vs_target["wins"]),
        "neural_last_train_residual_rmse": float(neural_last_diag["train_residual_rmse"]),
        "neural_last_val_residual_rmse": float(neural_last_diag["val_residual_rmse"]),
        "appliances_lag_weather_rmse": float(appliances.loc["Lag-weather ridge", "rmse_wh"]),
        "appliances_random_window_rmse": float(appliances.loc["Random-window ridge", "rmse_wh"]),
        "appliances_persistence_rmse": float(appliances.loc["Persistence-current", "rmse_wh"]),
        "appliances_seasonal_rmse": float(appliances.loc["Seasonal-24h", "rmse_wh"]),
        "appliances_mh_1h_best": str(appliances_mh.loc["1h", "best_model"]),
        "appliances_mh_1h_best_rmse": float(appliances_mh.loc["1h", "best_rmse_wh"]),
        "appliances_mh_3h_best": str(appliances_mh.loc["3h", "best_model"]),
        "appliances_mh_3h_best_rmse": float(appliances_mh.loc["3h", "best_rmse_wh"]),
        "appliances_mh_6h_best": str(appliances_mh.loc["6h", "best_model"]),
        "appliances_mh_6h_best_rmse": float(appliances_mh.loc["6h", "best_rmse_wh"]),
        "appliances_mh_12h_best": str(appliances_mh.loc["12h", "best_model"]),
        "appliances_mh_12h_best_rmse": float(appliances_mh.loc["12h", "best_rmse_wh"]),
        "appliances_mh_12h_lag_weather_gain": float(appliances_mh.loc["12h", "lag_weather_gain_vs_persistence_pct"]),
        "opsd_load_zones": opsd_best_zones,
        "doi_verified": int((reg["status"] == "doi_verified").sum()),
        "url_verified": int((reg["status"] == "url_verified_no_doi").sum()),
        "review_required": int((~reg["status"].isin(["doi_verified", "url_verified_no_doi"])).sum()),
    }
    checks["numeric_checks_pass"] = all(
        [
            require_close(checks["target_28d_rmse"], 82.784, 0.02),
            require_close(checks["ssl_source_rmse"], 75.689, 0.02),
            checks["ssl_source_wins"] == 9,
            require_close(checks["ssl_source_p"], 0.021484375, 1e-9),
            require_close(checks["ssl_adapter_rmse"], 74.962, 0.02),
            checks["ssl_adapter_wins"] == 9,
            require_close(checks["ssl_adapter_p"], 0.021484375, 1e-9),
            require_close(checks["rc_28d_rmse"], 67.500, 0.02),
            checks["rc_vs_mr_wins"] == 10,
            require_close(checks["rc_vs_mr_gain_pct"], 11.330, 0.02),
            require_close(checks["rc_vs_mr_p"], 0.001953125, 1e-9),
            checks["rc7_vs_target_wins"] == 10,
            require_close(checks["tdconv_28d_rmse"], 65.291, 0.02),
            require_close(checks["tdconv_source_rmse"], 65.471, 0.02),
            checks["tdconv_vs_rc_wins"] == 10,
            require_close(checks["tdconv_vs_rc_gain_pct"], 3.322, 0.02),
            require_close(checks["tdconv_vs_rc_p"], 0.001953125, 1e-9),
            checks["tdconv_source_vs_rc_wins"] == 10,
            checks["tdconv_vs_target_wins"] == 9,
            require_close(checks["tdconv_vs_target_gain_pct"], 23.779, 0.02),
            require_close(checks["tdconv_vs_target_p"], 0.021484375, 1e-9),
            require_close(checks["tdconv_target_head_28d_rmse"], 91.788, 0.02),
            checks["tdconv_target_head_vs_target_gain_pct"] < 0,
            require_close(checks["tdconv_multiseed_rmse_mean"], 65.338, 0.02),
            require_close(checks["tdconv_multiseed_rmse_std"], 0.056, 0.01),
            require_close(checks["tdconv_multiseed_rmse_min"], 65.283, 0.02),
            require_close(checks["tdconv_multiseed_rmse_max"], 65.465, 0.02),
            checks["tdconv_multiseed_target_min_wins"] == 9,
            checks["tdconv_multiseed_target_max_losses"] == 1,
            checks["tdconv_multiseed_rc_min_wins"] == 10,
            checks["tdconv_multiseed_rc_max_losses"] == 0,
            require_close(checks["patch_attention_28d_rmse"], 64.761, 0.02),
            require_close(checks["patch_attention_source_rmse"], 65.119, 0.02),
            checks["patch_attention_vs_td_wins"] == 9,
            require_close(checks["patch_attention_vs_td_gain_pct"], 1.900, 0.02),
            require_close(checks["patch_attention_vs_td_p"], 0.021484375, 1e-9),
            checks["patch_attention_source_vs_td_wins"] == 9,
            checks["patch_attention_vs_rc_wins"] == 9,
            checks["patch_attention_vs_target_wins"] == 9,
            require_close(checks["patch_attention_vs_target_gain_pct"], 24.941, 0.02),
            checks["patch_attention_target_head_vs_target_gain_pct"] < 0,
            require_close(checks["source_mlp_28d_rmse"], 86.056, 0.02),
            require_close(checks["source_mlp_source_rmse"], 88.212, 0.02),
            require_close(checks["source_mlp_hidden_28d_rmse"], 85.883, 0.02),
            require_close(checks["source_mlp_validation_rmse"], 0.091722, 1e-6),
            checks["source_mlp_selected_epoch"] == 32,
            checks["source_mlp_vs_patch_wins"] == 0,
            checks["source_mlp_vs_patch_losses"] == 10,
            require_close(checks["source_mlp_vs_patch_gain_pct"], -36.581, 0.02),
            require_close(checks["source_mlp_vs_patch_p"], 0.001953125, 1e-9),
            checks["source_mlp_vs_td_wins"] == 0,
            checks["source_mlp_vs_td_losses"] == 10,
            checks["source_mlp_source_vs_td_wins"] == 0,
            checks["source_mlp_source_vs_td_losses"] == 10,
            checks["source_mlp_vs_target_wins"] == 5,
            checks["source_mlp_vs_target_losses"] == 5,
            require_close(checks["source_mlp_vs_target_p"], 1.0, 1e-9),
            checks["source_mlp_hidden_vs_target_wins"] == 4,
            checks["source_mlp_hidden_vs_target_losses"] == 6,
            require_close(checks["neural_28d_rmse"], 65.582, 0.02),
            require_close(checks["neural_source_rmse"], 65.592, 0.02),
            require_close(checks["neural_7d_rmse"], 66.566, 0.02),
            require_close(checks["neural_residual_shrinkage"], 0.25, 1e-9),
            checks["neural_vs_td_wins"] == 5,
            checks["neural_vs_td_losses"] == 5,
            require_close(checks["neural_vs_td_p"], 1.0, 1e-9),
            checks["neural_source_vs_td_wins"] == 5,
            checks["neural_vs_rc_wins"] == 10,
            require_close(checks["neural_vs_rc_gain_pct"], 2.977, 0.02),
            require_close(checks["neural_vs_rc_p"], 0.001953125, 1e-9),
            checks["neural_vs_target_wins"] == 9,
            require_close(checks["neural_vs_target_gain_pct"], 23.471, 0.02),
            require_close(checks["neural_vs_target_p"], 0.021484375, 1e-9),
            checks["neural7_vs_target_wins"] == 10,
            require_close(checks["appliances_lag_weather_rmse"], 78.589, 0.02),
            require_close(checks["appliances_random_window_rmse"], 78.612, 0.02),
            checks["appliances_lag_weather_rmse"] < checks["appliances_persistence_rmse"],
            checks["appliances_lag_weather_rmse"] < checks["appliances_seasonal_rmse"],
            checks["appliances_mh_1h_best"] == "Lag-weather ridge",
            require_close(checks["appliances_mh_1h_best_rmse"], 78.596, 0.02),
            checks["appliances_mh_3h_best"] == "Random-window ridge",
            require_close(checks["appliances_mh_3h_best_rmse"], 83.544, 0.02),
            checks["appliances_mh_6h_best"] == "Random-window ridge",
            require_close(checks["appliances_mh_6h_best_rmse"], 85.445, 0.02),
            checks["appliances_mh_12h_best"] == "Random-window ridge",
            require_close(checks["appliances_mh_12h_best_rmse"], 84.900, 0.02),
            checks["appliances_mh_12h_lag_weather_gain"] > 30.0,
            checks["opsd_load_zones"] == 4,
        ]
    )
    return checks


def verify_privacy_boundary() -> tuple[list[str], list[str]]:
    forbidden_path_terms = ["hunan", "shandong", "daily_disclosure", "node_price", "user_load"]
    content_terms = ["hunan", "shandong", "daily_disclosure", "node_price", "user_load"]
    flagged_paths = []
    flagged_content = []
    for path in SUPP.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(SUPP))
        lower = rel.lower()
        if any(term in lower for term in forbidden_path_terms):
            flagged_paths.append(rel)
        if path.suffix.lower() in {".md", ".txt", ".csv", ".py", ".sh"}:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if any(term in text for term in content_terms):
                flagged_content.append(rel)
    return flagged_paths, flagged_content


def main() -> None:
    rows, missing, mismatched = verify_manifest()
    zip_members, zip_uncompressed, zip_compressed, bad_zip_member = verify_zip()
    key = verify_key_results()
    privacy_path_flags, privacy_content_mentions = verify_privacy_boundary()
    ok = (
        not missing
        and not mismatched
        and not bad_zip_member
        and not privacy_path_flags
        and key["review_required"] == 0
        and key["numeric_checks_pass"]
    )
    audit = f"""# Paper 2 Supplement Package Audit

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

- UCI 28-day target-only ridge RMSE: {key['target_28d_rmse']:.6f}.
- Zero-label SSL source-head RMSE/wins/p: {key['ssl_source_rmse']:.6f} / {key['ssl_source_wins']} / {key['ssl_source_p']:.9f}.
- 28-day SSL adapter RMSE/wins/p: {key['ssl_adapter_rmse']:.6f} / {key['ssl_adapter_wins']} / {key['ssl_adapter_p']:.9f}.
- Random-convolution 28-day adapter RMSE: {key['rc_28d_rmse']:.6f}.
- Random-convolution 28-day adapter vs MR 28-day adapter wins/gain/p: {key['rc_vs_mr_wins']} / {key['rc_vs_mr_gain_pct']:.6f}% / {key['rc_vs_mr_p']:.9f}.
- Random-convolution 7-day adapter vs 7-day target ridge wins: {key['rc7_vs_target_wins']}.
- Trainable dilated-convolution 28-day adapter RMSE: {key['tdconv_28d_rmse']:.6f}.
- Trainable dilated-convolution source-head RMSE: {key['tdconv_source_rmse']:.6f}.
- Trainable dilated-convolution 28-day adapter vs random-convolution 28-day adapter wins/gain/p: {key['tdconv_vs_rc_wins']} / {key['tdconv_vs_rc_gain_pct']:.6f}% / {key['tdconv_vs_rc_p']:.9f}.
- Trainable dilated-convolution source-head vs random-convolution source-head wins: {key['tdconv_source_vs_rc_wins']}.
- Trainable dilated-convolution 28-day adapter vs 28-day target ridge wins/gain/p: {key['tdconv_vs_target_wins']} / {key['tdconv_vs_target_gain_pct']:.6f}% / {key['tdconv_vs_target_p']:.9f}.
- Target-only trainable dilated-convolution 28-day head RMSE/gain vs target ridge: {key['tdconv_target_head_28d_rmse']:.6f} / {key['tdconv_target_head_vs_target_gain_pct']:.6f}%.
- Multi-seed TDConv 28-day adapter RMSE mean/std/min/max: {key['tdconv_multiseed_rmse_mean']:.6f} / {key['tdconv_multiseed_rmse_std']:.6f} / {key['tdconv_multiseed_rmse_min']:.6f} / {key['tdconv_multiseed_rmse_max']:.6f}.
- Multi-seed TDConv minimum wins vs target ridge and random-convolution adapter: {key['tdconv_multiseed_target_min_wins']} / {key['tdconv_multiseed_rc_min_wins']}.
- CPU-only patch-attention 28-day adapter/source-head RMSE: {key['patch_attention_28d_rmse']:.6f} / {key['patch_attention_source_rmse']:.6f}.
- CPU-only patch-attention 28-day adapter vs TDConv 28-day adapter wins/gain/p: {key['patch_attention_vs_td_wins']} / {key['patch_attention_vs_td_gain_pct']:.6f}% / {key['patch_attention_vs_td_p']:.9f}.
- CPU-only patch-attention 28-day adapter vs random-convolution and target ridge wins: {key['patch_attention_vs_rc_wins']} / {key['patch_attention_vs_target_wins']}.
- Target-only patch-attention 28-day head RMSE/gain vs target ridge: {key['patch_attention_target_head_28d_rmse']:.6f} / {key['patch_attention_target_head_vs_target_gain_pct']:.6f}%.
- Source-trained MLP 28-day adapter/source-head/hidden-head RMSE: {key['source_mlp_28d_rmse']:.6f} / {key['source_mlp_source_rmse']:.6f} / {key['source_mlp_hidden_28d_rmse']:.6f}.
- Source-trained MLP validation RMSE and selected epoch: {key['source_mlp_validation_rmse']:.6f} / {key['source_mlp_selected_epoch']}.
- Source-trained MLP 28-day adapter vs patch-attention wins-losses/gain/p: {key['source_mlp_vs_patch_wins']}-{key['source_mlp_vs_patch_losses']} / {key['source_mlp_vs_patch_gain_pct']:.6f}% / {key['source_mlp_vs_patch_p']:.9f}.
- Source-trained MLP 28-day adapter vs target ridge wins-losses/p: {key['source_mlp_vs_target_wins']}-{key['source_mlp_vs_target_losses']} / {key['source_mlp_vs_target_p']:.9f}.
- Neural TDConv residual 28-day adapter RMSE / shrinkage: {key['neural_28d_rmse']:.6f} / {key['neural_residual_shrinkage']:.2f}.
- Neural TDConv residual source-head / 7-day adapter RMSE: {key['neural_source_rmse']:.6f} / {key['neural_7d_rmse']:.6f}.
- Neural TDConv residual 28-day adapter vs trainable TDConv 28-day adapter wins-losses/p: {key['neural_vs_td_wins']}-{key['neural_vs_td_losses']} / {key['neural_vs_td_p']:.9f}.
- Neural TDConv residual 28-day adapter vs random-convolution 28-day adapter wins/gain/p: {key['neural_vs_rc_wins']} / {key['neural_vs_rc_gain_pct']:.6f}% / {key['neural_vs_rc_p']:.9f}.
- Neural TDConv residual 28-day adapter vs 28-day target ridge wins/gain/p: {key['neural_vs_target_wins']} / {key['neural_vs_target_gain_pct']:.6f}% / {key['neural_vs_target_p']:.9f}.
- Neural TDConv residual 7-day adapter vs 7-day target ridge wins: {key['neural7_vs_target_wins']}.
- Neural residual final train/validation residual RMSE: {key['neural_last_train_residual_rmse']:.6f} / {key['neural_last_val_residual_rmse']:.6f}.
- UCI Appliances lag-weather / random-window / persistence / seasonal RMSE: {key['appliances_lag_weather_rmse']:.6f} / {key['appliances_random_window_rmse']:.6f} / {key['appliances_persistence_rmse']:.6f} / {key['appliances_seasonal_rmse']:.6f}.
- UCI Appliances multi-horizon best models and RMSE: 1h {key['appliances_mh_1h_best']} {key['appliances_mh_1h_best_rmse']:.6f}; 3h {key['appliances_mh_3h_best']} {key['appliances_mh_3h_best_rmse']:.6f}; 6h {key['appliances_mh_6h_best']} {key['appliances_mh_6h_best_rmse']:.6f}; 12h {key['appliances_mh_12h_best']} {key['appliances_mh_12h_best_rmse']:.6f}.
- OPSD public load baseline zones represented: {key['opsd_load_zones']}.
- Numeric/result checks pass: {key['numeric_checks_pass']}.
- Reference register: {key['doi_verified']} DOI-verified, {key['url_verified']} URL entries, {key['review_required']} review-required.

## Privacy Boundary

- Local/private-data filename flags: {len(privacy_path_flags)}
- Local/private-data content mentions in text-like files: {len(privacy_content_mentions)}

Flagged paths:

{chr(10).join('- `' + item + '`' for item in privacy_path_flags) if privacy_path_flags else '- None'}

Text files mentioning local-data terms:

{chr(10).join('- `' + item + '`' for item in privacy_content_mentions) if privacy_content_mentions else '- None'}

## Interpretation

This audit verifies package structure, checksums, zip readability, key public result consistency, reference-register status, and absence of local/private-data filenames in bundled paths. Text mentions of local application boundaries are reported for human review but do not fail the package unless raw/private filenames are bundled as file paths.
"""
    AUDIT.write_text(audit, encoding="utf-8")
    print(AUDIT)
    print("PASS" if ok else "REVIEW REQUIRED")


if __name__ == "__main__":
    main()
