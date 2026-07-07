from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_uci_ssl_representation_prototype import (
    HISTORY,
    RESULTS,
    FIGURES,
    TEST_START,
    SOURCE_TRAIN_START,
    build_scales,
    calendar_features,
    client_supervised_arrays,
    fit_adapter,
    fit_ridge,
    load_data,
    metrics,
    predict_adapter,
    predict_ridge,
)


RESULTS_OUT = RESULTS / "uci_patch_attention_transfer_results.csv"
SUMMARY_OUT = RESULTS / "uci_patch_attention_transfer_summary.csv"
TESTS_OUT = RESULTS / "uci_patch_attention_transfer_client_level_tests.csv"
FIGURE_OUT = FIGURES / "paper2_fig16_uci_patch_attention_transfer_baseline.png"

PATCH_HOURS = 24
PATCH_COUNT = HISTORY // PATCH_HOURS
ADAPT_DAYS = [7, 28]
MAX_SOURCE_WINDOWS = 70000
RIDGE_L2 = 5e-2
RNG_SEED = 20260706


def font(size: int = 28, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def exact_two_sided_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def patch_attention_features(windows: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    patches = windows.reshape(len(windows), PATCH_COUNT, PATCH_HOURS)
    query = patches[:, -1:, :]

    centered_patches = patches - patches.mean(axis=2, keepdims=True)
    centered_query = query - query.mean(axis=2, keepdims=True)
    denom = np.linalg.norm(centered_patches, axis=2) * np.linalg.norm(centered_query, axis=2)
    denom[denom < 1e-6] = 1.0
    similarity = (centered_patches * centered_query).sum(axis=2) / denom
    weights = softmax(2.0 * similarity, axis=1).astype(np.float32)
    attended_patch = (patches * weights[:, :, None]).sum(axis=1)

    patch_stats = np.column_stack(
        [
            patches.mean(axis=2),
            patches.std(axis=2),
            patches[:, :, -1],
            weights,
            weights.max(axis=1),
            weights[:, -1],
            attended_patch.mean(axis=1),
            attended_patch.std(axis=1),
        ]
    )
    lag_features = np.column_stack(
        [
            windows[:, -1],
            windows[:, -24],
            windows[:, -168],
            windows[:, -24:].mean(axis=1),
            windows[:, -24:].std(axis=1),
            windows[:, -168:].mean(axis=1),
            windows[:, -168:].std(axis=1),
        ]
    )
    return np.column_stack(
        [attended_patch, patch_stats, lag_features, calendar_features(times)]
    ).astype(np.float32)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def collect_source_training(hourly, source_clients, scales):
    windows_all = []
    y_all = []
    times_all = []
    for client in source_clients:
        windows, y, times = client_supervised_arrays(hourly, client, scales[client])
        mask = (times >= SOURCE_TRAIN_START) & (times < TEST_START)
        if mask.any():
            windows_all.append(windows[mask])
            y_all.append(y[mask])
            times_all.append(times[mask])
    windows = np.vstack(windows_all)
    y = np.concatenate(y_all)
    times = times_all[0].append(times_all[1:]) if len(times_all) > 1 else times_all[0]
    if len(windows) > MAX_SOURCE_WINDOWS:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(len(windows), size=MAX_SOURCE_WINDOWS, replace=False)
        windows = windows[idx]
        y = y[idx]
        times = times[idx]
    x = patch_attention_features(windows, times)
    mean, std = standardize_fit(x)
    return standardize_apply(x, mean, std), y, mean, std


def evaluate_target_patch_attention(windows, y_norm, times, test_mask, days, feat_mean, feat_std):
    start = TEST_START - pd.Timedelta(days=days)
    adapt_mask = (times >= start) & (times < TEST_START)
    if adapt_mask.sum() < max(12, int(24 * days * 0.6)):
        return None
    x_adapt = standardize_apply(patch_attention_features(windows[adapt_mask], times[adapt_mask]), feat_mean, feat_std)
    x_test = standardize_apply(patch_attention_features(windows[test_mask], times[test_mask]), feat_mean, feat_std)
    beta = fit_ridge(x_adapt, y_norm[adapt_mask], l2=RIDGE_L2)
    return predict_ridge(x_test, beta)


def evaluate(hourly, source_clients, target_clients, scales):
    source_x, source_y, feat_mean, feat_std = collect_source_training(hourly, source_clients, scales)
    source_beta = fit_ridge(source_x, source_y, l2=RIDGE_L2)
    rows = []
    for client in target_clients:
        windows, y_norm, times = client_supervised_arrays(hourly, client, scales[client])
        test_mask = times >= TEST_START
        if test_mask.sum() < 24 * 30:
            continue
        scale = scales[client]
        all_x = standardize_apply(patch_attention_features(windows, times), feat_mean, feat_std)
        source_pred_all = predict_ridge(all_x, source_beta)
        source_pred = source_pred_all[test_mask]
        test_y = y_norm[test_mask]
        test_times = times[test_mask]

        rows.append(
            {
                "dataset": "UCI Electricity Load Diagrams 2011-2014",
                "target_client": client,
                "model": "PatchAttention-ridge-source-head",
                "protocol": "CPU-only patch-attention source representation",
                "adapt_days": 0,
                **metrics(test_y * scale, source_pred * scale),
            }
        )

        for days in ADAPT_DAYS:
            start = TEST_START - pd.Timedelta(days=days)
            adapt_mask = (times >= start) & (times < TEST_START)
            if adapt_mask.sum() < max(12, int(24 * days * 0.6)):
                continue
            adapter_beta = fit_adapter(source_pred_all[adapt_mask], y_norm[adapt_mask], times[adapt_mask])
            adapter_pred = predict_adapter(source_pred, test_times, adapter_beta)
            rows.append(
                {
                    "dataset": "UCI Electricity Load Diagrams 2011-2014",
                    "target_client": client,
                    "model": f"PatchAttention-ridge+adapter-{days}d",
                    "protocol": "CPU-only patch-attention source representation with target adapter",
                    "adapt_days": days,
                    **metrics(test_y * scale, adapter_pred * scale),
                }
            )

            target_pred = evaluate_target_patch_attention(windows, y_norm, times, test_mask, days, feat_mean, feat_std)
            if target_pred is not None:
                rows.append(
                    {
                        "dataset": "UCI Electricity Load Diagrams 2011-2014",
                        "target_client": client,
                        "model": f"PatchAttention-ridge+target-head-{days}d",
                        "protocol": "target-only CPU-only patch-attention ridge",
                        "adapt_days": days,
                        **metrics(test_y * scale, target_pred * scale),
                    }
                )

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_OUT, index=False)
    summary = (
        result.groupby(["model", "protocol", "adapt_days"], as_index=False)
        .agg(
            target_clients=("target_client", "nunique"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_smape=("smape", "mean"),
            total_n=("n", "sum"),
        )
        .sort_values(["mean_rmse", "adapt_days", "model"])
    )
    summary.to_csv(SUMMARY_OUT, index=False)
    return result, summary


def paired_stats(result: pd.DataFrame) -> pd.DataFrame:
    td = pd.read_csv(RESULTS / "uci_trainable_tdconv_baseline_results.csv")
    rc = pd.read_csv(RESULTS / "uci_random_conv_representation_results.csv")
    base = pd.read_csv(RESULTS / "uci_ssl_cold_start_results.csv")
    joined = (
        pd.concat([result, td, rc, base], ignore_index=True)
        .sort_values(["model", "target_client"])
        .drop_duplicates(["model", "target_client"], keep="first")
    )
    specs = [
        ("PatchAttn 28d adapter vs TDConv 28d adapter", "PatchAttention-ridge+adapter-28d", "TDConv-ridge+adapter-28d"),
        ("PatchAttn source vs TDConv source", "PatchAttention-ridge-source-head", "TDConv-ridge-source-head"),
        ("PatchAttn 28d adapter vs RC 28d adapter", "PatchAttention-ridge+adapter-28d", "RC-lag+adapter-28d"),
        ("PatchAttn 28d adapter vs target ridge 28d", "PatchAttention-ridge+adapter-28d", "Target-linear-28d"),
        ("PatchAttn target head 28d vs target ridge 28d", "PatchAttention-ridge+target-head-28d", "Target-linear-28d"),
    ]
    rows = []
    for comparison, candidate, baseline in specs:
        cand = joined[joined["model"] == candidate][["target_client", "rmse", "mae"]].rename(
            columns={"rmse": "candidate_rmse", "mae": "candidate_mae"}
        )
        comp = joined[joined["model"] == baseline][["target_client", "rmse", "mae"]].rename(
            columns={"rmse": "baseline_rmse", "mae": "baseline_mae"}
        )
        merged = comp.merge(cand, on="target_client", how="inner")
        if merged.empty:
            continue
        merged["rmse_delta"] = merged["baseline_rmse"] - merged["candidate_rmse"]
        wins = int((merged["rmse_delta"] > 1e-12).sum())
        losses = int((merged["rmse_delta"] < -1e-12).sum())
        ties = int(len(merged) - wins - losses)
        rows.append(
            {
                "comparison": comparison,
                "candidate_model": candidate,
                "baseline_model": baseline,
                "target_clients": int(len(merged)),
                "mean_baseline_rmse": float(merged["baseline_rmse"].mean()),
                "mean_candidate_rmse": float(merged["candidate_rmse"].mean()),
                "mean_rmse_gain_pct": float((merged["rmse_delta"] / merged["baseline_rmse"] * 100).mean()),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "sign_test_p_two_sided": exact_two_sided_sign_p(wins, losses),
            }
        )
    tests = pd.DataFrame(rows)
    tests.to_csv(TESTS_OUT, index=False)
    return tests


def plot_summary(summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    td_summary = pd.read_csv(RESULTS / "uci_trainable_tdconv_baseline_summary.csv")
    rc_summary = pd.read_csv(RESULTS / "uci_random_conv_representation_summary.csv")
    ssl_summary = pd.read_csv(RESULTS / "uci_ssl_cold_start_summary.csv")
    keep_models = [
        "PatchAttention-ridge+adapter-28d",
        "PatchAttention-ridge-source-head",
        "PatchAttention-ridge+target-head-28d",
        "TDConv-ridge+adapter-28d",
        "TDConv-ridge-source-head",
        "RC-lag+adapter-28d",
        "Target-linear-28d",
    ]
    combined = pd.concat([summary, td_summary, rc_summary, ssl_summary], ignore_index=True)
    combined = combined[combined["model"].isin(keep_models)].drop_duplicates("model", keep="first")
    combined = combined.sort_values("mean_rmse")

    width, height = 1960, 1120
    left, top, right, bottom = 720, 165, 1740, 850
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((86, 52), "CPU-only patch-attention check on UCI load transfer", fill="#172033", font=font(40, True))
    d.text((90, 106), "Mean RMSE across ten target clients; lower is better", fill="#5f6b7a", font=font(24))

    max_v = float(combined["mean_rmse"].max()) * 1.10
    for i in range(6):
        x = left + i * (right - left) / 5
        d.line((x, top, x, bottom), fill="#D7DCE2", width=1)
        label = f"{max_v * i / 5:.0f}"
        d.text((x - d.textlength(label, font=font(20)) / 2, bottom + 18), label, fill="#526070", font=font(20))

    row_h = (bottom - top) / len(combined)
    for idx, row in enumerate(combined.itertuples(index=False)):
        y = top + idx * row_h + row_h * 0.18
        bar_h = row_h * 0.58
        model = row.model
        if model.startswith("PatchAttention"):
            color = "#2F6F73"
        elif model.startswith("TDConv"):
            color = "#4C78A8"
        elif model.startswith("RC"):
            color = "#7F3C8D"
        elif model.startswith("Target"):
            color = "#F58518"
        else:
            color = "#54A24B"
        x1 = left + float(row.mean_rmse) / max_v * (right - left)
        d.rounded_rectangle((left, y, x1, y + bar_h), radius=8, fill=color)
        d.text((90, y + 4), model, fill="#1f2937", font=font(21, True if idx == 0 else False))
        d.text((x1 + 12, y + 4), f"{float(row.mean_rmse):.2f}", fill="#1f2937", font=font(21, True))

    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    stat = tests[tests["comparison"] == "PatchAttn 28d adapter vs TDConv 28d adapter"]
    if not stat.empty:
        s = stat.iloc[0]
        note = (
            f"Patch-attention 28d vs TDConv 28d: {int(s.wins)}/{int(s.target_clients)} wins, "
            f"mean gain {float(s.mean_rmse_gain_pct):.2f}%, p={float(s.sign_test_p_two_sided):.3f}"
        )
    else:
        note = "Patch-attention is a CPU-only reviewer-response baseline, not full PatchTST/TFT."
    d.text((90, 942), note, fill="#374151", font=font(23, True))
    d.text(
        (90, 992),
        "The baseline pools seven 24-hour patches from the 168-hour history window with deterministic similarity weights.",
        fill="#6b7280",
        font=font(19),
    )
    img.save(FIGURE_OUT)


def main() -> None:
    hourly, source_clients, target_clients = load_data()
    scales = build_scales(hourly, source_clients, target_clients)
    result, summary = evaluate(hourly, source_clients, target_clients, scales)
    tests = paired_stats(result)
    plot_summary(summary, tests)
    print(RESULTS_OUT)
    print(SUMMARY_OUT)
    print(TESTS_OUT)
    print(FIGURE_OUT)
    print(summary.to_string(index=False))
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
