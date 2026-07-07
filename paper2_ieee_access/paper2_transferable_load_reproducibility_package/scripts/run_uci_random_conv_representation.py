from __future__ import annotations

from math import comb
from pathlib import Path

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


ADAPT_DAYS = [1, 3, 7, 28]
N_KERNELS = 96
RNG_SEED = 20260630
MAX_SOURCE_WINDOWS = 80000
RIDGE_L2 = 1e-2

RESULTS_OUT = RESULTS / "uci_random_conv_representation_results.csv"
SUMMARY_OUT = RESULTS / "uci_random_conv_representation_summary.csv"
TESTS_OUT = RESULTS / "uci_random_conv_client_level_tests.csv"
FIGURE_OUT = FIGURES / "paper2_fig9_uci_random_conv_encoder_comparison.png"


def font(size=28, bold=False):
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


def make_kernels():
    rng = np.random.default_rng(RNG_SEED)
    lengths = [7, 9, 12, 24, 48]
    kernels = []
    for _ in range(N_KERNELS):
        length = int(rng.choice(lengths))
        valid_dilations = [d for d in [1, 2, 3, 4, 6, 8] if (length - 1) * d < HISTORY]
        dilation = int(rng.choice(valid_dilations))
        weights = rng.normal(size=length).astype(np.float32)
        weights = weights - weights.mean()
        norm = float(np.linalg.norm(weights))
        if norm < 1e-8:
            weights[0] = 1.0
            norm = 1.0
        weights = weights / norm
        bias = float(rng.uniform(-0.25, 0.25))
        kernels.append({"length": length, "dilation": dilation, "weights": weights, "bias": bias})
    return kernels


def transform_random_conv(windows: np.ndarray, kernels: list[dict], batch_size: int = 2048) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    features = np.empty((len(windows), len(kernels) * 3), dtype=np.float32)
    for start in range(0, len(windows), batch_size):
        end = min(start + batch_size, len(windows))
        batch = windows[start:end]
        out = np.empty((len(batch), len(kernels) * 3), dtype=np.float32)
        for k_idx, kernel in enumerate(kernels):
            length = kernel["length"]
            dilation = kernel["dilation"]
            weights = kernel["weights"]
            bias = kernel["bias"]
            max_start = HISTORY - (length - 1) * dilation
            positions = np.arange(max_start)[:, None] + np.arange(length)[None, :] * dilation
            segments = batch[:, positions]
            conv = np.tensordot(segments, weights, axes=([2], [0])) + bias
            base = k_idx * 3
            out[:, base] = conv.max(axis=1)
            out[:, base + 1] = (conv > 0).mean(axis=1)
            out[:, base + 2] = conv[:, -1]
        features[start:end] = out
    return features


def lag_calendar_features(windows, times):
    lag_features = np.column_stack([windows[:, -1], windows[:, -24], windows[:, 0]])
    return np.column_stack([lag_features, calendar_features(times)])


def feature_matrix(windows, times, kernels):
    rc = transform_random_conv(windows, kernels)
    return np.column_stack([rc, lag_calendar_features(windows, times)]).astype(np.float32)


def standardize_fit(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x, mean, std):
    return (x - mean) / std


def collect_source_training(hourly, source_clients, scales, kernels):
    xs = []
    ys = []
    for client in source_clients:
        windows, y, times = client_supervised_arrays(hourly, client, scales[client])
        mask = (times >= SOURCE_TRAIN_START) & (times < TEST_START)
        if mask.any():
            xs.append(feature_matrix(windows[mask], times[mask], kernels))
            ys.append(y[mask])
    x = np.vstack(xs)
    y = np.concatenate(ys)
    if len(x) > MAX_SOURCE_WINDOWS:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(len(x), size=MAX_SOURCE_WINDOWS, replace=False)
        x = x[idx]
        y = y[idx]
    mean, std = standardize_fit(x)
    return standardize_apply(x, mean, std), y, mean, std


def evaluate_target_linear(windows, y_norm, times, test_mask, days):
    start = TEST_START - pd.Timedelta(days=days)
    adapt_mask = (times >= start) & (times < TEST_START)
    if adapt_mask.sum() < max(12, int(24 * days * 0.6)):
        return None
    beta = fit_ridge(lag_calendar_features(windows[adapt_mask], times[adapt_mask]), y_norm[adapt_mask], l2=1e-3)
    return predict_ridge(lag_calendar_features(windows[test_mask], times[test_mask]), beta)


def evaluate(hourly, source_clients, target_clients, scales, kernels):
    source_x, source_y, feat_mean, feat_std = collect_source_training(hourly, source_clients, scales, kernels)
    source_beta = fit_ridge(source_x, source_y, l2=RIDGE_L2)
    rows = []
    for client in target_clients:
        windows, y_norm, times = client_supervised_arrays(hourly, client, scales[client])
        test_mask = times >= TEST_START
        if test_mask.sum() < 24 * 30:
            continue
        scale = scales[client]
        all_x = standardize_apply(feature_matrix(windows, times, kernels), feat_mean, feat_std)
        source_pred_all = predict_ridge(all_x, source_beta)
        source_pred = source_pred_all[test_mask]
        test_y = y_norm[test_mask]
        test_times = times[test_mask]
        test_windows = windows[test_mask]

        for model, pred_norm, protocol, adapt_days in [
            ("RC-lag-source-head", source_pred, "random-convolution source representation", 0),
            ("Seasonal-24h", test_windows[:, -24], "zero-label seasonal", 0),
            ("Seasonal-168h", test_windows[:, 0], "zero-label seasonal", 0),
        ]:
            rows.append(
                {
                    "dataset": "UCI Electricity Load Diagrams 2011-2014",
                    "target_client": client,
                    "model": model,
                    "protocol": protocol,
                    "adapt_days": adapt_days,
                    **metrics(test_y * scale, pred_norm * scale),
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
                    "model": f"RC-lag+adapter-{days}d",
                    "protocol": "random-convolution representation with target adapter",
                    "adapt_days": days,
                    **metrics(test_y * scale, adapter_pred * scale),
                }
            )

            target_pred = evaluate_target_linear(windows, y_norm, times, test_mask, days)
            if target_pred is not None:
                rows.append(
                    {
                        "dataset": "UCI Electricity Load Diagrams 2011-2014",
                        "target_client": client,
                        "model": f"Target-linear-{days}d",
                        "protocol": "target-only lag-calendar ridge",
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


def paired_stats(result):
    specs = [
        ("RC source vs 28d target ridge", "RC-lag-source-head", "Target-linear-28d"),
        ("RC adapter 28d vs 28d target ridge", "RC-lag+adapter-28d", "Target-linear-28d"),
        ("RC adapter 7d vs 7d target ridge", "RC-lag+adapter-7d", "Target-linear-7d"),
        ("RC source vs MR source head", "RC-lag-source-head", "SSL-MR-lag-source-head"),
        ("RC adapter 28d vs MR adapter 28d", "RC-lag+adapter-28d", "SSL-MR-lag+adapter-28d"),
    ]
    mr = pd.read_csv(RESULTS / "uci_ssl_cold_start_results.csv")
    joined_frame = (
        pd.concat([result, mr], ignore_index=True)
        .sort_values(["model", "target_client"])
        .drop_duplicates(["model", "target_client"], keep="first")
    )
    rows = []
    for comparison, candidate, baseline in specs:
        cand = joined_frame[joined_frame["model"] == candidate][["target_client", "rmse", "mae"]].rename(
            columns={"rmse": "candidate_rmse", "mae": "candidate_mae"}
        )
        base = joined_frame[joined_frame["model"] == baseline][["target_client", "rmse", "mae"]].rename(
            columns={"rmse": "baseline_rmse", "mae": "baseline_mae"}
        )
        merged = base.merge(cand, on="target_client", how="inner")
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


def plot_summary(summary, tests):
    baseline = pd.read_csv(RESULTS / "uci_ssl_cold_start_summary.csv")
    keep_models = [
        "RC-lag+adapter-28d",
        "RC-lag-source-head",
        "SSL-MR-lag+adapter-28d",
        "SSL-MR-lag-source-head",
        "Target-linear-28d",
        "Target-linear-7d",
        "Seasonal-168h",
        "Seasonal-24h",
    ]
    combined = pd.concat([summary, baseline], ignore_index=True)
    combined = combined[combined["model"].isin(keep_models)].drop_duplicates("model", keep="first")
    combined = combined.sort_values("mean_rmse")

    width, height = 1900, 1160
    left, top, right, bottom = 660, 170, 1710, 900
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((86, 54), "Random-convolution representation check on UCI load transfer", fill="#172033", font=font(40, True))
    d.text((90, 108), "Mean RMSE across ten target clients; lower is better", fill="#5f6b7a", font=font(24))

    max_v = float(combined["mean_rmse"].max()) * 1.10
    for i in range(6):
        x = left + i * (right - left) / 5
        d.line((x, top, x, bottom), fill="#D7DCE2", width=1)
        label = f"{max_v * i / 5:.0f}"
        d.text((x - d.textlength(label, font=font(20)) / 2, bottom + 18), label, fill="#526070", font=font(20))

    colors = {
        "random-convolution": "#7F3C8D",
        "masked-reconstruction": "#E45756",
        "target": "#4C78A8",
        "seasonal": "#54A24B",
    }
    row_h = (bottom - top) / len(combined)
    for idx, row in enumerate(combined.itertuples(index=False)):
        y = top + idx * row_h + row_h * 0.18
        bar_h = row_h * 0.58
        model = row.model
        if model.startswith("RC"):
            color = colors["random-convolution"]
        elif model.startswith("SSL-MR"):
            color = colors["masked-reconstruction"]
        elif model.startswith("Target"):
            color = colors["target"]
        else:
            color = colors["seasonal"]
        x1 = left + float(row.mean_rmse) / max_v * (right - left)
        d.rounded_rectangle((left, y, x1, y + bar_h), radius=8, fill=color)
        d.text((90, y + 4), model, fill="#1f2937", font=font(23, True if idx == 0 else False))
        d.text((x1 + 12, y + 4), f"{float(row.mean_rmse):.2f}", fill="#1f2937", font=font(22, True))

    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    stat = tests[tests["comparison"] == "RC adapter 28d vs MR adapter 28d"]
    if not stat.empty:
        s = stat.iloc[0]
        note = (
            f"RC 28d adapter vs MR 28d adapter: {int(s.wins)}/{int(s.target_clients)} wins, "
            f"mean gain {float(s.mean_rmse_gain_pct):.2f}%, p={float(s.sign_test_p_two_sided):.3f}"
        )
    else:
        note = "Random-convolution representation is evaluated as a stronger encoder-style robustness check."
    d.text((90, 985), note, fill="#374151", font=font(23, True))
    d.text((90, 1032), "Source: UCI Electricity Load Diagrams 2011-2014; source clients train the representation head, target clients test transfer.", fill="#6b7280", font=font(20))
    img.save(FIGURE_OUT)


def main():
    hourly, source_clients, target_clients = load_data()
    scales = build_scales(hourly, source_clients, target_clients)
    kernels = make_kernels()
    result, summary = evaluate(hourly, source_clients, target_clients, scales, kernels)
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
