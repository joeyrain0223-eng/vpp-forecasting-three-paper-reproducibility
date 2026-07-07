from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_uci_ssl_representation_prototype import (
    RESULTS,
    FIGURES,
    TEST_START,
    SOURCE_TRAIN_START,
    build_scales,
    client_supervised_arrays,
    fit_adapter,
    fit_ridge,
    load_data,
    metrics,
    predict_adapter,
    predict_ridge,
)
from run_uci_trainable_tdconv_baseline import (
    MAX_SOURCE_WINDOWS,
    RIDGE_L2,
    tdconv_features,
    standardize_fit,
    standardize_apply,
)


SEEDS = [20260710, 20260711, 20260712, 20260713, 20260714, 20260715, 20260716, 20260717]
RESULTS_OUT = RESULTS / "uci_tdconv_multiseed_stability_results.csv"
BY_SEED_OUT = RESULTS / "uci_tdconv_multiseed_stability_by_seed.csv"
SUMMARY_OUT = RESULTS / "uci_tdconv_multiseed_stability_summary.csv"
TESTS_OUT = RESULTS / "uci_tdconv_multiseed_stability_tests.csv"
FIGURE_OUT = FIGURES / "paper2_fig15_uci_tdconv_multiseed_stability.png"


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


def collect_source_training_seed(hourly, source_clients, scales, seed: int):
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
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(windows), size=MAX_SOURCE_WINDOWS, replace=False)
        windows = windows[idx]
        y = y[idx]
        times = times[idx]
    x = tdconv_features(windows, times)
    mean, std = standardize_fit(x)
    return standardize_apply(x, mean, std), y, mean, std


def evaluate_seed(hourly, source_clients, target_clients, scales, seed: int) -> pd.DataFrame:
    source_x, source_y, feat_mean, feat_std = collect_source_training_seed(hourly, source_clients, scales, seed)
    source_beta = fit_ridge(source_x, source_y, l2=RIDGE_L2)
    rows = []
    for client in target_clients:
        windows, y_norm, times = client_supervised_arrays(hourly, client, scales[client])
        test_mask = times >= TEST_START
        if test_mask.sum() < 24 * 30:
            continue
        scale = scales[client]
        all_x = standardize_apply(tdconv_features(windows, times), feat_mean, feat_std)
        source_pred_all = predict_ridge(all_x, source_beta)
        source_pred = source_pred_all[test_mask]
        test_y = y_norm[test_mask]
        test_times = times[test_mask]

        rows.append(
            {
                "seed": seed,
                "dataset": "UCI Electricity Load Diagrams 2011-2014",
                "target_client": client,
                "model": "TDConv-ridge-source-head",
                "protocol": "source-subsampled trainable TDConv source head",
                "adapt_days": 0,
                **metrics(test_y * scale, source_pred * scale),
            }
        )

        start = TEST_START - pd.Timedelta(days=28)
        adapt_mask = (times >= start) & (times < TEST_START)
        if adapt_mask.sum() >= max(12, int(24 * 28 * 0.6)):
            adapter_beta = fit_adapter(source_pred_all[adapt_mask], y_norm[adapt_mask], times[adapt_mask])
            adapter_pred = predict_adapter(source_pred, test_times, adapter_beta)
            rows.append(
                {
                    "seed": seed,
                    "dataset": "UCI Electricity Load Diagrams 2011-2014",
                    "target_client": client,
                    "model": "TDConv-ridge+adapter-28d",
                    "protocol": "source-subsampled trainable TDConv with 28d target adapter",
                    "adapt_days": 28,
                    **metrics(test_y * scale, adapter_pred * scale),
                }
            )
    return pd.DataFrame(rows)


def summarize(result: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_seed = (
        result.groupby(["seed", "model", "protocol", "adapt_days"], as_index=False)
        .agg(
            target_clients=("target_client", "nunique"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_smape=("smape", "mean"),
            total_n=("n", "sum"),
        )
        .sort_values(["model", "seed"])
    )
    baseline = pd.read_csv(RESULTS / "uci_random_conv_representation_results.csv")
    target = baseline[baseline["model"] == "Target-linear-28d"][["target_client", "rmse"]].rename(
        columns={"rmse": "target_linear_28d_rmse"}
    )
    rc = baseline[baseline["model"] == "RC-lag+adapter-28d"][["target_client", "rmse"]].rename(
        columns={"rmse": "rc_adapter_28d_rmse"}
    )
    rows = []
    for seed in sorted(result["seed"].unique()):
        candidate = result[
            (result["seed"] == seed) & (result["model"] == "TDConv-ridge+adapter-28d")
        ][["target_client", "rmse"]].rename(columns={"rmse": "tdconv_adapter_28d_rmse"})
        merged = candidate.merge(target, on="target_client").merge(rc, on="target_client")
        if merged.empty:
            continue
        for baseline_name, col in [
            ("Target-linear-28d", "target_linear_28d_rmse"),
            ("RC-lag+adapter-28d", "rc_adapter_28d_rmse"),
        ]:
            delta = merged[col] - merged["tdconv_adapter_28d_rmse"]
            wins = int((delta > 1e-12).sum())
            losses = int((delta < -1e-12).sum())
            rows.append(
                {
                    "seed": seed,
                    "comparison": f"TDConv 28d adapter vs {baseline_name}",
                    "baseline_model": baseline_name,
                    "target_clients": int(len(merged)),
                    "mean_baseline_rmse": float(merged[col].mean()),
                    "mean_candidate_rmse": float(merged["tdconv_adapter_28d_rmse"].mean()),
                    "mean_rmse_gain_pct": float((delta / merged[col] * 100).mean()),
                    "wins": wins,
                    "losses": losses,
                    "ties": int(len(merged) - wins - losses),
                    "sign_test_p_two_sided": exact_two_sided_sign_p(wins, losses),
                }
            )
    tests = pd.DataFrame(rows)

    summary_rows = []
    for model in ["TDConv-ridge-source-head", "TDConv-ridge+adapter-28d"]:
        subset = by_seed[by_seed["model"] == model]
        summary_rows.append(
            {
                "model": model,
                "seeds": int(subset["seed"].nunique()),
                "mean_rmse_mean": float(subset["mean_rmse"].mean()),
                "mean_rmse_std": float(subset["mean_rmse"].std(ddof=1)),
                "mean_rmse_min": float(subset["mean_rmse"].min()),
                "mean_rmse_max": float(subset["mean_rmse"].max()),
                "mean_smape_mean": float(subset["mean_smape"].mean()),
                "mean_smape_std": float(subset["mean_smape"].std(ddof=1)),
            }
        )
    target_tests = tests[tests["baseline_model"] == "Target-linear-28d"]
    rc_tests = tests[tests["baseline_model"] == "RC-lag+adapter-28d"]
    summary_rows.append(
        {
            "model": "TDConv-ridge+adapter-28d stability tests",
            "seeds": int(by_seed["seed"].nunique()),
            "mean_rmse_mean": float(by_seed[by_seed["model"] == "TDConv-ridge+adapter-28d"]["mean_rmse"].mean()),
            "mean_rmse_std": float(by_seed[by_seed["model"] == "TDConv-ridge+adapter-28d"]["mean_rmse"].std(ddof=1)),
            "mean_rmse_min": float(by_seed[by_seed["model"] == "TDConv-ridge+adapter-28d"]["mean_rmse"].min()),
            "mean_rmse_max": float(by_seed[by_seed["model"] == "TDConv-ridge+adapter-28d"]["mean_rmse"].max()),
            "mean_smape_mean": float(by_seed[by_seed["model"] == "TDConv-ridge+adapter-28d"]["mean_smape"].mean()),
            "mean_smape_std": float(by_seed[by_seed["model"] == "TDConv-ridge+adapter-28d"]["mean_smape"].std(ddof=1)),
            "target_baseline_min_wins": int(target_tests["wins"].min()),
            "target_baseline_max_losses": int(target_tests["losses"].max()),
            "rc_baseline_min_wins": int(rc_tests["wins"].min()),
            "rc_baseline_max_losses": int(rc_tests["losses"].max()),
        }
    )
    summary = pd.DataFrame(summary_rows)
    return by_seed, tests, summary


def plot(by_seed: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame) -> None:
    adapter = by_seed[by_seed["model"] == "TDConv-ridge+adapter-28d"].sort_values("seed")
    source = by_seed[by_seed["model"] == "TDConv-ridge-source-head"].sort_values("seed")
    rc_rmse = float(pd.read_csv(RESULTS / "uci_random_conv_representation_summary.csv").query("model == 'RC-lag+adapter-28d'")["mean_rmse"].iloc[0])
    target_rmse = float(pd.read_csv(RESULTS / "uci_random_conv_representation_summary.csv").query("model == 'Target-linear-28d'")["mean_rmse"].iloc[0])

    width, height = 2150, 1050
    left, top, right, bottom = 170, 170, 1720, 790
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 48), "Multi-seed TDConv stability on UCI load transfer", fill="#172033", font=font(42, True))
    d.text((82, 104), "Eight source-window subsampling seeds; mean RMSE across ten held-out target clients", fill="#5f6b7a", font=font(24))

    y_min = min(float(adapter["mean_rmse"].min()), float(source["mean_rmse"].min()), rc_rmse) - 1.0
    y_max = max(target_rmse, float(adapter["mean_rmse"].max()), float(source["mean_rmse"].max())) + 3.0
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        val = y_min + i * (y_max - y_min) / 5
        d.line((left, y, right, y), fill="#D7DCE2", width=1)
        d.text((88, y - 12), f"{val:.1f}", fill="#526070", font=font(20))

    def xy(seed_index: int, value: float) -> tuple[float, float]:
        x = left + seed_index * (right - left) / (len(SEEDS) - 1)
        y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        return x, y

    for val, label, color in [
        (rc_rmse, "RC 28d adapter", "#7F3C8D"),
        (target_rmse, "Target ridge 28d", "#F58518"),
    ]:
        y = xy(0, val)[1]
        d.line((left, y, right, y), fill=color, width=3)
        d.text((right + 18, y - 13), f"{label}: {val:.2f}", fill=color, font=font(20, True))

    for frame, color, label in [
        (source, "#54A24B", "TDConv source"),
        (adapter, "#4C78A8", "TDConv 28d adapter"),
    ]:
        points = [xy(i, float(row.mean_rmse)) for i, row in enumerate(frame.itertuples(index=False))]
        d.line(points, fill=color, width=4)
        for x, y in points:
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        lx, ly = points[-1]
        label_y_offset = -22 if label == "TDConv source" else 6
        d.text((lx + 14, ly + label_y_offset), label, fill=color, font=font(20, True))

    for i, seed in enumerate(SEEDS):
        x, _ = xy(i, y_min)
        d.text((x - 34, bottom + 20), str(seed)[-2:], fill="#526070", font=font(19))
    d.text((left, bottom + 58), "Seed suffix", fill="#526070", font=font(20))

    stab = summary[summary["model"] == "TDConv-ridge+adapter-28d stability tests"].iloc[0]
    note = (
        f"Adapter RMSE {stab.mean_rmse_mean:.3f} +/- {stab.mean_rmse_std:.3f}; "
        f"all seeds beat RC and target baselines on at least {int(stab.rc_baseline_min_wins)}/10 and "
        f"{int(stab.target_baseline_min_wins)}/10 clients."
    )
    d.text((82, 875), note, fill="#374151", font=font(23, True))
    d.text(
        (82, 925),
        "Source: UCI Electricity Load Diagrams 2011-2014; only pre-test source windows are resampled, preserving the chronological target split.",
        fill="#6b7280",
        font=font(19),
    )
    img.save(FIGURE_OUT)


def main() -> None:
    hourly, source_clients, target_clients = load_data()
    scales = build_scales(hourly, source_clients, target_clients)
    result = pd.concat(
        [evaluate_seed(hourly, source_clients, target_clients, scales, seed) for seed in SEEDS],
        ignore_index=True,
    )
    result.to_csv(RESULTS_OUT, index=False)
    by_seed, tests, summary = summarize(result)
    by_seed.to_csv(BY_SEED_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    tests.to_csv(TESTS_OUT, index=False)
    plot(by_seed, tests, summary)
    print(RESULTS_OUT)
    print(BY_SEED_OUT)
    print(SUMMARY_OUT)
    print(TESTS_OUT)
    print(FIGURE_OUT)
    print(summary.to_string(index=False))
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
