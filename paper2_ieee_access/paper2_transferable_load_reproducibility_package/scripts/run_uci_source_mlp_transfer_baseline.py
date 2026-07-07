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


RESULTS_OUT = RESULTS / "uci_source_mlp_transfer_results.csv"
SUMMARY_OUT = RESULTS / "uci_source_mlp_transfer_summary.csv"
TESTS_OUT = RESULTS / "uci_source_mlp_transfer_client_level_tests.csv"
DIAGNOSTICS_OUT = RESULTS / "uci_source_mlp_transfer_training_diagnostics.csv"
FIGURE_OUT = FIGURES / "paper2_fig17_uci_source_mlp_transfer_baseline.png"

RNG_SEED = 20260706
MAX_SOURCE_WINDOWS = 70000
HIDDEN = 64
EPOCHS = 32
BATCH_SIZE = 1024
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 2e-5
RIDGE_L2 = 5e-2
ADAPT_DAYS = [7, 28]


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


def supervised_features(windows: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    recent = windows[:, -72:]
    daily = windows.reshape(len(windows), 7, 24)
    parts = [
        recent,
        daily.mean(axis=2),
        daily.std(axis=2),
        daily[:, :, -1],
        np.column_stack(
            [
                windows[:, -1],
                windows[:, -24],
                windows[:, -168],
                windows[:, -24:].mean(axis=1),
                windows[:, -24:].std(axis=1),
                windows[:, -168:].mean(axis=1),
                windows[:, -168:].std(axis=1),
            ]
        ),
        calendar_features(times),
    ]
    return np.column_stack(parts).astype(np.float32)


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
    x = supervised_features(windows, times)
    mean, std = standardize_fit(x)
    return standardize_apply(x, mean, std), y.astype(np.float32), mean, std


def init_params(n_features: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w1": rng.normal(0.0, np.sqrt(2.0 / n_features), size=(n_features, HIDDEN)).astype(np.float32),
        "b1": np.zeros(HIDDEN, dtype=np.float32),
        "w2": rng.normal(0.0, np.sqrt(2.0 / HIDDEN), size=(HIDDEN, 1)).astype(np.float32),
        "b2": np.zeros(1, dtype=np.float32),
    }


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def forward(x: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    hidden = relu(x @ params["w1"] + params["b1"])
    pred = (hidden @ params["w2"] + params["b2"]).reshape(-1)
    return pred.astype(np.float32), hidden.astype(np.float32)


def train_source_mlp(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, np.ndarray], float, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    order = rng.permutation(len(x))
    val_n = max(5000, int(0.12 * len(order)))
    val_idx = order[:val_n]
    train_idx = order[val_n:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    params = init_params(x.shape[1], rng)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    rows = []
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    for epoch in range(1, EPOCHS + 1):
        epoch_order = rng.permutation(len(x_train))
        for start in range(0, len(epoch_order), BATCH_SIZE):
            step += 1
            idx = epoch_order[start : start + BATCH_SIZE]
            xb = x_train[idx]
            yb = y_train[idx]
            pred, hidden = forward(xb, params)
            err = pred - yb
            grad_pred = (2.0 / len(xb)) * err
            grad_w2 = hidden.T @ grad_pred[:, None] + WEIGHT_DECAY * params["w2"]
            grad_b2 = np.asarray([grad_pred.sum()], dtype=np.float32)
            grad_hidden = grad_pred[:, None] @ params["w2"].T
            grad_z1 = grad_hidden * (hidden > 0)
            grad_w1 = xb.T @ grad_z1 + WEIGHT_DECAY * params["w1"]
            grad_b1 = grad_z1.sum(axis=0)
            grads = {
                "w1": grad_w1.astype(np.float32),
                "b1": grad_b1.astype(np.float32),
                "w2": grad_w2.astype(np.float32),
                "b2": grad_b2.astype(np.float32),
            }
            for key in params:
                m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
                v[key] = beta2 * v[key] + (1 - beta2) * (grads[key] ** 2)
                m_hat = m[key] / (1 - beta1**step)
                v_hat = v[key] / (1 - beta2**step)
                params[key] = (params[key] - LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)
        train_pred, _ = forward(x_train, params)
        val_pred, _ = forward(x_val, params)
        rows.append(
            {
                "epoch": epoch,
                "train_rmse_normalized": float(np.sqrt(np.mean((y_train - train_pred) ** 2))),
                "validation_rmse_normalized": float(np.sqrt(np.mean((y_val - val_pred) ** 2))),
            }
        )
    diagnostics = pd.DataFrame(rows)
    best_epoch = int(diagnostics.sort_values("validation_rmse_normalized").iloc[0]["epoch"])
    diagnostics["selected_epoch"] = best_epoch
    diagnostics["source_train_windows"] = int(len(x_train))
    diagnostics["source_validation_windows"] = int(len(x_val))
    diagnostics.to_csv(DIAGNOSTICS_OUT, index=False)
    return params, float(diagnostics["validation_rmse_normalized"].min()), diagnostics


def evaluate_target_mlp_head(windows, y_norm, times, test_mask, days, feat_mean, feat_std):
    start = TEST_START - pd.Timedelta(days=days)
    adapt_mask = (times >= start) & (times < TEST_START)
    if adapt_mask.sum() < max(12, int(24 * days * 0.6)):
        return None
    x_adapt = standardize_apply(supervised_features(windows[adapt_mask], times[adapt_mask]), feat_mean, feat_std)
    x_test = standardize_apply(supervised_features(windows[test_mask], times[test_mask]), feat_mean, feat_std)
    beta = fit_ridge(x_adapt, y_norm[adapt_mask], l2=RIDGE_L2)
    return predict_ridge(x_test, beta)


def evaluate(hourly, source_clients, target_clients, scales):
    source_x, source_y, feat_mean, feat_std = collect_source_training(hourly, source_clients, scales)
    params, best_val_rmse, _ = train_source_mlp(source_x, source_y)
    rows = []
    for client in target_clients:
        windows, y_norm, times = client_supervised_arrays(hourly, client, scales[client])
        test_mask = times >= TEST_START
        if test_mask.sum() < 24 * 30:
            continue
        scale = scales[client]
        all_x = standardize_apply(supervised_features(windows, times), feat_mean, feat_std)
        source_pred_all, hidden_all = forward(all_x, params)
        source_pred = source_pred_all[test_mask]
        hidden_test = hidden_all[test_mask]
        test_y = y_norm[test_mask]
        test_times = times[test_mask]

        rows.append(
            {
                "dataset": "UCI Electricity Load Diagrams 2011-2014",
                "target_client": client,
                "model": "SourceMLP-source-head",
                "protocol": "source-trained one-hidden-layer neural encoder",
                "adapt_days": 0,
                "source_validation_rmse_normalized": best_val_rmse,
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
                    "model": f"SourceMLP+adapter-{days}d",
                    "protocol": "source-trained neural encoder with target adapter",
                    "adapt_days": days,
                    "source_validation_rmse_normalized": best_val_rmse,
                    **metrics(test_y * scale, adapter_pred * scale),
                }
            )

            hidden_adapt = hidden_all[adapt_mask]
            hidden_beta = fit_ridge(hidden_adapt, y_norm[adapt_mask], l2=RIDGE_L2)
            hidden_pred = predict_ridge(hidden_test, hidden_beta)
            rows.append(
                {
                    "dataset": "UCI Electricity Load Diagrams 2011-2014",
                    "target_client": client,
                    "model": f"SourceMLP-hidden-target-head-{days}d",
                    "protocol": "frozen source-trained hidden representation with target ridge head",
                    "adapt_days": days,
                    "source_validation_rmse_normalized": best_val_rmse,
                    **metrics(test_y * scale, hidden_pred * scale),
                }
            )

            target_pred = evaluate_target_mlp_head(windows, y_norm, times, test_mask, days, feat_mean, feat_std)
            if target_pred is not None:
                rows.append(
                    {
                        "dataset": "UCI Electricity Load Diagrams 2011-2014",
                        "target_client": client,
                        "model": f"SourceMLP-target-ridge-features-{days}d",
                        "protocol": "target-only ridge on SourceMLP input features",
                        "adapt_days": days,
                        "source_validation_rmse_normalized": best_val_rmse,
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
            source_validation_rmse_normalized=("source_validation_rmse_normalized", "first"),
        )
        .sort_values(["mean_rmse", "adapt_days", "model"])
    )
    summary.to_csv(SUMMARY_OUT, index=False)
    return result, summary


def paired_stats(result: pd.DataFrame) -> pd.DataFrame:
    td = pd.read_csv(RESULTS / "uci_trainable_tdconv_baseline_results.csv")
    patch = pd.read_csv(RESULTS / "uci_patch_attention_transfer_results.csv")
    rc = pd.read_csv(RESULTS / "uci_random_conv_representation_results.csv")
    base = pd.read_csv(RESULTS / "uci_ssl_cold_start_results.csv")
    joined = (
        pd.concat([result, td, patch, rc, base], ignore_index=True)
        .sort_values(["model", "target_client"])
        .drop_duplicates(["model", "target_client"], keep="first")
    )
    specs = [
        ("SourceMLP 28d adapter vs patch-attention 28d", "SourceMLP+adapter-28d", "PatchAttention-ridge+adapter-28d"),
        ("SourceMLP 28d adapter vs TDConv 28d", "SourceMLP+adapter-28d", "TDConv-ridge+adapter-28d"),
        ("SourceMLP source vs TDConv source", "SourceMLP-source-head", "TDConv-ridge-source-head"),
        ("SourceMLP 28d adapter vs RC 28d", "SourceMLP+adapter-28d", "RC-lag+adapter-28d"),
        ("SourceMLP 28d adapter vs target ridge 28d", "SourceMLP+adapter-28d", "Target-linear-28d"),
        ("SourceMLP hidden target head 28d vs target ridge 28d", "SourceMLP-hidden-target-head-28d", "Target-linear-28d"),
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
    patch_summary = pd.read_csv(RESULTS / "uci_patch_attention_transfer_summary.csv")
    td_summary = pd.read_csv(RESULTS / "uci_trainable_tdconv_baseline_summary.csv")
    rc_summary = pd.read_csv(RESULTS / "uci_random_conv_representation_summary.csv")
    base_summary = pd.read_csv(RESULTS / "uci_ssl_cold_start_summary.csv")
    keep_models = [
        "SourceMLP+adapter-28d",
        "SourceMLP-source-head",
        "SourceMLP-hidden-target-head-28d",
        "PatchAttention-ridge+adapter-28d",
        "TDConv-ridge+adapter-28d",
        "RC-lag+adapter-28d",
        "Target-linear-28d",
    ]
    combined = pd.concat([summary, patch_summary, td_summary, rc_summary, base_summary], ignore_index=True)
    combined = combined[combined["model"].isin(keep_models)].drop_duplicates("model", keep="first")
    combined = combined.sort_values("mean_rmse")

    width, height = 2020, 1120
    left, top, right, bottom = 760, 165, 1800, 850
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((86, 52), "Source-trained MLP encoder check on UCI load transfer", fill="#172033", font=font(40, True))
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
        if model.startswith("SourceMLP"):
            color = "#2F6F73"
        elif model.startswith("PatchAttention"):
            color = "#9467BD"
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
        d.text((90, y + 3), model, fill="#1f2937", font=font(21, True if idx == 0 else False))
        d.text((x1 + 12, y + 3), f"{float(row.mean_rmse):.2f}", fill="#1f2937", font=font(21, True))

    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    stat = tests[tests["comparison"] == "SourceMLP 28d adapter vs patch-attention 28d"]
    if not stat.empty:
        s = stat.iloc[0]
        note = (
            f"SourceMLP 28d vs patch-attention 28d: {int(s.wins)}/{int(s.target_clients)} wins, "
            f"mean gain {float(s.mean_rmse_gain_pct):.2f}%, p={float(s.sign_test_p_two_sided):.3f}"
        )
    else:
        note = "SourceMLP is a CPU-only neural representation boundary check."
    d.text((90, 940), note, fill="#374151", font=font(23, True))
    d.text(
        (90, 990),
        "The model trains a one-hidden-layer encoder on pooled source clients, then evaluates source-head, adapter, and frozen-hidden target-head protocols.",
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
    print(DIAGNOSTICS_OUT)
    print(FIGURE_OUT)
    print(summary.to_string(index=False))
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
