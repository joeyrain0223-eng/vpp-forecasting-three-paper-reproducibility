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
    collect_source_training,
    standardize_apply,
    tdconv_features,
)


RESULTS_OUT = RESULTS / "uci_neural_tdconv_residual_results.csv"
SUMMARY_OUT = RESULTS / "uci_neural_tdconv_residual_summary.csv"
TESTS_OUT = RESULTS / "uci_neural_tdconv_residual_client_level_tests.csv"
DIAGNOSTICS_OUT = RESULTS / "uci_neural_tdconv_residual_training_diagnostics.csv"
FIGURE_OUT = FIGURES / "paper2_fig14_uci_neural_tdconv_residual_check.png"

RNG_SEED = 20260706
HIDDEN = 48
EPOCHS = 36
BATCH_SIZE = 1024
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-5
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


def init_params(n_features: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    scale1 = np.sqrt(2.0 / n_features)
    scale2 = np.sqrt(2.0 / HIDDEN)
    return {
        "w1": (rng.normal(0.0, scale1, size=(n_features, HIDDEN))).astype(np.float32),
        "b1": np.zeros(HIDDEN, dtype=np.float32),
        "w2": (rng.normal(0.0, scale2, size=(HIDDEN, 1))).astype(np.float32),
        "b2": np.zeros(1, dtype=np.float32),
    }


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def forward(x: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    hidden = relu(x @ params["w1"] + params["b1"])
    pred = (hidden @ params["w2"] + params["b2"]).reshape(-1)
    return pred.astype(np.float32), hidden.astype(np.float32)


def adam_train_residual(
    x: np.ndarray,
    residual: np.ndarray,
    x_val: np.ndarray,
    residual_val: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    params = init_params(x.shape[1], rng)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    rows = []
    step = 0
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(order), BATCH_SIZE):
            step += 1
            idx = order[start : start + BATCH_SIZE]
            xb = x[idx]
            yb = residual[idx]
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
        train_pred, _ = forward(x, params)
        val_pred, _ = forward(x_val, params)
        rows.append(
            {
                "epoch": epoch,
                "train_residual_rmse": float(np.sqrt(np.mean((residual - train_pred) ** 2))),
                "val_residual_rmse": float(np.sqrt(np.mean((residual_val - val_pred) ** 2))),
            }
        )
    return params, pd.DataFrame(rows)


def train_source_residual_head(source_x: np.ndarray, source_y: np.ndarray):
    rng = np.random.default_rng(RNG_SEED)
    order = rng.permutation(len(source_x))
    val_n = max(5000, int(0.12 * len(order)))
    val_idx = order[:val_n]
    train_idx = order[val_n:]
    x_train = source_x[train_idx]
    y_train = source_y[train_idx]
    x_val = source_x[val_idx]
    y_val = source_y[val_idx]

    base_beta = fit_ridge(x_train, y_train, l2=5e-2)
    train_base = predict_ridge(x_train, base_beta)
    val_base = predict_ridge(x_val, base_beta)
    residual_train = (y_train - train_base).astype(np.float32)
    residual_val = (y_val - val_base).astype(np.float32)
    params, diagnostics = adam_train_residual(x_train, residual_train, x_val, residual_val)
    val_resid_pred, _ = forward(x_val, params)
    shrinkage_grid = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    scored = []
    for shrinkage in shrinkage_grid:
        pred = val_base + shrinkage * val_resid_pred
        scored.append((float(shrinkage), float(np.sqrt(np.mean((y_val - pred) ** 2)))))
    shrinkage, val_rmse = min(scored, key=lambda item: item[1])
    diagnostics["selected_residual_shrinkage"] = shrinkage
    diagnostics["selected_val_rmse"] = val_rmse
    diagnostics["base_val_rmse"] = float(np.sqrt(np.mean((y_val - val_base) ** 2)))
    diagnostics["source_train_windows"] = int(len(x_train))
    diagnostics["source_validation_windows"] = int(len(x_val))
    diagnostics.to_csv(DIAGNOSTICS_OUT, index=False)
    full_beta = fit_ridge(source_x, source_y, l2=5e-2)
    return full_beta, params, shrinkage, diagnostics


def predict_neural_tdconv(x: np.ndarray, beta: np.ndarray, params: dict[str, np.ndarray], shrinkage: float) -> np.ndarray:
    base = predict_ridge(x, beta)
    residual, _ = forward(x, params)
    return base + shrinkage * residual


def evaluate(hourly, source_clients, target_clients, scales):
    source_x, source_y, feat_mean, feat_std = collect_source_training(hourly, source_clients, scales)
    beta, params, shrinkage, _ = train_source_residual_head(source_x, source_y)
    rows = []
    for client in target_clients:
        windows, y_norm, times = client_supervised_arrays(hourly, client, scales[client])
        test_mask = times >= TEST_START
        if test_mask.sum() < 24 * 30:
            continue
        scale = scales[client]
        all_x = standardize_apply(tdconv_features(windows, times), feat_mean, feat_std)
        source_pred_all = predict_neural_tdconv(all_x, beta, params, shrinkage)
        source_pred = source_pred_all[test_mask]
        test_y = y_norm[test_mask]
        test_times = times[test_mask]
        rows.append(
            {
                "dataset": "UCI Electricity Load Diagrams 2011-2014",
                "target_client": client,
                "model": "Neural-TDConv-residual-source-head",
                "protocol": "source-pooled TDConv ridge plus neural residual head",
                "adapt_days": 0,
                "residual_shrinkage": shrinkage,
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
                    "model": f"Neural-TDConv-residual+adapter-{days}d",
                    "protocol": "source-pooled neural residual representation with target adapter",
                    "adapt_days": days,
                    "residual_shrinkage": shrinkage,
                    **metrics(test_y * scale, adapter_pred * scale),
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
            residual_shrinkage=("residual_shrinkage", "first"),
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
        ("Neural residual 28d vs TDConv 28d", "Neural-TDConv-residual+adapter-28d", "TDConv-ridge+adapter-28d"),
        ("Neural residual source vs TDConv source", "Neural-TDConv-residual-source-head", "TDConv-ridge-source-head"),
        ("Neural residual 28d vs RC 28d", "Neural-TDConv-residual+adapter-28d", "RC-lag+adapter-28d"),
        ("Neural residual 28d vs target ridge 28d", "Neural-TDConv-residual+adapter-28d", "Target-linear-28d"),
        ("Neural residual 7d vs target ridge 7d", "Neural-TDConv-residual+adapter-7d", "Target-linear-7d"),
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
        "Neural-TDConv-residual+adapter-28d",
        "Neural-TDConv-residual-source-head",
        "TDConv-ridge+adapter-28d",
        "TDConv-ridge-source-head",
        "RC-lag+adapter-28d",
        "SSL-MR-lag+adapter-28d",
        "Target-linear-28d",
        "Seasonal-168h",
    ]
    combined = pd.concat([summary, td_summary, rc_summary, ssl_summary], ignore_index=True)
    combined = combined[combined["model"].isin(keep_models)].drop_duplicates("model", keep="first")
    combined = combined.sort_values("mean_rmse")

    width, height = 2000, 1160
    left, top, right, bottom = 720, 170, 1780, 900
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((86, 54), "Neural TDConv residual check on UCI load transfer", fill="#172033", font=font(40, True))
    d.text((90, 108), "Mean RMSE across ten target clients; lower is better", fill="#5f6b7a", font=font(24))

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
        if model.startswith("Neural"):
            color = "#2F6F73"
        elif model.startswith("TDConv"):
            color = "#4C78A8"
        elif model.startswith("RC"):
            color = "#7F3C8D"
        elif model.startswith("SSL-MR"):
            color = "#E45756"
        elif model.startswith("Target"):
            color = "#F58518"
        else:
            color = "#54A24B"
        x1 = left + float(row.mean_rmse) / max_v * (right - left)
        d.rounded_rectangle((left, y, x1, y + bar_h), radius=8, fill=color)
        d.text((90, y + 4), model, fill="#1f2937", font=font(21, True if idx == 0 else False))
        d.text((x1 + 12, y + 4), f"{float(row.mean_rmse):.2f}", fill="#1f2937", font=font(21, True))

    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    stat = tests[tests["comparison"] == "Neural residual 28d vs TDConv 28d"]
    if not stat.empty:
        s = stat.iloc[0]
        note = (
            f"Neural residual 28d vs TDConv 28d: {int(s.wins)}/{int(s.target_clients)} wins, "
            f"mean gain {float(s.mean_rmse_gain_pct):.2f}%, p={float(s.sign_test_p_two_sided):.3f}"
        )
    else:
        note = "The neural residual head is retained as a nonlinear reviewer check, not as an unsupported deep-model claim."
    d.text((90, 985), note, fill="#374151", font=font(23, True))
    d.text(
        (90, 1035),
        "Source: UCI Electricity Load Diagrams 2011-2014; residual shrinkage is selected on source-domain validation only.",
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
    print(pd.read_csv(DIAGNOSTICS_OUT).tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
