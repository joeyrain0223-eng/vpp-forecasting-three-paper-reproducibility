from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

WIDE = PROCESSED / "uci_electricity_hourly_selected_clients_wide.csv"
STATS = RESULTS / "uci_load_dataset_stats.csv"

TEST_START = pd.Timestamp("2014-10-01 00:00:00")
SOURCE_TRAIN_START = pd.Timestamp("2014-01-08 00:00:00")
ADAPT_7_START = TEST_START - pd.Timedelta(days=7)
ADAPT_28_START = TEST_START - pd.Timedelta(days=28)
HISTORY = 168
N_COMPONENTS = 16
MASK_RATE = 0.20
MAX_PRETRAIN_WINDOWS = 80000
RNG_SEED = 20260630

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


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


def smape(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    den = np.abs(y) + np.abs(pred)
    out = np.zeros_like(den, dtype=float)
    mask = den != 0
    out[mask] = 2 * np.abs(y[mask] - pred[mask]) / den[mask]
    return float(np.mean(out) * 100)


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = y - pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "smape": smape(y, pred),
        "n": int(len(y)),
    }


def fit_ridge(x, y, l2=1e-4, fit_intercept=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if fit_intercept:
        x = np.column_stack([np.ones(len(x)), x])
    eye = np.eye(x.shape[1])
    if fit_intercept:
        eye[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + l2 * eye, x.T @ y)


def predict_ridge(x, beta, fit_intercept=True):
    x = np.asarray(x, dtype=float)
    if fit_intercept:
        x = np.column_stack([np.ones(len(x)), x])
    return x @ beta


def load_data():
    hourly = pd.read_csv(WIDE, parse_dates=["timestamp"]).set_index("timestamp")
    stats = pd.read_csv(STATS)
    source_clients = stats[stats["role"] == "source"]["client"].tolist()
    target_clients = stats[stats["role"] == "target"]["client"].tolist()
    return hourly, source_clients, target_clients


def build_scales(hourly, source_clients, target_clients):
    scales = {}
    for client in source_clients:
        train = hourly.loc[(hourly.index >= SOURCE_TRAIN_START) & (hourly.index < TEST_START), client]
        scales[client] = max(float(train.median()), 1e-6)
    for client in target_clients:
        adapt = hourly.loc[(hourly.index >= ADAPT_28_START) & (hourly.index < TEST_START), client]
        scales[client] = max(float(adapt.median()), 1e-6)
    return scales


def client_supervised_arrays(hourly, client, scale):
    values = hourly[client].to_numpy(float) / scale
    timestamps = hourly.index
    rows = []
    times = []
    y = []
    for idx in range(HISTORY, len(values)):
        window = values[idx - HISTORY : idx]
        target = values[idx]
        if np.isfinite(window).all() and np.isfinite(target):
            rows.append(window)
            times.append(timestamps[idx])
            y.append(target)
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    times = pd.DatetimeIndex(times)
    return x, y, times


def calendar_features(times):
    hour = times.hour.to_numpy(dtype=float)
    dow = times.dayofweek.to_numpy(dtype=float)
    month = times.month.to_numpy(dtype=float)
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            month / 12.0,
        ]
    )


def collect_source_windows(hourly, source_clients, scales):
    windows = []
    for client in source_clients:
        x, _, times = client_supervised_arrays(hourly, client, scales[client])
        mask = (times >= SOURCE_TRAIN_START) & (times < TEST_START)
        if mask.any():
            windows.append(x[mask])
    source = np.vstack(windows)
    if len(source) > MAX_PRETRAIN_WINDOWS:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(len(source), size=MAX_PRETRAIN_WINDOWS, replace=False)
        source = source[idx]
    return source.astype(np.float32)


def fit_masked_reconstruction_basis(source_windows):
    rng = np.random.default_rng(RNG_SEED)
    row_mean = source_windows.mean(axis=1, keepdims=True)
    corrupted = source_windows.copy()
    mask = rng.random(corrupted.shape) < MASK_RATE
    corrupted[mask] = row_mean.repeat(corrupted.shape[1], axis=1)[mask]
    mean_vec = corrupted.mean(axis=0)
    centered = corrupted - mean_vec
    cov = (centered.T @ centered) / max(1, len(centered) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :N_COMPONENTS].astype(np.float32)
    clean_centered = source_windows - mean_vec
    latent = clean_centered @ components
    recon = latent @ components.T + mean_vec
    rec = metrics(source_windows.reshape(-1), recon.reshape(-1))
    explained = float(eigvals[:N_COMPONENTS].sum() / eigvals.clip(min=0).sum())
    pd.DataFrame(
        [
            {
                "history_hours": HISTORY,
                "components": N_COMPONENTS,
                "mask_rate": MASK_RATE,
                "pretrain_windows": int(len(source_windows)),
                "explained_variance_ratio": explained,
                "source_reconstruction_rmse": rec["rmse"],
                "source_reconstruction_mae": rec["mae"],
            }
        ]
    ).to_csv(RESULTS / "uci_ssl_pretraining_diagnostics.csv", index=False)
    return mean_vec.astype(np.float32), components, explained, rec


def encode(windows, mean_vec, components):
    return (windows - mean_vec) @ components


def feature_matrix(windows, times, mean_vec, components, include_lags=False):
    latent = encode(windows, mean_vec, components)
    cal = calendar_features(times)
    parts = [latent, cal]
    if include_lags:
        lag_features = np.column_stack([windows[:, -1], windows[:, -24], windows[:, 0]])
        parts.append(lag_features)
    return np.column_stack(parts)


def source_training_matrix(hourly, source_clients, scales, mean_vec, components, include_lags=False):
    xs = []
    ys = []
    for client in source_clients:
        windows, y, times = client_supervised_arrays(hourly, client, scales[client])
        mask = (times >= SOURCE_TRAIN_START) & (times < TEST_START)
        if mask.any():
            xs.append(feature_matrix(windows[mask], times[mask], mean_vec, components, include_lags=include_lags))
            ys.append(y[mask])
    return np.vstack(xs), np.concatenate(ys)


def fit_adapter(pred_norm, y_norm, times, l2=1e-3):
    cal = calendar_features(times)
    x = np.column_stack([pred_norm, cal[:, :4]])
    return fit_ridge(x, y_norm, l2=l2)


def predict_adapter(pred_norm, times, beta):
    cal = calendar_features(times)
    x = np.column_stack([pred_norm, cal[:, :4]])
    return predict_ridge(x, beta)


def evaluate_ssl_variant(hourly, source_clients, target_clients, scales, mean_vec, components, include_lags, prefix):
    source_x, source_y = source_training_matrix(hourly, source_clients, scales, mean_vec, components, include_lags=include_lags)
    source_beta = fit_ridge(source_x, source_y, l2=1e-3)
    rows = []
    for client in target_clients:
        windows, y_norm, times = client_supervised_arrays(hourly, client, scales[client])
        x_all = feature_matrix(windows, times, mean_vec, components, include_lags=include_lags)
        test_mask = times >= TEST_START
        if test_mask.sum() < 24 * 30:
            continue
        test_x = x_all[test_mask]
        test_y_norm = y_norm[test_mask]
        test_times = times[test_mask]
        scale = scales[client]

        pred_norm = predict_ridge(test_x, source_beta)
        rows.append(
            {
                "dataset": "UCI Electricity Load Diagrams 2011-2014",
                "target_client": client,
                "model": f"{prefix}-source-head",
                "protocol": "source-only frozen representation",
                **metrics(test_y_norm * scale, pred_norm * scale),
            }
        )

        for days, start in [(7, ADAPT_7_START), (28, ADAPT_28_START)]:
            adapt_mask = (times >= start) & (times < TEST_START)
            if adapt_mask.sum() < 24 * days * 0.8:
                continue
            adapt_x = x_all[adapt_mask]
            adapt_y_norm = y_norm[adapt_mask]
            adapt_times = times[adapt_mask]
            adapt_pred_norm = predict_ridge(adapt_x, source_beta)

            adapter_beta = fit_adapter(adapt_pred_norm, adapt_y_norm, adapt_times)
            adapter_pred_norm = predict_adapter(pred_norm, test_times, adapter_beta)
            rows.append(
                {
                    "dataset": "UCI Electricity Load Diagrams 2011-2014",
                    "target_client": client,
                    "model": f"{prefix}+adapter-{days}d",
                    "protocol": "frozen representation with target adapter",
                    **metrics(test_y_norm * scale, adapter_pred_norm * scale),
                }
            )

            target_beta = fit_ridge(adapt_x, adapt_y_norm, l2=1e-3)
            target_pred_norm = predict_ridge(test_x, target_beta)
            rows.append(
                {
                    "dataset": "UCI Electricity Load Diagrams 2011-2014",
                    "target_client": client,
                    "model": f"{prefix}+target-head-{days}d",
                    "protocol": "frozen representation with target head",
                    **metrics(test_y_norm * scale, target_pred_norm * scale),
                }
            )
    return rows


def evaluate_ssl(hourly, source_clients, target_clients, scales, mean_vec, components):
    rows = []
    rows.extend(evaluate_ssl_variant(hourly, source_clients, target_clients, scales, mean_vec, components, include_lags=False, prefix="SSL-MR"))
    rows.extend(evaluate_ssl_variant(hourly, source_clients, target_clients, scales, mean_vec, components, include_lags=True, prefix="SSL-MR-lag"))
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "uci_ssl_representation_results.csv", index=False)
    summary = (
        result.groupby(["model", "protocol"], as_index=False)
        .agg(
            target_clients=("target_client", "nunique"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_smape=("smape", "mean"),
            total_n=("n", "sum"),
        )
        .sort_values("mean_rmse")
    )
    summary.to_csv(RESULTS / "uci_ssl_representation_summary.csv", index=False)
    return result, summary


def combined_summary_for_plot(ssl_summary):
    baseline = pd.read_csv(RESULTS / "uci_load_transfer_summary.csv")
    keep_base = baseline[baseline["model"].isin(["Target-28d-linear", "Target-7d-linear", "Pooled-source-linear", "Seasonal-168h", "Seasonal-24h"])].copy()
    keep_ssl = ssl_summary.copy()
    keep_base["family"] = "transparent baseline"
    keep_ssl["family"] = "SSL representation prototype"
    combined = pd.concat([keep_base, keep_ssl], ignore_index=True)
    combined = combined.sort_values("mean_rmse").reset_index(drop=True)
    combined.to_csv(RESULTS / "uci_combined_load_transfer_and_ssl_summary.csv", index=False)
    return combined


def plot_combined(combined):
    plot = combined[combined["mean_rmse"] <= 140].sort_values("mean_rmse", ascending=True).copy()
    width, height = 1900, 1180
    left, top, right, bottom = 650, 170, 1730, 980
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "UCI load transfer: baselines and SSL prototype", fill="#172033", font=font(44, True))
    d.text((92, 112), "Mean RMSE across ten target clients; unstable target-head variants are retained in CSV, not plotted", fill="#5f6b7a", font=font(24))
    max_v = float(plot["mean_rmse"].max()) * 1.10
    for i in range(6):
        x = left + i * (right - left) / 5
        d.line((x, top, x, bottom), fill="#D7DCE2", width=1)
        label = f"{max_v * i / 5:.0f}"
        d.text((x - d.textlength(label, font=font(21)) / 2, bottom + 18), label, fill="#526070", font=font(21))
    row_h = (bottom - top) / len(plot)
    colors = {
        "transparent baseline": "#4C78A8",
        "SSL representation prototype": "#E45756",
    }
    for idx, row in enumerate(plot.itertuples(index=False)):
        y = top + idx * row_h + row_h * 0.18
        bar_h = row_h * 0.58
        color = colors.get(row.family, "#72B7B2")
        x1 = left + float(row.mean_rmse) / max_v * (right - left)
        d.rounded_rectangle((left, y, x1, y + bar_h), radius=8, fill=color)
        d.text((90, y + 3), row.model, fill="#1f2937", font=font(22, True if idx == 0 else False))
        val = f"{float(row.mean_rmse):.2f}"
        d.text((x1 + 12, y + 3), val, fill="#1f2937", font=font(21, True))
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    d.rectangle((90, 1020, 124, 1048), fill=colors["transparent baseline"])
    d.text((136, 1016), "Transparent lag/seasonal baselines", fill="#526070", font=font(22))
    d.rectangle((520, 1020, 554, 1048), fill=colors["SSL representation prototype"])
    d.text((566, 1016), "Masked-reconstruction representation prototype", fill="#526070", font=font(22))
    out = FIGURES / "paper2_fig5_uci_ssl_prototype_rmse.png"
    img.save(out)
    return out


def main():
    hourly, source_clients, target_clients = load_data()
    scales = build_scales(hourly, source_clients, target_clients)
    source_windows = collect_source_windows(hourly, source_clients, scales)
    mean_vec, components, explained, rec = fit_masked_reconstruction_basis(source_windows)
    _, summary = evaluate_ssl(hourly, source_clients, target_clients, scales, mean_vec, components)
    combined = combined_summary_for_plot(summary)
    fig = plot_combined(combined)
    print(RESULTS / "uci_ssl_pretraining_diagnostics.csv")
    print(RESULTS / "uci_ssl_representation_results.csv")
    print(RESULTS / "uci_ssl_representation_summary.csv")
    print(RESULTS / "uci_combined_load_transfer_and_ssl_summary.csv")
    print(fig)
    print(pd.read_csv(RESULTS / "uci_ssl_pretraining_diagnostics.csv").to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
