from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_opsd_graph_temporal_price_ablation import (
    TARGET,
    TIDY,
    RESULTS,
    FIGURES,
    point_metrics,
    split_zone,
)
from run_opsd_modern_sequence_price_baselines import (
    LOOKBACK,
    fit_sequence_models,
    predict_sequence_models,
)
from run_opsd_tdconv_sequence_anchor_graphpatch_price_model import (
    fit_tdconv,
    predict_tdconv,
)

PATCH_HOURS = 24
PATCH_COUNT = LOOKBACK // PATCH_HOURS
L2 = 4e-2


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


def fit_standardized_ridge_np(x, y, l2=L2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std
    xs = np.column_stack([np.ones(len(xs)), xs])
    eye = np.eye(xs.shape[1])
    eye[0, 0] = 0.0
    beta = np.linalg.solve(xs.T @ xs + l2 * eye, xs.T @ y)
    return beta, mean, std


def predict_standardized_ridge_np(x, model):
    beta, mean, std = model
    x = np.asarray(x, dtype=float)
    xs = (x - mean) / std
    xs = np.column_stack([np.ones(len(xs)), xs])
    return xs @ beta


def paired_sign_test(base_abs, model_abs):
    diff = np.asarray(base_abs, dtype=float) - np.asarray(model_abs, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    n = int(len(diff))
    if n == 0:
        return {
            "paired_n": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_abs_error_delta": np.nan,
            "sign_test_z": np.nan,
            "sign_test_p_approx": np.nan,
        }
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    z = (wins - n / 2) / sqrt(n / 4)
    return {
        "paired_n": n,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n,
        "mean_abs_error_delta": float(np.mean(diff)),
        "sign_test_z": float(z),
        "sign_test_p_approx": float(erfc(abs(z) / sqrt(2))),
    }


def patch_attention_features(frame: pd.DataFrame) -> np.ndarray:
    lag_cols = [f"price_lag_{lag}" for lag in range(LOOKBACK, 0, -1)]
    seq = frame[lag_cols].to_numpy(np.float32)
    patches = seq.reshape(len(seq), PATCH_COUNT, PATCH_HOURS)
    patch_mean = patches.mean(axis=2)
    patch_std = patches.std(axis=2)
    patch_last = patches[:, :, -1]
    patch_min = patches.min(axis=2)
    patch_max = patches.max(axis=2)
    patch_slope = (patches[:, :, -1] - patches[:, :, 0]) / PATCH_HOURS

    query = np.column_stack([patch_mean[:, -1], patch_std[:, -1], patch_last[:, -1]])
    keys = np.stack([patch_mean, patch_std, patch_last], axis=2)
    key_scale = np.nanstd(keys.reshape(-1, keys.shape[-1]), axis=0)
    key_scale[key_scale < 1e-8] = 1.0
    distance = np.mean(np.abs((keys - query[:, None, :]) / key_scale[None, None, :]), axis=2)
    logits = -distance
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / weights.sum(axis=1, keepdims=True)

    attended = np.column_stack(
        [
            np.sum(weights * patch_mean, axis=1),
            np.sum(weights * patch_std, axis=1),
            np.sum(weights * patch_last, axis=1),
            np.sum(weights * patch_min, axis=1),
            np.sum(weights * patch_max, axis=1),
            np.sum(weights * patch_slope, axis=1),
            np.max(weights, axis=1),
            -np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0)), axis=1),
        ]
    )
    calendar_cols = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month",
        "load_mw",
        "solar_mw",
        "wind_mw",
    ]
    calendar = frame[calendar_cols].to_numpy(np.float32)
    return np.column_stack(
        [
            patch_mean,
            patch_std,
            patch_last,
            patch_min,
            patch_max,
            patch_slope,
            attended,
            calendar,
        ]
    ).astype(np.float32)


def fit_patch_attention(train: pd.DataFrame):
    return fit_standardized_ridge_np(
        patch_attention_features(train),
        train[TARGET].to_numpy(float),
    )


def predict_patch_attention(frame: pd.DataFrame, model):
    return predict_standardized_ridge_np(patch_attention_features(frame), model)


def make_full_sequence_frame() -> pd.DataFrame:
    if not TIDY.exists():
        raise SystemExit(f"Missing processed OPSD file: {TIDY}")
    tidy = pd.read_csv(TIDY, parse_dates=["timestamp_utc"])
    tidy = tidy.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    lag_frame = pd.concat(
        {
            f"price_lag_{lag}": tidy.groupby("zone")[TARGET].shift(lag)
            for lag in range(1, LOOKBACK + 1)
        },
        axis=1,
    )
    tidy = pd.concat([tidy, lag_frame], axis=1)
    tidy["lag_1_24_abs"] = (tidy["price_lag_1"] - tidy["price_lag_24"]).abs()
    return tidy


def weighted_objective(frame: pd.DataFrame, pred: np.ndarray):
    y = frame[TARGET].to_numpy(float)
    threshold = float(np.nanquantile(frame["lag_1_24_abs"].to_numpy(float), 0.90))
    mask = frame["lag_1_24_abs"].to_numpy(float) >= threshold
    all_rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    spike_rmse = float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2))) if np.any(mask) else all_rmse
    return 0.55 * all_rmse + 0.45 * spike_rmse


def evaluate_zone(zone: str, zdf: pd.DataFrame):
    sequence_lags = [f"price_lag_{lag}" for lag in range(1, LOOKBACK + 1)]
    cols = [
        "timestamp_utc",
        "zone",
        TARGET,
        "lag_1_24_abs",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month",
        "load_mw",
        "solar_mw",
        "wind_mw",
    ] + sequence_lags
    clean = zdf[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train, cal, test = split_zone(clean)

    seq_models = fit_sequence_models(train)
    td_model = fit_tdconv(train)
    patch_model = fit_patch_attention(train)
    for part in [cal, test]:
        dlinear, nlinear = predict_sequence_models(part, seq_models)
        part["dlinear_seq_pred"] = dlinear
        part["nlinear_seq_pred"] = nlinear
        part["tdconv_seq_pred"] = predict_tdconv(part, td_model)
        part["patch_attention_seq_pred"] = predict_patch_attention(part, patch_model)

    cal_choices = [
        ("DLinear-style sequence ridge", "dlinear_seq_pred"),
        ("NLinear-style sequence ridge", "nlinear_seq_pred"),
        ("TDConv-style sequence ridge", "tdconv_seq_pred"),
        ("Patch-attention sequence ridge", "patch_attention_seq_pred"),
    ]
    selected = min(cal_choices, key=lambda item: weighted_objective(cal, cal[item[1]].to_numpy(float)))
    test["selected_strong_anchor_pred"] = test[selected[1]]

    spike_threshold = float(np.nanquantile(train["lag_1_24_abs"], 0.90))
    test["spike_regime"] = test["lag_1_24_abs"] >= spike_threshold
    rows = []
    paired_rows = []
    model_specs = [
        ("DLinear-style sequence ridge", "dlinear_seq_pred"),
        ("NLinear-style sequence ridge", "nlinear_seq_pred"),
        ("TDConv-style sequence ridge", "tdconv_seq_pred"),
        ("Patch-attention sequence ridge", "patch_attention_seq_pred"),
        (f"Calibration-selected strong anchor ({selected[0]})", "selected_strong_anchor_pred"),
    ]
    for regime, mask in [
        ("all", np.ones(len(test), dtype=bool)),
        ("spike", test["spike_regime"].to_numpy(bool)),
        ("non_spike", ~test["spike_regime"].to_numpy(bool)),
    ]:
        sub = test.loc[mask]
        base_abs = np.abs(sub[TARGET].to_numpy(float) - sub["tdconv_seq_pred"].to_numpy(float))
        for model_name, col in model_specs:
            rows.append(
                {
                    "dataset": "OPSD",
                    "zone": zone,
                    "regime": regime,
                    "model": model_name,
                    "lookback_hours": LOOKBACK,
                    "patch_hours": PATCH_HOURS,
                    "patch_count": PATCH_COUNT,
                    "selected_anchor": selected[0],
                    "spike_threshold": spike_threshold,
                    **point_metrics(sub[TARGET], sub[col]),
                }
            )
            model_abs = np.abs(sub[TARGET].to_numpy(float) - sub[col].to_numpy(float))
            paired_rows.append(
                {
                    "dataset": "OPSD",
                    "zone": zone,
                    "regime": regime,
                    "model": model_name,
                    "baseline": "TDConv-style sequence ridge",
                    **paired_sign_test(base_abs, model_abs),
                }
            )
    daily = test[
        [
            "timestamp_utc",
            "zone",
            TARGET,
            "lag_1_24_abs",
            "spike_regime",
            "dlinear_seq_pred",
            "nlinear_seq_pred",
            "tdconv_seq_pred",
            "patch_attention_seq_pred",
            "selected_strong_anchor_pred",
        ]
    ].copy()
    return rows, paired_rows, daily


def run():
    tidy = make_full_sequence_frame()
    rows = []
    paired_rows = []
    daily_parts = []
    for zone, zdf in tidy.groupby("zone", sort=True):
        zone_rows, zone_paired, zone_daily = evaluate_zone(zone, zdf)
        rows.extend(zone_rows)
        paired_rows.extend(zone_paired)
        daily_parts.append(zone_daily)
    summary = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)
    daily = pd.concat(daily_parts, ignore_index=True)
    summary.to_csv(RESULTS / "opsd_patch_attention_price_baseline_summary.csv", index=False)
    paired.to_csv(RESULTS / "opsd_patch_attention_price_baseline_paired_tests.csv", index=False)
    daily.to_csv(RESULTS / "opsd_patch_attention_price_baseline_daily.csv", index=False)
    return summary, paired


def plot_patch_attention(summary: pd.DataFrame):
    spike = summary[summary["regime"] == "spike"].copy()
    zones = sorted(spike["zone"].unique())
    models = [
        "TDConv-style sequence ridge",
        "Patch-attention sequence ridge",
    ]
    selected = spike[spike["model"].str.startswith("Calibration-selected strong anchor")].copy()
    plot = pd.concat([spike[spike["model"].isin(models)], selected], ignore_index=True)
    model_order = models + sorted(selected["model"].unique())
    colors = {
        "TDConv-style sequence ridge": "#4C78A8",
        "Patch-attention sequence ridge": "#E45756",
    }
    for m in model_order:
        colors.setdefault(m, "#54A24B")
    img = Image.new("RGB", (1580, 1060), "white")
    d = ImageDraw.Draw(img)
    d.text((84, 56), "Patch-attention reviewer baseline on OPSD spike hours", fill="#172033", font=font(40, True))
    d.text(
        (86, 114),
        "CPU-only patch pooling ridge; designed as PatchTST/TFT-family evidence without claiming full Transformer training",
        fill="#5f6b7a",
        font=font(23),
    )
    left, top, width, height = 145, 190, 1220, 650
    vals = [
        float(plot[(plot["zone"] == z) & (plot["model"] == m)]["rmse"].iloc[0])
        for z in zones
        for m in model_order
        if len(plot[(plot["zone"] == z) & (plot["model"] == m)])
    ]
    ymax = max(vals) * 1.18
    for i in range(6):
        y = top + height - i * height / 5
        d.line((left, y, left + width, y), fill="#e5e7eb", width=1)
        d.text((left - 78, y - 14), f"{ymax * i / 5:.1f}", fill="#657385", font=font(20))
    group_w = width / len(zones)
    bar_w = min(58, group_w / 6)
    for zi, zone in enumerate(zones):
        gx = left + zi * group_w + 48
        for mi, model in enumerate(model_order):
            mdf = plot[(plot["zone"] == zone) & (plot["model"] == model)]
            if mdf.empty:
                continue
            val = float(mdf["rmse"].iloc[0])
            bh = val / ymax * height
            x0 = gx + mi * (bar_w + 12)
            y0 = top + height - bh
            d.rounded_rectangle((x0, y0, x0 + bar_w, top + height), radius=6, fill=colors[model])
            d.text((x0 - 3, y0 - 28), f"{val:.2f}", fill="#1f2937", font=font(18, True))
        d.text((gx + 70, top + height + 38), zone, fill="#1f2937", font=font(23, True), anchor="mm")
    lx, ly = 145, 908
    labels = {
        "TDConv-style sequence ridge": "TDConv-style",
        "Patch-attention sequence ridge": "Patch-attention",
    }
    for i, model in enumerate(model_order):
        x = lx + i * 390
        d.rectangle((x, ly, x + 28, ly + 28), fill=colors[model])
        label = labels.get(model, "Calibration-selected")
        d.text((x + 42, ly - 4), label, fill="#526070", font=font(20))
    d.text(
        (86, 990),
        "Source: OPSD public data; final chronological test split; spike threshold fitted on each zone's training split.",
        fill="#6b7280",
        font=font(20),
    )
    out = FIGURES / "paper1_fig17_opsd_patch_attention_baseline.png"
    img.save(out)
    return out


def main():
    summary, paired = run()
    print(RESULTS / "opsd_patch_attention_price_baseline_summary.csv")
    print(RESULTS / "opsd_patch_attention_price_baseline_paired_tests.csv")
    print(RESULTS / "opsd_patch_attention_price_baseline_daily.csv")
    print(plot_patch_attention(summary))
    print(paired[(paired["regime"] == "spike") & (paired["model"] == "Patch-attention sequence ridge")])


if __name__ == "__main__":
    main()
