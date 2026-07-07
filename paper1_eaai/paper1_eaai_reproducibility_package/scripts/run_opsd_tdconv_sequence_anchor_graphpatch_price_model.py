from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_opsd_graph_temporal_price_ablation import (
    TARGET,
    RESULTS,
    FIGURES,
    fit_standardized_ridge,
    point_metrics,
    predict_standardized_ridge,
    split_zone,
)
from run_opsd_deep_graph_patch_price_model import (
    GRAPH_LAGS,
    PATCH_LAGS,
    prepare_patch_graph_frame,
)
from run_opsd_modern_sequence_price_baselines import (
    LOOKBACK,
    fit_sequence_models,
    predict_sequence_models,
)


RNG_SEED = 20260705
SHRINKAGE_GRID = np.linspace(0, 1, 11)
TD_CONV_SPECS = [
    ("last_12_d1", 12, 1),
    ("last_24_d1", 24, 1),
    ("last_48_d1", 48, 1),
    ("last_24_d2", 24, 2),
    ("last_48_d2", 48, 2),
    ("last_42_d4", 42, 4),
    ("last_21_d8", 21, 8),
]


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


def objective(y, pred, lag_1_24_abs):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    all_rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    threshold = float(np.nanquantile(lag_1_24_abs, 0.90))
    mask = np.asarray(lag_1_24_abs, dtype=float) >= threshold
    spike_rmse = float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2))) if np.any(mask) else all_rmse
    return 0.55 * all_rmse + 0.45 * spike_rmse, all_rmse, spike_rmse


def tdconv_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    lag_cols = [f"price_lag_{lag}" for lag in range(LOOKBACK, 0, -1)]
    seq = frame[lag_cols].to_numpy(np.float32)
    parts = []
    for _, length, dilation in TD_CONV_SPECS:
        offsets = np.arange(length, dtype=int) * dilation
        idx = LOOKBACK - 1 - offsets
        idx = idx[idx >= 0][::-1]
        parts.append(seq[:, idx])
    parts.append(
        np.column_stack(
            [
                seq[:, -1],
                seq[:, -24],
                seq[:, 0],
                seq[:, -24:].mean(axis=1),
                seq[:, -24:].std(axis=1),
                seq[:, -168:].mean(axis=1),
                seq[:, -168:].std(axis=1),
            ]
        ).astype(np.float32)
    )
    calendar = frame[
        [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month",
            "load_mw",
            "solar_mw",
            "wind_mw",
        ]
    ].to_numpy(np.float32)
    parts.append(calendar)
    return np.column_stack(parts).astype(np.float32)


def fit_standardized_ridge_np(x, y, l2=5e-2):
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


def fit_tdconv(train: pd.DataFrame):
    return fit_standardized_ridge_np(tdconv_feature_matrix(train), train[TARGET].to_numpy(float))


def predict_tdconv(frame: pd.DataFrame, model):
    return predict_standardized_ridge_np(tdconv_feature_matrix(frame), model)


def choose_anchor(cal: pd.DataFrame):
    y = cal[TARGET].to_numpy(float)
    lag = cal["lag_1_24_abs"].to_numpy(float)
    choices = []
    for model, col in [
        ("DLinear-style sequence ridge", "dlinear_seq_pred"),
        ("NLinear-style sequence ridge", "nlinear_seq_pred"),
        ("TDConv-style sequence ridge", "tdconv_seq_pred"),
    ]:
        obj, all_rmse, spike_rmse = objective(y, cal[col].to_numpy(float), lag)
        choices.append(
            {
                "anchor_model": model,
                "anchor_col": col,
                "cal_objective": obj,
                "cal_all_rmse": all_rmse,
                "cal_spike_rmse": spike_rmse,
            }
        )
    return min(choices, key=lambda x: x["cal_objective"])


def choose_shrinkage(cal, anchor_col, residual_col):
    y = cal[TARGET].to_numpy(float)
    lag = cal["lag_1_24_abs"].to_numpy(float)
    best = None
    for w in SHRINKAGE_GRID:
        pred = cal[anchor_col].to_numpy(float) + w * cal[residual_col].to_numpy(float)
        obj, all_rmse, spike_rmse = objective(y, pred, lag)
        row = {
            "residual_shrinkage": float(w),
            "cal_objective": obj,
            "cal_all_rmse": all_rmse,
            "cal_spike_rmse": spike_rmse,
        }
        if best is None or obj < best["cal_objective"]:
            best = row
    return best


def evaluate_zone(zone, zdf):
    patch_features = [f"price_lag_{lag}" for lag in PATCH_LAGS] + [
        "lag_1_24_abs",
        "lag_1_168_abs",
        "patch_mean_24",
        "patch_std_24",
        "patch_mean_168",
        "patch_std_168",
    ]
    graph_features = []
    for lag in GRAPH_LAGS:
        graph_features.extend([f"graph_mean_lag_{lag}", f"graph_std_lag_{lag}"])
    model_features = list(
        dict.fromkeys(
            [
                "sequence_anchor_pred",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "month",
                "load_mw",
                "solar_mw",
                "wind_mw",
            ]
            + patch_features
            + graph_features
        )
    )
    sequence_lags = [f"price_lag_{lag}" for lag in range(1, LOOKBACK + 1)]
    cols = list(dict.fromkeys(["timestamp_utc", "zone", TARGET, "hour"] + model_features[1:] + sequence_lags))
    clean = zdf[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train, cal, test = split_zone(clean)

    seq_models = fit_sequence_models(train)
    tdconv_model = fit_tdconv(train)
    for part in [train, cal, test]:
        dlinear, nlinear = predict_sequence_models(part, seq_models)
        part["dlinear_seq_pred"] = dlinear
        part["nlinear_seq_pred"] = nlinear
        part["tdconv_seq_pred"] = predict_tdconv(part, tdconv_model)

    anchor = choose_anchor(cal)
    for part in [train, cal, test]:
        part["sequence_anchor_pred"] = part[anchor["anchor_col"]]

    residual_train = train.copy()
    residual_train["residual"] = residual_train[TARGET] - residual_train["sequence_anchor_pred"]
    residual_model = fit_standardized_ridge(residual_train, model_features, target_col="residual", l2=3e-3)
    for part in [cal, test]:
        part["graphpatch_residual"] = predict_standardized_ridge(part, model_features, residual_model)

    shrink = choose_shrinkage(cal, "sequence_anchor_pred", "graphpatch_residual")
    w = shrink["residual_shrinkage"]
    for part in [cal, test]:
        part["tdconv_inclusive_graphpatch_pred"] = part["sequence_anchor_pred"] + w * part["graphpatch_residual"]

    spike_threshold = float(np.nanquantile(train["lag_1_24_abs"], 0.90))
    test["spike_regime"] = test["lag_1_24_abs"] >= spike_threshold
    model_specs = [
        ("DLinear-style sequence ridge", "dlinear_seq_pred"),
        ("NLinear-style sequence ridge", "nlinear_seq_pred"),
        ("TDConv-style sequence ridge", "tdconv_seq_pred"),
        (f"Selected anchor: {anchor['anchor_model']}", "sequence_anchor_pred"),
        ("TDConv-inclusive GraphPatch residual", "tdconv_inclusive_graphpatch_pred"),
    ]
    rows = []
    paired_rows = []
    for regime, mask in [
        ("all", np.ones(len(test), dtype=bool)),
        ("spike", test["spike_regime"].to_numpy(bool)),
        ("non_spike", ~test["spike_regime"].to_numpy(bool)),
    ]:
        sub = test.loc[mask]
        anchor_abs = np.abs(sub[TARGET].to_numpy(float) - sub["sequence_anchor_pred"].to_numpy(float))
        dlinear_nlinear_best_abs = np.minimum(
            np.abs(sub[TARGET].to_numpy(float) - sub["dlinear_seq_pred"].to_numpy(float)),
            np.abs(sub[TARGET].to_numpy(float) - sub["nlinear_seq_pred"].to_numpy(float)),
        )
        for model_name, col in model_specs:
            rows.append(
                {
                    "dataset": "OPSD",
                    "zone": zone,
                    "regime": regime,
                    "model": model_name,
                    "anchor_model": anchor["anchor_model"],
                    "residual_shrinkage": w if model_name == "TDConv-inclusive GraphPatch residual" else 0.0,
                    "spike_threshold": spike_threshold,
                    **point_metrics(sub[TARGET], sub[col]),
                }
            )
        for model_name, col, baseline_name, base_abs in [
            (
                "TDConv-style sequence ridge",
                "tdconv_seq_pred",
                "Best DLinear/NLinear sequence ridge",
                dlinear_nlinear_best_abs,
            ),
            (
                "TDConv-inclusive GraphPatch residual",
                "tdconv_inclusive_graphpatch_pred",
                anchor["anchor_model"],
                anchor_abs,
            ),
        ]:
            model_abs = np.abs(sub[TARGET].to_numpy(float) - sub[col].to_numpy(float))
            paired_rows.append(
                {
                    "dataset": "OPSD",
                    "zone": zone,
                    "regime": regime,
                    "model": model_name,
                    "baseline": baseline_name,
                    **paired_sign_test(base_abs, model_abs),
                }
            )

    daily = test[["timestamp_utc", "zone", TARGET, "lag_1_24_abs", "spike_regime"]].copy()
    daily["dlinear_seq_pred"] = test["dlinear_seq_pred"]
    daily["nlinear_seq_pred"] = test["nlinear_seq_pred"]
    daily["tdconv_seq_pred"] = test["tdconv_seq_pred"]
    daily["anchor_model"] = anchor["anchor_model"]
    daily["sequence_anchor_pred"] = test["sequence_anchor_pred"]
    daily["graphpatch_residual"] = test["graphpatch_residual"]
    daily["tdconv_inclusive_graphpatch_pred"] = test["tdconv_inclusive_graphpatch_pred"]
    daily["residual_shrinkage"] = w
    diagnostics = {
        "dataset": "OPSD",
        "zone": zone,
        **anchor,
        **{f"residual_{k}": v for k, v in shrink.items()},
    }
    diagnostics.pop("anchor_col", None)
    return rows, paired_rows, diagnostics, daily


def run():
    tidy = prepare_patch_graph_frame()
    missing_lags = [lag for lag in range(1, LOOKBACK + 1) if f"price_lag_{lag}" not in tidy.columns]
    if missing_lags:
        lag_frame = pd.concat(
            {
                f"price_lag_{lag}": tidy.groupby("zone")[TARGET].shift(lag)
                for lag in missing_lags
            },
            axis=1,
        )
        tidy = pd.concat([tidy, lag_frame], axis=1)
    rows = []
    paired_rows = []
    diagnostics = []
    daily_parts = []
    for zone, zdf in tidy.groupby("zone", sort=True):
        zone_rows, zone_paired, zone_diag, zone_daily = evaluate_zone(zone, zdf)
        rows.extend(zone_rows)
        paired_rows.extend(zone_paired)
        diagnostics.append(zone_diag)
        daily_parts.append(zone_daily)

    summary = pd.DataFrame(rows)
    anchor = summary[summary["model"].str.startswith("Selected anchor:")][
        ["zone", "regime", "rmse", "mae"]
    ].rename(columns={"rmse": "anchor_rmse", "mae": "anchor_mae"})
    summary = summary.merge(anchor, on=["zone", "regime"], how="left")
    summary["rmse_improvement_pct_vs_anchor"] = (summary["anchor_rmse"] - summary["rmse"]) / summary["anchor_rmse"] * 100
    summary["mae_improvement_pct_vs_anchor"] = (summary["anchor_mae"] - summary["mae"]) / summary["anchor_mae"] * 100
    paired = pd.DataFrame(paired_rows)
    diag = pd.DataFrame(diagnostics)
    daily = pd.concat(daily_parts, ignore_index=True)
    summary.to_csv(RESULTS / "opsd_tdconv_sequence_anchor_graphpatch_price_summary.csv", index=False)
    paired.to_csv(RESULTS / "opsd_tdconv_sequence_anchor_graphpatch_price_paired_tests.csv", index=False)
    diag.to_csv(RESULTS / "opsd_tdconv_sequence_anchor_graphpatch_diagnostics.csv", index=False)
    daily.to_csv(RESULTS / "opsd_tdconv_sequence_anchor_graphpatch_price_daily.csv", index=False)
    return summary, paired, diag


def plot_tdconv_anchor(summary):
    spike = summary[summary["regime"] == "spike"].copy()
    zones = sorted(spike["zone"].unique())
    models = [
        "DLinear-style sequence ridge",
        "NLinear-style sequence ridge",
        "TDConv-style sequence ridge",
        "TDConv-inclusive GraphPatch residual",
    ]
    labels = {
        "DLinear-style sequence ridge": "DLinear",
        "NLinear-style sequence ridge": "NLinear",
        "TDConv-style sequence ridge": "TDConv",
        "TDConv-inclusive GraphPatch residual": "GraphPatch",
    }
    colors = {
        "DLinear-style sequence ridge": "#F58518",
        "NLinear-style sequence ridge": "#B279A2",
        "TDConv-style sequence ridge": "#4C78A8",
        "TDConv-inclusive GraphPatch residual": "#54A24B",
    }
    img = Image.new("RGB", (1640, 1120), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "TCN-family sequence comparator and GraphPatch residual", fill="#172033", font=font(40, True))
    d.text((88, 112), "TDConv is a trainable dilated-convolution ridge comparator reproduced without GPU-dependent libraries", fill="#5f6b7a", font=font(22))
    left, top, width, height = 150, 190, 1290, 690
    vals = []
    for zone in zones:
        for model in models:
            row = spike[(spike["zone"] == zone) & (spike["model"] == model)]
            if not row.empty:
                vals.append(float(row["rmse"].iloc[0]))
    ymax = max(vals) * 1.18
    for i in range(6):
        y = top + height - i * height / 5
        d.line((left, y, left + width, y), fill="#e5e7eb", width=1)
        d.text((left - 76, y - 14), f"{ymax * i / 5:.1f}", fill="#657385", font=font(20))
    group_w = width / len(zones)
    bar_w = min(54, group_w / 7)
    for zi, zone in enumerate(zones):
        gx = left + zi * group_w + 34
        for mi, model in enumerate(models):
            row = spike[(spike["zone"] == zone) & (spike["model"] == model)]
            if row.empty:
                continue
            val = float(row["rmse"].iloc[0])
            bh = val / ymax * height
            x0 = gx + mi * (bar_w + 8)
            y0 = top + height - bh
            d.rounded_rectangle((x0, y0, x0 + bar_w, top + height), radius=6, fill=colors[model])
            d.text((x0 - 4, y0 - 28), f"{val:.2f}", fill="#1f2937", font=font(18, True))
        d.text((gx + 80, top + height + 36), zone, fill="#1f2937", font=font(23, True), anchor="mm")
    d.line((left, top + height, left + width, top + height), fill="#94a3b8", width=2)
    lx, ly = 210, 955
    for i, model in enumerate(models):
        x = lx + i * 330
        d.rectangle((x, ly, x + 28, ly + 28), fill=colors[model])
        d.text((x + 42, ly - 4), labels[model], fill="#526070", font=font(21))
    d.text(
        (90, 1032),
        "Source: OPSD public day-ahead prices; final 20% chronological test split; TDConv and residual shrinkage use calibration-only selection.",
        fill="#6b7280",
        font=font(20),
    )
    out = FIGURES / "paper1_fig16_opsd_tdconv_anchor_graphpatch.png"
    img.save(out)
    return out


def main():
    summary, paired, diag = run()
    print(RESULTS / "opsd_tdconv_sequence_anchor_graphpatch_price_summary.csv")
    print(RESULTS / "opsd_tdconv_sequence_anchor_graphpatch_price_paired_tests.csv")
    print(RESULTS / "opsd_tdconv_sequence_anchor_graphpatch_diagnostics.csv")
    print(RESULTS / "opsd_tdconv_sequence_anchor_graphpatch_price_daily.csv")
    print(plot_tdconv_anchor(summary))
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
