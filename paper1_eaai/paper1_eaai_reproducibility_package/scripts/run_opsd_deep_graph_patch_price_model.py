from pathlib import Path
from math import erfc, sqrt

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_opsd_graph_temporal_price_ablation import (
    TARGET,
    TIDY,
    RESULTS,
    FIGURES,
    ALPHA,
    conformal_quantile,
    fit_standardized_ridge,
    interval_metrics,
    point_metrics,
    predict_standardized_ridge,
    split_zone,
)


RNG_SEED = 20260701
PATCH_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
GRAPH_LAGS = [1, 3, 6, 24, 72, 168]
HIDDEN_1 = 56
HIDDEN_2 = 24


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


def stable_zone_seed(zone):
    return RNG_SEED + sum((i + 1) * ord(ch) for i, ch in enumerate(str(zone)))


def prepare_patch_graph_frame():
    if not TIDY.exists():
        raise SystemExit(f"Missing processed OPSD file: {TIDY}")
    tidy = pd.read_csv(TIDY, parse_dates=["timestamp_utc"])
    tidy = tidy.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    for lag in PATCH_LAGS:
        tidy[f"price_lag_{lag}"] = tidy.groupby("zone")[TARGET].shift(lag)
    tidy["lag_1_24_abs"] = (tidy["price_lag_1"] - tidy["price_lag_24"]).abs()
    tidy["lag_1_168_abs"] = (tidy["price_lag_1"] - tidy["price_lag_168"]).abs()
    tidy["patch_mean_24"] = tidy.groupby("zone")[TARGET].shift(1).rolling(24, min_periods=12).mean().reset_index(level=0, drop=True)
    tidy["patch_std_24"] = tidy.groupby("zone")[TARGET].shift(1).rolling(24, min_periods=12).std().reset_index(level=0, drop=True)
    tidy["patch_mean_168"] = tidy.groupby("zone")[TARGET].shift(1).rolling(168, min_periods=72).mean().reset_index(level=0, drop=True)
    tidy["patch_std_168"] = tidy.groupby("zone")[TARGET].shift(1).rolling(168, min_periods=72).std().reset_index(level=0, drop=True)

    pivot = tidy.pivot_table(index="timestamp_utc", columns="zone", values=TARGET, aggfunc="mean").sort_index()
    for lag in GRAPH_LAGS:
        shifted = pivot.shift(lag)
        cols = []
        for zone in pivot.columns:
            others = [col for col in pivot.columns if col != zone]
            tmp = shifted[others]
            cols.append(tmp.mean(axis=1).rename(f"graph_mean_lag_{lag}_{zone}"))
            cols.append(tmp.std(axis=1).fillna(0).rename(f"graph_std_lag_{lag}_{zone}"))
        graph = pd.concat(cols, axis=1)
        tidy = tidy.merge(graph.reset_index(), on="timestamp_utc", how="left")
        tidy[f"graph_mean_lag_{lag}"] = np.nan
        tidy[f"graph_std_lag_{lag}"] = np.nan
        for zone in sorted(tidy["zone"].dropna().unique()):
            mask = tidy["zone"] == zone
            tidy.loc[mask, f"graph_mean_lag_{lag}"] = tidy.loc[mask, f"graph_mean_lag_{lag}_{zone}"]
            tidy.loc[mask, f"graph_std_lag_{lag}"] = tidy.loc[mask, f"graph_std_lag_{lag}_{zone}"]
        tidy = tidy.drop(
            columns=[
                c
                for c in tidy.columns
                if c.startswith(f"graph_mean_lag_{lag}_") or c.startswith(f"graph_std_lag_{lag}_")
            ]
        )
    return tidy


def standardize_fit(x):
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize_apply(x, mean, std):
    return (x - mean) / std


def tanh_forward(x, params):
    h1_pre = x @ params["w1"] + params["b1"]
    h1 = np.tanh(h1_pre)
    h2_pre = h1 @ params["w2"] + params["b2"]
    h2 = np.tanh(h2_pre)
    out = h2 @ params["w3"] + params["b3"]
    return out.ravel(), (x, h1, h2)


def fit_graphpatch_mlp(train, cal, feature_cols, target_col, seed):
    x_train_raw = train[feature_cols].to_numpy(float)
    y_train_raw = train[target_col].to_numpy(float)
    x_cal_raw = cal[feature_cols].to_numpy(float)
    y_cal_raw = cal[target_col].to_numpy(float)
    x_mean, x_std = standardize_fit(x_train_raw)
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw))
    if y_std < 1e-8:
        y_std = 1.0
    x_train = standardize_apply(x_train_raw, x_mean, x_std)
    x_cal = standardize_apply(x_cal_raw, x_mean, x_std)
    y_train = (y_train_raw - y_mean) / y_std
    y_cal = (y_cal_raw - y_mean) / y_std

    rng = np.random.default_rng(seed)
    d = x_train.shape[1]
    params = {
        "w1": rng.normal(0, np.sqrt(2 / max(1, d)), size=(d, HIDDEN_1)),
        "b1": np.zeros(HIDDEN_1),
        "w2": rng.normal(0, np.sqrt(2 / HIDDEN_1), size=(HIDDEN_1, HIDDEN_2)),
        "b2": np.zeros(HIDDEN_2),
        "w3": rng.normal(0, np.sqrt(2 / HIDDEN_2), size=(HIDDEN_2, 1)),
        "b3": np.zeros(1),
    }
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    second = {k: np.zeros_like(v) for k, v in params.items()}
    best = {k: v.copy() for k, v in params.items()}
    best_loss = np.inf
    best_epoch = 0
    lr = 0.0025
    l2 = 2e-4
    batch_size = min(768, len(x_train))
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0

    for epoch in range(420):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            pred, cache = tanh_forward(xb, params)
            x0, h1, h2 = cache
            n = len(xb)
            grad_out = (2 / n) * (pred - yb)
            grads = {
                "w3": h2.T @ grad_out[:, None] + l2 * params["w3"],
                "b3": np.array([np.sum(grad_out)]),
            }
            dh2 = grad_out[:, None] @ params["w3"].T
            dz2 = dh2 * (1 - h2**2)
            grads["w2"] = h1.T @ dz2 + l2 * params["w2"]
            grads["b2"] = np.sum(dz2, axis=0)
            dh1 = dz2 @ params["w2"].T
            dz1 = dh1 * (1 - h1**2)
            grads["w1"] = x0.T @ dz1 + l2 * params["w1"]
            grads["b1"] = np.sum(dz1, axis=0)
            step += 1
            for key in params:
                velocity[key] = beta1 * velocity[key] + (1 - beta1) * grads[key]
                second[key] = beta2 * second[key] + (1 - beta2) * (grads[key] ** 2)
                m_hat = velocity[key] / (1 - beta1**step)
                v_hat = second[key] / (1 - beta2**step)
                params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        cal_pred, _ = tanh_forward(x_cal, params)
        cal_loss = float(np.mean((cal_pred - y_cal) ** 2))
        if cal_loss < best_loss:
            best_loss = cal_loss
            best_epoch = epoch
            best = {k: v.copy() for k, v in params.items()}
        if epoch - best_epoch >= 45:
            break

    return {
        "params": best,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "feature_cols": feature_cols,
        "best_epoch": best_epoch,
        "best_cal_loss": best_loss,
    }


def predict_graphpatch_mlp(frame, model):
    x = frame[model["feature_cols"]].to_numpy(float)
    x = standardize_apply(x, model["x_mean"], model["x_std"])
    pred, _ = tanh_forward(x, model["params"])
    return pred * model["y_std"] + model["y_mean"]


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


def choose_blend_weight(cal, target_col, ridge_col, mlp_col):
    weights = np.linspace(0, 1, 11)
    spike_threshold = float(np.nanquantile(cal["lag_1_24_abs"], 0.90))
    spike_mask = cal["lag_1_24_abs"].to_numpy(float) >= spike_threshold
    best = {"blend_weight_mlp": 0.0, "cal_objective": np.inf, "cal_all_rmse": np.inf, "cal_spike_rmse": np.inf}
    y = cal[target_col].to_numpy(float)
    for w in weights:
        pred = (1 - w) * cal[ridge_col].to_numpy(float) + w * cal[mlp_col].to_numpy(float)
        all_rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        if np.any(spike_mask):
            spike_rmse = float(np.sqrt(np.mean((y[spike_mask] - pred[spike_mask]) ** 2)))
        else:
            spike_rmse = all_rmse
        objective = 0.55 * all_rmse + 0.45 * spike_rmse
        if objective < best["cal_objective"]:
            best = {
                "blend_weight_mlp": float(w),
                "cal_objective": objective,
                "cal_all_rmse": all_rmse,
                "cal_spike_rmse": spike_rmse,
            }
    return best


def evaluate_zone(zone, zdf):
    base_features = [
        "price_lag_1",
        "price_lag_24",
        "price_lag_168",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month",
        "load_mw",
        "solar_mw",
        "wind_mw",
    ]
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
                "base_pred",
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
    source_features = [col for col in model_features if col != "base_pred"]
    cols = list(dict.fromkeys(["timestamp_utc", "zone", TARGET, "hour"] + base_features + source_features))
    clean = zdf[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train, cal, test = split_zone(clean)

    base_model = fit_standardized_ridge(train, base_features, l2=1e-4)
    for part in [train, cal, test]:
        part["base_pred"] = predict_standardized_ridge(part, base_features, base_model)

    residual_train = train.copy()
    residual_cal = cal.copy()
    residual_train["residual"] = residual_train[TARGET] - residual_train["base_pred"]
    residual_cal["residual"] = residual_cal[TARGET] - residual_cal["base_pred"]
    ridge_model = fit_standardized_ridge(residual_train, model_features, target_col="residual", l2=2e-3)
    mlp_model = fit_graphpatch_mlp(residual_train, residual_cal, model_features, "residual", stable_zone_seed(zone))
    for part in [cal, test]:
        part["graphpatch_ridge_residual"] = predict_standardized_ridge(part, model_features, ridge_model)
        part["graphpatch_mlp_residual"] = predict_graphpatch_mlp(part, mlp_model)
        part["graphpatch_ridge_pred"] = part["base_pred"] + part["graphpatch_ridge_residual"]
        part["graphpatch_mlp_pred"] = part["base_pred"] + part["graphpatch_mlp_residual"]

    blend = choose_blend_weight(cal, TARGET, "graphpatch_ridge_pred", "graphpatch_mlp_pred")
    w = blend["blend_weight_mlp"]
    cal["graphpatch_blend_pred"] = (1 - w) * cal["graphpatch_ridge_pred"] + w * cal["graphpatch_mlp_pred"]
    test["graphpatch_blend_pred"] = (1 - w) * test["graphpatch_ridge_pred"] + w * test["graphpatch_mlp_pred"]

    spike_threshold = float(np.nanquantile(train["lag_1_24_abs"], 0.90))
    test["spike_regime"] = test["lag_1_24_abs"] >= spike_threshold
    rows = []
    paired_rows = []
    model_specs = [
        ("Local ridge", "base_pred"),
        ("GraphPatch ridge", "graphpatch_ridge_pred"),
        ("GraphPatch MLP", "graphpatch_mlp_pred"),
        ("Calibrated GraphPatch blend", "graphpatch_blend_pred"),
    ]
    for regime, mask in [
        ("all", np.ones(len(test), dtype=bool)),
        ("spike", test["spike_regime"].to_numpy(bool)),
        ("non_spike", ~test["spike_regime"].to_numpy(bool)),
    ]:
        sub = test.loc[mask]
        base_abs = np.abs(sub[TARGET].to_numpy(float) - sub["base_pred"].to_numpy(float))
        for model_name, col in model_specs:
            rows.append(
                {
                    "dataset": "OPSD",
                    "zone": zone,
                    "regime": regime,
                    "model": model_name,
                    "spike_threshold": spike_threshold,
                    "blend_weight_mlp": w if model_name == "Calibrated GraphPatch blend" else np.nan,
                    "mlp_best_epoch": mlp_model["best_epoch"] if model_name == "GraphPatch MLP" else np.nan,
                    **point_metrics(sub[TARGET], sub[col]),
                }
            )
        for model_name, col in model_specs[1:]:
            model_abs = np.abs(sub[TARGET].to_numpy(float) - sub[col].to_numpy(float))
            paired_rows.append(
                {
                    "dataset": "OPSD",
                    "zone": zone,
                    "regime": regime,
                    "model": model_name,
                    "baseline": "Local ridge",
                    **paired_sign_test(base_abs, model_abs),
                }
            )

    q = conformal_quantile(np.abs(cal[TARGET].to_numpy(float) - cal["graphpatch_blend_pred"].to_numpy(float)), ALPHA)
    lower = test["graphpatch_blend_pred"].to_numpy(float) - q
    upper = test["graphpatch_blend_pred"].to_numpy(float) + q
    interval = {
        "dataset": "OPSD",
        "zone": zone,
        "model": "Calibrated GraphPatch blend conformal",
        "target_coverage": 1 - ALPHA,
        "blend_weight_mlp": w,
        "cal_objective": blend["cal_objective"],
        "cal_all_rmse": blend["cal_all_rmse"],
        "cal_spike_rmse": blend["cal_spike_rmse"],
        "calibration_width": 2 * q,
        **interval_metrics(test[TARGET], test["graphpatch_blend_pred"], lower, upper, ALPHA),
    }
    daily = test[
        [
            "timestamp_utc",
            "zone",
            TARGET,
            "lag_1_24_abs",
            "spike_regime",
            "base_pred",
            "graphpatch_ridge_pred",
            "graphpatch_mlp_pred",
            "graphpatch_blend_pred",
        ]
    ].copy()
    daily["graphpatch_lower"] = lower
    daily["graphpatch_upper"] = upper
    daily["blend_weight_mlp"] = w
    return rows, paired_rows, interval, daily


def run():
    tidy = prepare_patch_graph_frame()
    rows = []
    paired_rows = []
    intervals = []
    daily_parts = []
    for zone, zdf in tidy.groupby("zone", sort=True):
        zone_rows, zone_paired, zone_interval, zone_daily = evaluate_zone(zone, zdf)
        rows.extend(zone_rows)
        paired_rows.extend(zone_paired)
        intervals.append(zone_interval)
        daily_parts.append(zone_daily)

    summary = pd.DataFrame(rows)
    local = summary[summary["model"] == "Local ridge"][["zone", "regime", "rmse", "mae"]].rename(
        columns={"rmse": "local_rmse", "mae": "local_mae"}
    )
    summary = summary.merge(local, on=["zone", "regime"], how="left")
    summary["rmse_improvement_pct"] = (summary["local_rmse"] - summary["rmse"]) / summary["local_rmse"] * 100
    summary["mae_improvement_pct"] = (summary["local_mae"] - summary["mae"]) / summary["local_mae"] * 100
    paired = pd.DataFrame(paired_rows)
    interval = pd.DataFrame(intervals)
    daily = pd.concat(daily_parts, ignore_index=True)
    summary.to_csv(RESULTS / "opsd_deep_graphpatch_price_summary.csv", index=False)
    paired.to_csv(RESULTS / "opsd_deep_graphpatch_price_paired_tests.csv", index=False)
    interval.to_csv(RESULTS / "opsd_deep_graphpatch_price_conformal_summary.csv", index=False)
    daily.to_csv(RESULTS / "opsd_deep_graphpatch_price_daily.csv", index=False)
    return summary, paired, interval


def plot_spike_rmse(summary):
    plot = summary[
        (summary["regime"] == "spike")
        & summary["model"].isin(["Local ridge", "GraphPatch ridge", "GraphPatch MLP", "Calibrated GraphPatch blend"])
    ].copy()
    zones = sorted(plot["zone"].unique())
    models = ["Local ridge", "GraphPatch ridge", "GraphPatch MLP", "Calibrated GraphPatch blend"]
    labels = {
        "Local ridge": "Local",
        "GraphPatch ridge": "Patch ridge",
        "GraphPatch MLP": "Patch MLP",
        "Calibrated GraphPatch blend": "Blend",
    }
    colors = {
        "Local ridge": "#4C78A8",
        "GraphPatch ridge": "#54A24B",
        "GraphPatch MLP": "#E45756",
        "Calibrated GraphPatch blend": "#8F63B8",
    }
    img = Image.new("RGB", (1500, 1120), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "Deep GraphPatch residual price model on OPSD spike regimes", fill="#172033", font=font(40, True))
    d.text((88, 112), "Two-layer residual MLP with calibration-selected ridge/MLP blending", fill="#5f6b7a", font=font(23))
    left, top, width, height = 150, 190, 1180, 690
    vals = [float(plot[(plot["zone"] == z) & (plot["model"] == m)]["rmse"].iloc[0]) for z in zones for m in models]
    ymax = max(vals) * 1.18
    for i in range(6):
        y = top + height - i * height / 5
        d.line((left, y, left + width, y), fill="#e5e7eb", width=1)
        d.text((left - 76, y - 14), f"{ymax * i / 5:.1f}", fill="#657385", font=font(20))
    group_w = width / len(zones)
    bar_w = min(52, group_w / 7)
    for zi, zone in enumerate(zones):
        gx = left + zi * group_w + 34
        for mi, model in enumerate(models):
            row = plot[(plot["zone"] == zone) & (plot["model"] == model)]
            val = float(row["rmse"].iloc[0])
            bh = val / ymax * height
            x0 = gx + mi * (bar_w + 8)
            y0 = top + height - bh
            d.rounded_rectangle((x0, y0, x0 + bar_w, top + height), radius=6, fill=colors[model])
            d.text((x0 - 4, y0 - 28), f"{val:.2f}", fill="#1f2937", font=font(18, True))
        d.text((gx + 45, top + height + 34), zone, fill="#1f2937", font=font(23, True), anchor="mm")
    d.line((left, top + height, left + width, top + height), fill="#94a3b8", width=2)
    lx, ly = 260, 945
    for i, model in enumerate(models):
        x = lx + i * 275
        d.rectangle((x, ly, x + 28, ly + 28), fill=colors[model])
        d.text((x + 42, ly - 4), labels[model], fill="#526070", font=font(20))
    d.text((90, 1032), "Source: OPSD public day-ahead prices; final 20% chronological test split; blend weight selected on calibration data.", fill="#6b7280", font=font(20))
    out = FIGURES / "paper1_fig10_opsd_deep_graphpatch_spike_rmse.png"
    img.save(out)
    return out


def plot_interval_and_blend(interval):
    zones = list(interval["zone"])
    img = Image.new("RGB", (1500, 1120), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "Deep GraphPatch conformal coverage and calibration-selected blend", fill="#172033", font=font(40, True))
    d.text((88, 112), "Coverage target is 90%; purple line reports selected MLP residual weight", fill="#5f6b7a", font=font(23))
    left, top, width, height = 150, 210, 1160, 620
    for i in range(6):
        y = top + height - i * height / 5
        d.line((left, y, left + width, y), fill="#e5e7eb", width=1)
        d.text((left - 78, y - 14), f"{20 * i}%", fill="#657385", font=font(20))
    group_w = width / len(zones)
    picps = interval["picp"].to_numpy(float)
    weights = interval["blend_weight_mlp"].to_numpy(float)
    for i, zone in enumerate(zones):
        gx = left + i * group_w + group_w / 2
        cov_h = picps[i] * height
        w_h = weights[i] * height
        d.rounded_rectangle((gx - 54, top + height - cov_h, gx - 12, top + height), radius=6, fill="#4C78A8")
        d.rounded_rectangle((gx + 12, top + height - w_h, gx + 54, top + height), radius=6, fill="#8F63B8")
        d.text((gx - 70, top + height - cov_h - 30), f"{picps[i]*100:.1f}%", fill="#1f2937", font=font(18, True))
        d.text((gx + 2, top + height - w_h - 30), f"{weights[i]*100:.0f}%", fill="#1f2937", font=font(18, True))
        d.text((gx, top + height + 36), zone, fill="#1f2937", font=font(23, True), anchor="mm")
    target_y = top + height - 0.90 * height
    d.line((left, target_y, left + width, target_y), fill="#172033", width=2)
    d.text((left + width + 10, target_y - 14), "90%", fill="#172033", font=font(20, True))
    d.line((left, top + height, left + width, top + height), fill="#94a3b8", width=2)
    d.rectangle((430, 900, 462, 928), fill="#4C78A8")
    d.text((476, 894), "PICP", fill="#526070", font=font(22))
    d.rectangle((650, 900, 682, 928), fill="#8F63B8")
    d.text((696, 894), "Selected MLP weight", fill="#526070", font=font(22))
    d.text((90, 1012), "Source: OPSD calibration/test split; intervals use split conformal residual quantile around calibrated GraphPatch blend.", fill="#6b7280", font=font(20))
    out = FIGURES / "paper1_fig11_opsd_deep_graphpatch_conformal_blend.png"
    img.save(out)
    return out


def main():
    summary, paired, interval = run()
    print(RESULTS / "opsd_deep_graphpatch_price_summary.csv")
    print(RESULTS / "opsd_deep_graphpatch_price_paired_tests.csv")
    print(RESULTS / "opsd_deep_graphpatch_price_conformal_summary.csv")
    print(plot_spike_rmse(summary))
    print(plot_interval_and_blend(interval))


if __name__ == "__main__":
    main()
