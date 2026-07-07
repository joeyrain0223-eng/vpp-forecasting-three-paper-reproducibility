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
    point_metrics,
    prepare_frame,
    split_zone,
    fit_standardized_ridge,
    predict_standardized_ridge,
)


RNG_SEED = 20260630
HIDDEN_UNITS = 96


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


def standardize_fit(frame, feature_cols):
    x = frame[feature_cols].to_numpy(float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize_apply(frame, feature_cols, mean, std):
    return (frame[feature_cols].to_numpy(float) - mean) / std


def fit_linear_from_matrix(x, y, l2=1e-3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    eye = np.eye(x.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + l2 * eye, x.T @ y)


def predict_linear_from_matrix(x, beta):
    x = np.asarray(x, dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    return x @ beta


def fit_random_hidden_residual(train, feature_cols, target_col, seed, hidden_units=HIDDEN_UNITS, l2=1e-3):
    mean, std = standardize_fit(train, feature_cols)
    x = standardize_apply(train, feature_cols, mean, std)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 1 / max(1, np.sqrt(x.shape[1])), size=(x.shape[1], hidden_units))
    bias = rng.normal(0, 0.1, size=hidden_units)
    hidden = np.tanh(x @ weights + bias)
    design = np.column_stack([x, hidden])
    beta = fit_linear_from_matrix(design, train[target_col].to_numpy(float), l2=l2)
    return {
        "mean": mean,
        "std": std,
        "weights": weights,
        "bias": bias,
        "beta": beta,
        "feature_cols": feature_cols,
    }


def predict_random_hidden_residual(frame, model):
    x = standardize_apply(frame, model["feature_cols"], model["mean"], model["std"])
    hidden = np.tanh(x @ model["weights"] + model["bias"])
    design = np.column_stack([x, hidden])
    return predict_linear_from_matrix(design, model["beta"])


def paired_sign_test(base_abs, model_abs):
    base_abs = np.asarray(base_abs, dtype=float)
    model_abs = np.asarray(model_abs, dtype=float)
    diff = base_abs - model_abs
    mask = np.isfinite(diff) & (diff != 0)
    diff = diff[mask]
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
    graph_features = [
        "base_pred",
        "price_lag_1",
        "price_lag_24",
        "price_lag_168",
        "lag_1_24_abs",
        "graph_mean_lag_1",
        "graph_mean_lag_24",
        "graph_mean_lag_168",
        "graph_std_lag_1",
        "graph_std_lag_24",
        "graph_std_lag_168",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month",
        "load_mw",
        "solar_mw",
        "wind_mw",
    ]
    source_graph_features = [col for col in graph_features if col != "base_pred"]
    cols = list(dict.fromkeys(["timestamp_utc", "zone", TARGET, "hour", "lag_1_24_abs"] + sorted(set(base_features + source_graph_features))))
    clean = zdf[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train, cal, test = split_zone(clean)

    base_model = fit_standardized_ridge(train, base_features, l2=1e-4)
    for part in [train, cal, test]:
        part["base_pred"] = predict_standardized_ridge(part, base_features, base_model)

    residual_train = train.copy()
    residual_train["residual"] = residual_train[TARGET] - residual_train["base_pred"]
    ridge_residual_model = fit_standardized_ridge(residual_train, graph_features, target_col="residual", l2=1e-3)
    elm_residual_model = fit_random_hidden_residual(
        residual_train,
        graph_features,
        "residual",
        seed=RNG_SEED + abs(hash(zone)) % 10000,
        hidden_units=HIDDEN_UNITS,
        l2=1e-2,
    )
    test["graph_ridge_pred"] = test["base_pred"] + predict_standardized_ridge(test, graph_features, ridge_residual_model)
    test["graph_elm_pred"] = test["base_pred"] + predict_random_hidden_residual(test, elm_residual_model)
    spike_threshold = float(np.nanquantile(train["lag_1_24_abs"], 0.90))
    test["spike_regime"] = test["lag_1_24_abs"] >= spike_threshold

    rows = []
    paired_rows = []
    model_specs = [
        ("Local ridge", "base_pred"),
        ("Graph residual ridge", "graph_ridge_pred"),
        ("Graph residual ELM", "graph_elm_pred"),
    ]
    regime_specs = [
        ("all", np.ones(len(test), dtype=bool)),
        ("spike", test["spike_regime"].to_numpy(bool)),
        ("non_spike", ~test["spike_regime"].to_numpy(bool)),
    ]
    base_abs_all = np.abs(test[TARGET].to_numpy(float) - test["base_pred"].to_numpy(float))
    for regime, mask in regime_specs:
        sub = test.loc[mask]
        for model_name, col in model_specs:
            rows.append(
                {
                    "dataset": "OPSD",
                    "zone": zone,
                    "regime": regime,
                    "model": model_name,
                    "spike_threshold": spike_threshold,
                    **point_metrics(sub[TARGET], sub[col]),
                }
            )
        base_abs = base_abs_all[mask]
        for model_name, col in model_specs[1:]:
            model_abs = np.abs(test.loc[mask, TARGET].to_numpy(float) - test.loc[mask, col].to_numpy(float))
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

    daily = test[["timestamp_utc", "zone", TARGET, "lag_1_24_abs", "spike_regime"]].copy()
    daily["local_ridge_pred"] = test["base_pred"]
    daily["graph_residual_ridge_pred"] = test["graph_ridge_pred"]
    daily["graph_residual_elm_pred"] = test["graph_elm_pred"]
    return rows, paired_rows, daily


def run():
    if not TIDY.exists():
        raise SystemExit(f"Missing processed OPSD file: {TIDY}")
    tidy = prepare_frame()
    rows = []
    paired_rows = []
    daily_parts = []
    for zone, zdf in tidy.groupby("zone", sort=True):
        zone_rows, zone_paired, zone_daily = evaluate_zone(zone, zdf)
        rows.extend(zone_rows)
        paired_rows.extend(zone_paired)
        daily_parts.append(zone_daily)

    summary = pd.DataFrame(rows)
    local = summary[summary["model"] == "Local ridge"][["zone", "regime", "rmse", "mae"]].rename(
        columns={"rmse": "local_rmse", "mae": "local_mae"}
    )
    summary = summary.merge(local, on=["zone", "regime"], how="left")
    summary["rmse_improvement_pct"] = (summary["local_rmse"] - summary["rmse"]) / summary["local_rmse"] * 100
    summary["mae_improvement_pct"] = (summary["local_mae"] - summary["mae"]) / summary["local_mae"] * 100
    paired = pd.DataFrame(paired_rows)
    daily = pd.concat(daily_parts, ignore_index=True)
    summary.to_csv(RESULTS / "opsd_nonlinear_price_baselines_summary.csv", index=False)
    paired.to_csv(RESULTS / "opsd_nonlinear_price_paired_tests.csv", index=False)
    daily.to_csv(RESULTS / "opsd_nonlinear_price_daily.csv", index=False)
    return summary, paired, daily


def plot_spike_nonlinear(summary):
    spike = summary[(summary["regime"] == "spike") & summary["model"].isin(["Local ridge", "Graph residual ridge", "Graph residual ELM"])].copy()
    zones = sorted(spike["zone"].unique())
    models = ["Local ridge", "Graph residual ridge", "Graph residual ELM"]
    colors = {"Local ridge": "#4C78A8", "Graph residual ridge": "#54A24B", "Graph residual ELM": "#E45756"}
    width, height = 1900, 1180
    left, top, right, bottom = 170, 190, 1710, 890
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "OPSD nonlinear graph-temporal price baselines", fill="#172033", font=font(44, True))
    d.text((88, 112), "Spike-regime RMSE; ELM is a random hidden-layer neural residual model", fill="#5f6b7a", font=font(24))
    ymax = float(spike["rmse"].max() * 1.18)
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        d.line((left, y, right, y), fill="#D7DCE2", width=1)
        d.text((left - 110, y - 13), f"{ymax * i / 5:.1f}", fill="#526070", font=font(22))
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    group_w = (right - left) / len(zones)
    bar_w = 78
    for idx, zone in enumerate(zones):
        base_x = left + idx * group_w + group_w / 2
        for j, model in enumerate(models):
            val = float(spike[(spike["zone"] == zone) & (spike["model"] == model)]["rmse"].iloc[0])
            x0 = base_x + (j - 1) * (bar_w + 16)
            x1 = x0 + bar_w
            y0 = bottom - val / ymax * (bottom - top)
            d.rounded_rectangle((x0, y0, x1, bottom), radius=8, fill=colors[model])
            d.text((x0 - 4, y0 - 29), f"{val:.2f}", fill="#1f2937", font=font(19, True))
        tw = d.textlength(zone, font=font(24, True))
        d.text((base_x - tw / 2, bottom + 35), zone, fill="#1f2937", font=font(24, True))
    x = 270
    for model in models:
        d.rectangle((x, 970, x + 32, 998), fill=colors[model])
        d.text((x + 44, 964), model, fill="#526070", font=font(22))
        x += 420
    d.text((90, 1060), "Source: OPSD public day-ahead prices; final 20% chronological test split; spike threshold fitted on training data.", fill="#6b7280", font=font(21))
    out = FIGURES / "paper1_fig8_opsd_nonlinear_spike_rmse.png"
    img.save(out)
    return out


def plot_sign_tests(paired):
    plot = paired[(paired["regime"] == "spike") & (paired["model"].isin(["Graph residual ridge", "Graph residual ELM"]))].copy()
    zones = sorted(plot["zone"].unique())
    width, height = 1900, 1180
    left, top, right, bottom = 170, 190, 1710, 890
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "Paired error-reduction evidence on OPSD spike regimes", fill="#172033", font=font(43, True))
    d.text((88, 112), "Win rate means the model has lower absolute error than local ridge at the same hour", fill="#5f6b7a", font=font(24))
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        d.line((left, y, right, y), fill="#D7DCE2", width=1)
        d.text((left - 92, y - 13), f"{i * 20}%", fill="#526070", font=font(22))
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    d.line((left, bottom - 0.5 * (bottom - top), right, bottom - 0.5 * (bottom - top)), fill="#111827", width=3)
    colors = {"Graph residual ridge": "#54A24B", "Graph residual ELM": "#E45756"}
    group_w = (right - left) / len(zones)
    bar_w = 105
    for idx, zone in enumerate(zones):
        base_x = left + idx * group_w + group_w / 2
        for j, model in enumerate(["Graph residual ridge", "Graph residual ELM"]):
            row = plot[(plot["zone"] == zone) & (plot["model"] == model)].iloc[0]
            val = float(row["win_rate"]) * 100
            x0 = base_x + (j - 0.5) * (bar_w + 20)
            x1 = x0 + bar_w
            y0 = bottom - val / 100 * (bottom - top)
            d.rounded_rectangle((x0, y0, x1, bottom), radius=8, fill=colors[model])
            p = float(row["sign_test_p_approx"])
            label = f"{val:.1f}%"
            d.text((x0 - 4, y0 - 31), label, fill="#1f2937", font=font(20, True))
            if p < 0.05:
                d.text((x0 + 28, y0 - 55), "*", fill="#111827", font=font(26, True))
        tw = d.textlength(zone, font=font(24, True))
        d.text((base_x - tw / 2, bottom + 35), zone, fill="#1f2937", font=font(24, True))
    d.rectangle((430, 970, 462, 998), fill=colors["Graph residual ridge"])
    d.text((476, 964), "Graph residual ridge", fill="#526070", font=font(22))
    d.rectangle((850, 970, 882, 998), fill=colors["Graph residual ELM"])
    d.text((896, 964), "Graph residual ELM", fill="#526070", font=font(22))
    d.text((90, 1060), "* Approximate two-sided paired sign test p < 0.05 against local ridge.", fill="#6b7280", font=font(21))
    out = FIGURES / "paper1_fig9_opsd_paired_spike_win_rate.png"
    img.save(out)
    return out


def main():
    summary, paired, _ = run()
    fig1 = plot_spike_nonlinear(summary)
    fig2 = plot_sign_tests(paired)
    print(RESULTS / "opsd_nonlinear_price_baselines_summary.csv")
    print(RESULTS / "opsd_nonlinear_price_paired_tests.csv")
    print(RESULTS / "opsd_nonlinear_price_daily.csv")
    print(fig1)
    print(fig2)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
