from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from run_opsd_graph_temporal_price_ablation import (
    TARGET,
    RESULTS,
    FIGURES,
    fit_standardized_ridge,
    point_metrics,
    predict_standardized_ridge,
)
from run_opsd_deep_graph_patch_price_model import (
    PATCH_LAGS,
    GRAPH_LAGS,
    choose_blend_weight,
    fit_graphpatch_mlp,
    font,
    paired_sign_test,
    prepare_patch_graph_frame,
    predict_graphpatch_mlp,
    stable_zone_seed,
)


ROLLING_FOLDS = [
    ("R1", 0.50, 0.10, 0.10),
    ("R2", 0.60, 0.10, 0.10),
    ("R3", 0.70, 0.10, 0.10),
]


def feature_sets():
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
    return base_features, model_features


def clean_zone_frame(zdf, base_features, model_features):
    source_features = [col for col in model_features if col != "base_pred"]
    cols = list(dict.fromkeys(["timestamp_utc", "zone", TARGET, "hour"] + base_features + source_features))
    return zdf[cols].replace([np.inf, -np.inf], np.nan).dropna().sort_values("timestamp_utc").reset_index(drop=True)


def chronological_split(clean, train_frac, cal_frac, test_frac):
    n = len(clean)
    train_end = int(n * train_frac)
    cal_end = int(n * (train_frac + cal_frac))
    test_end = int(n * (train_frac + cal_frac + test_frac))
    return clean.iloc[:train_end].copy(), clean.iloc[train_end:cal_end].copy(), clean.iloc[cal_end:test_end].copy()


def fixed_split(clean):
    n = len(clean)
    train_end = int(n * 0.60)
    cal_end = int(n * 0.80)
    return clean.iloc[:train_end].copy(), clean.iloc[train_end:cal_end].copy(), clean.iloc[cal_end:].copy()


def add_base_predictions(train, cal, test, base_features):
    model = fit_standardized_ridge(train, base_features, l2=1e-4)
    for part in [train, cal, test]:
        part["base_pred"] = predict_standardized_ridge(part, base_features, model)
    return train, cal, test


def fit_residual_models(residual_train, residual_cal, model_features, seed):
    ridge_model = fit_standardized_ridge(residual_train, model_features, target_col="residual", l2=2e-3)
    mlp_model = fit_graphpatch_mlp(residual_train, residual_cal, model_features, "residual", seed)
    return ridge_model, mlp_model


def predict_graphpatch_parts(parts, ridge_model, mlp_model, model_features):
    for part in parts:
        part["graphpatch_ridge_residual"] = predict_standardized_ridge(part, model_features, ridge_model)
        part["graphpatch_mlp_residual"] = predict_graphpatch_mlp(part, mlp_model)
        part["graphpatch_ridge_pred"] = part["base_pred"] + part["graphpatch_ridge_residual"]
        part["graphpatch_mlp_pred"] = part["base_pred"] + part["graphpatch_mlp_residual"]


def apply_blend(cal, test):
    blend = choose_blend_weight(cal, TARGET, "graphpatch_ridge_pred", "graphpatch_mlp_pred")
    w = blend["blend_weight_mlp"]
    for part in [cal, test]:
        part["graphpatch_blend_pred"] = (1 - w) * part["graphpatch_ridge_pred"] + w * part["graphpatch_mlp_pred"]
    return blend


def evaluate_test(protocol, zone, fold, train, cal, test, blend, model_name):
    spike_threshold = float(np.nanquantile(train["lag_1_24_abs"], 0.90))
    test["spike_regime"] = test["lag_1_24_abs"] >= spike_threshold
    rows = []
    paired_rows = []
    model_specs = [("Local ridge", "base_pred"), (model_name, "graphpatch_blend_pred")]
    for regime, mask in [
        ("all", np.ones(len(test), dtype=bool)),
        ("spike", test["spike_regime"].to_numpy(bool)),
        ("non_spike", ~test["spike_regime"].to_numpy(bool)),
    ]:
        sub = test.loc[mask]
        base_abs = np.abs(sub[TARGET].to_numpy(float) - sub["base_pred"].to_numpy(float))
        for label, col in model_specs:
            rows.append(
                {
                    "dataset": "OPSD",
                    "protocol": protocol,
                    "fold": fold,
                    "zone": zone,
                    "regime": regime,
                    "model": label,
                    "blend_weight_mlp": blend["blend_weight_mlp"] if label != "Local ridge" else np.nan,
                    "cal_objective": blend["cal_objective"] if label != "Local ridge" else np.nan,
                    "spike_threshold": spike_threshold,
                    **point_metrics(sub[TARGET], sub[col]),
                }
            )
        model_abs = np.abs(sub[TARGET].to_numpy(float) - sub["graphpatch_blend_pred"].to_numpy(float))
        paired_rows.append(
            {
                "dataset": "OPSD",
                "protocol": protocol,
                "fold": fold,
                "zone": zone,
                "regime": regime,
                "model": model_name,
                "baseline": "Local ridge",
                **paired_sign_test(base_abs, model_abs),
            }
        )
    return rows, paired_rows


def add_improvement(summary):
    local = summary[summary["model"] == "Local ridge"][
        ["protocol", "fold", "zone", "regime", "rmse", "mae"]
    ].rename(columns={"rmse": "local_rmse", "mae": "local_mae"})
    summary = summary.merge(local, on=["protocol", "fold", "zone", "regime"], how="left")
    summary["rmse_improvement_pct"] = (summary["local_rmse"] - summary["rmse"]) / summary["local_rmse"] * 100
    summary["mae_improvement_pct"] = (summary["local_mae"] - summary["mae"]) / summary["local_mae"] * 100
    return summary


def rolling_origin(tidy, base_features, model_features):
    rows = []
    paired_rows = []
    for zone, zdf in tidy.groupby("zone", sort=True):
        clean = clean_zone_frame(zdf, base_features, model_features)
        for fold, train_frac, cal_frac, test_frac in ROLLING_FOLDS:
            train, cal, test = chronological_split(clean, train_frac, cal_frac, test_frac)
            if min(len(train), len(cal), len(test)) < 100:
                continue
            train, cal, test = add_base_predictions(train, cal, test, base_features)
            residual_train = train.copy()
            residual_cal = cal.copy()
            residual_train["residual"] = residual_train[TARGET] - residual_train["base_pred"]
            residual_cal["residual"] = residual_cal[TARGET] - residual_cal["base_pred"]
            ridge_model, mlp_model = fit_residual_models(
                residual_train,
                residual_cal,
                model_features,
                stable_zone_seed(f"{zone}_{fold}"),
            )
            predict_graphpatch_parts([cal, test], ridge_model, mlp_model, model_features)
            blend = apply_blend(cal, test)
            zone_rows, zone_paired = evaluate_test(
                "rolling_origin",
                zone,
                fold,
                train,
                cal,
                test,
                blend,
                "Rolling GraphPatch blend",
            )
            rows.extend(zone_rows)
            paired_rows.extend(zone_paired)
    return rows, paired_rows


def zone_holdout(tidy, base_features, model_features):
    rows = []
    paired_rows = []
    clean_by_zone = {
        zone: clean_zone_frame(zdf, base_features, model_features)
        for zone, zdf in tidy.groupby("zone", sort=True)
    }
    zones = sorted(clean_by_zone)
    for heldout in zones:
        source_train_parts = []
        source_cal_parts = []
        for zone in zones:
            if zone == heldout:
                continue
            train, cal, test = fixed_split(clean_by_zone[zone])
            train, cal, _ = add_base_predictions(train, cal, test, base_features)
            train["residual"] = train[TARGET] - train["base_pred"]
            cal["residual"] = cal[TARGET] - cal["base_pred"]
            source_train_parts.append(train)
            source_cal_parts.append(cal)
        residual_train = pd.concat(source_train_parts, ignore_index=True)
        residual_cal = pd.concat(source_cal_parts, ignore_index=True)
        ridge_model, mlp_model = fit_residual_models(
            residual_train,
            residual_cal,
            model_features,
            stable_zone_seed(f"holdout_{heldout}"),
        )
        train, cal, test = fixed_split(clean_by_zone[heldout])
        train, cal, test = add_base_predictions(train, cal, test, base_features)
        predict_graphpatch_parts([cal, test], ridge_model, mlp_model, model_features)
        blend = apply_blend(cal, test)
        zone_rows, zone_paired = evaluate_test(
            "zone_holdout",
            heldout,
            f"holdout_{heldout}",
            train,
            cal,
            test,
            blend,
            "LOZO GraphPatch blend",
        )
        rows.extend(zone_rows)
        paired_rows.extend(zone_paired)
    return rows, paired_rows


def build_aggregate(summary, paired):
    model_rows = summary[summary["model"] != "Local ridge"].copy()
    aggregate = (
        model_rows.groupby(["protocol", "regime", "model"], as_index=False)
        .agg(
            mean_rmse_improvement_pct=("rmse_improvement_pct", "mean"),
            median_rmse_improvement_pct=("rmse_improvement_pct", "median"),
            min_rmse_improvement_pct=("rmse_improvement_pct", "min"),
            mean_mae_improvement_pct=("mae_improvement_pct", "mean"),
            cases=("zone", "count"),
            positive_rmse_cases=("rmse_improvement_pct", lambda s: int((s > 0).sum())),
        )
    )
    paired_model = paired[paired["regime"] == "spike"].copy()
    paired_agg = (
        paired_model.groupby(["protocol", "model"], as_index=False)
        .agg(
            mean_spike_win_rate=("win_rate", "mean"),
            mean_abs_error_delta=("mean_abs_error_delta", "mean"),
            significant_better_cases=(
                "sign_test_p_approx",
                lambda s: int(((s < 0.05) & (paired_model.loc[s.index, "win_rate"] > 0.5)).sum()),
            ),
            paired_cases=("zone", "count"),
        )
    )
    return aggregate, paired_agg


def plot_rolling(summary):
    plot = summary[
        (summary["protocol"] == "rolling_origin")
        & (summary["regime"] == "spike")
        & (summary["model"] == "Rolling GraphPatch blend")
    ].copy()
    folds = [fold for fold in ["R1", "R2", "R3"] if fold in set(plot["fold"])]
    zones = sorted(plot["zone"].unique())
    colors = {"R1": "#4C78A8", "R2": "#54A24B", "R3": "#E45756"}
    img = Image.new("RGB", (1500, 1120), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "Rolling-origin robustness of GraphPatch on OPSD spike regimes", fill="#172033", font=font(38, True))
    d.text((88, 112), "Positive values mean lower spike RMSE than the local ridge baseline", fill="#5f6b7a", font=font(23))
    left, top, width, height = 150, 210, 1160, 620
    ymin, ymax = -15, 25
    zero_y = top + height - (0 - ymin) / (ymax - ymin) * height
    for i in range(5):
        val = ymin + i * (ymax - ymin) / 4
        y = top + height - (val - ymin) / (ymax - ymin) * height
        d.line((left, y, left + width, y), fill="#e5e7eb", width=1)
        d.text((left - 75, y - 14), f"{val:.0f}%", fill="#657385", font=font(20))
    d.line((left, zero_y, left + width, zero_y), fill="#172033", width=2)
    group_w = width / len(zones)
    bar_w = min(54, group_w / 6)
    for zi, zone in enumerate(zones):
        gx = left + zi * group_w + 52
        for fi, fold in enumerate(folds):
            row = plot[(plot["zone"] == zone) & (plot["fold"] == fold)]
            if row.empty:
                continue
            val = float(row["rmse_improvement_pct"].iloc[0])
            y = top + height - (val - ymin) / (ymax - ymin) * height
            x0 = gx + fi * (bar_w + 12)
            d.rounded_rectangle((x0, min(y, zero_y), x0 + bar_w, max(y, zero_y)), radius=5, fill=colors[fold])
            d.text((x0 - 6, y - 28 if val >= 0 else y + 8), f"{val:.1f}", fill="#1f2937", font=font(18, True))
        d.text((gx + 58, top + height + 35), zone, fill="#1f2937", font=font(23, True), anchor="mm")
    lx, ly = 420, 905
    for i, fold in enumerate(folds):
        x = lx + i * 190
        d.rectangle((x, ly, x + 30, ly + 30), fill=colors[fold])
        d.text((x + 44, ly - 2), fold, fill="#526070", font=font(22))
    d.text((90, 1010), "Source: OPSD rolling-origin train/calibration/test windows; spike threshold fitted only on each fold's training split.", fill="#6b7280", font=font(20))
    out = FIGURES / "paper1_fig12_opsd_graphpatch_rolling_origin.png"
    img.save(out)
    return out


def plot_zone_holdout(summary):
    plot = summary[
        (summary["protocol"] == "zone_holdout")
        & (summary["regime"] == "spike")
        & (summary["model"] == "LOZO GraphPatch blend")
    ].copy()
    zones = sorted(plot["zone"].unique())
    img = Image.new("RGB", (1500, 1120), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "Leave-one-zone-out GraphPatch transfer robustness", fill="#172033", font=font(40, True))
    d.text((88, 112), "Residual learner is trained on other zones; target-zone blend weight is selected on calibration data", fill="#5f6b7a", font=font(22))
    left, top, width, height = 150, 210, 1160, 620
    ymin, ymax = -20, 20
    zero_y = top + height - (0 - ymin) / (ymax - ymin) * height
    for i in range(5):
        val = ymin + i * (ymax - ymin) / 4
        y = top + height - (val - ymin) / (ymax - ymin) * height
        d.line((left, y, left + width, y), fill="#e5e7eb", width=1)
        d.text((left - 75, y - 14), f"{val:.0f}%", fill="#657385", font=font(20))
    d.line((left, zero_y, left + width, zero_y), fill="#172033", width=2)
    group_w = width / len(zones)
    for i, zone in enumerate(zones):
        row = plot[plot["zone"] == zone].iloc[0]
        val = float(row["rmse_improvement_pct"])
        x0 = left + i * group_w + group_w / 2 - 34
        y = top + height - (val - ymin) / (ymax - ymin) * height
        color = "#54A24B" if val >= 0 else "#E45756"
        d.rounded_rectangle((x0, min(y, zero_y), x0 + 68, max(y, zero_y)), radius=6, fill=color)
        d.text((x0 - 2, y - 30 if val >= 0 else y + 8), f"{val:.1f}%", fill="#1f2937", font=font(20, True))
        d.text((x0 + 34, top + height + 36), zone, fill="#1f2937", font=font(23, True), anchor="mm")
    d.text((90, 1010), "Source: OPSD leave-one-zone-out residual transfer; local ridge remains trained on the held-out zone's training window.", fill="#6b7280", font=font(20))
    out = FIGURES / "paper1_fig13_opsd_graphpatch_zone_holdout.png"
    img.save(out)
    return out


def main():
    tidy = prepare_patch_graph_frame()
    base_features, model_features = feature_sets()
    rolling_rows, rolling_paired = rolling_origin(tidy, base_features, model_features)
    holdout_rows, holdout_paired = zone_holdout(tidy, base_features, model_features)
    summary = add_improvement(pd.DataFrame(rolling_rows + holdout_rows))
    paired = pd.DataFrame(rolling_paired + holdout_paired)
    aggregate, paired_aggregate = build_aggregate(summary, paired)

    summary.to_csv(RESULTS / "opsd_graphpatch_robustness_summary.csv", index=False)
    paired.to_csv(RESULTS / "opsd_graphpatch_robustness_paired_tests.csv", index=False)
    aggregate.to_csv(RESULTS / "opsd_graphpatch_robustness_aggregate.csv", index=False)
    paired_aggregate.to_csv(RESULTS / "opsd_graphpatch_robustness_paired_aggregate.csv", index=False)
    print(RESULTS / "opsd_graphpatch_robustness_summary.csv")
    print(RESULTS / "opsd_graphpatch_robustness_paired_tests.csv")
    print(RESULTS / "opsd_graphpatch_robustness_aggregate.csv")
    print(RESULTS / "opsd_graphpatch_robustness_paired_aggregate.csv")
    print(plot_rolling(summary))
    print(plot_zone_holdout(summary))


if __name__ == "__main__":
    main()
