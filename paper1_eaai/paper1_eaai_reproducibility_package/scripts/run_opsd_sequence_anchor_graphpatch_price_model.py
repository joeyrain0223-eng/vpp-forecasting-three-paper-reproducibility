from pathlib import Path
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


SHRINKAGE_GRID = np.linspace(0, 1, 11)


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


def choose_sequence_anchor(cal):
    y = cal[TARGET].to_numpy(float)
    lag = cal["lag_1_24_abs"].to_numpy(float)
    choices = []
    for model, col in [
        ("DLinear-style sequence ridge", "dlinear_seq_pred"),
        ("NLinear-style sequence ridge", "nlinear_seq_pred"),
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
    for part in [train, cal, test]:
        dlinear, nlinear = predict_sequence_models(part, seq_models)
        part["dlinear_seq_pred"] = dlinear
        part["nlinear_seq_pred"] = nlinear

    anchor = choose_sequence_anchor(cal)
    for part in [train, cal, test]:
        part["sequence_anchor_pred"] = part[anchor["anchor_col"]]

    residual_train = train.copy()
    residual_train["residual"] = residual_train[TARGET] - residual_train["sequence_anchor_pred"]
    residual_model = fit_standardized_ridge(residual_train, model_features, target_col="residual", l2=3e-3)
    for part in [cal, test]:
        part["graphpatch_residual"] = predict_standardized_ridge(part, model_features, residual_model)

    shrink = choose_shrinkage(cal, "sequence_anchor_pred", "graphpatch_residual")
    w = shrink["residual_shrinkage"]
    test["sequence_graphpatch_pred"] = test["sequence_anchor_pred"] + w * test["graphpatch_residual"]
    cal["sequence_graphpatch_pred"] = cal["sequence_anchor_pred"] + w * cal["graphpatch_residual"]

    spike_threshold = float(np.nanquantile(train["lag_1_24_abs"], 0.90))
    test["spike_regime"] = test["lag_1_24_abs"] >= spike_threshold
    model_specs = [
        (anchor["anchor_model"], "sequence_anchor_pred"),
        ("Sequence-anchored GraphPatch residual", "sequence_graphpatch_pred"),
    ]
    rows = []
    paired_rows = []
    for regime, mask in [
        ("all", np.ones(len(test), dtype=bool)),
        ("spike", test["spike_regime"].to_numpy(bool)),
        ("non_spike", ~test["spike_regime"].to_numpy(bool)),
    ]:
        sub = test.loc[mask]
        base_abs = np.abs(sub[TARGET].to_numpy(float) - sub["sequence_anchor_pred"].to_numpy(float))
        for model_name, col in model_specs:
            rows.append(
                {
                    "dataset": "OPSD",
                    "zone": zone,
                    "regime": regime,
                    "model": model_name,
                    "anchor_model": anchor["anchor_model"],
                    "residual_shrinkage": w if model_name == "Sequence-anchored GraphPatch residual" else 0.0,
                    "spike_threshold": spike_threshold,
                    **point_metrics(sub[TARGET], sub[col]),
                }
            )
        model_abs = np.abs(sub[TARGET].to_numpy(float) - sub["sequence_graphpatch_pred"].to_numpy(float))
        paired_rows.append(
            {
                "dataset": "OPSD",
                "zone": zone,
                "regime": regime,
                "model": "Sequence-anchored GraphPatch residual",
                "baseline": anchor["anchor_model"],
                **paired_sign_test(base_abs, model_abs),
            }
        )

    daily = test[["timestamp_utc", "zone", TARGET, "lag_1_24_abs", "spike_regime"]].copy()
    daily["anchor_model"] = anchor["anchor_model"]
    daily["sequence_anchor_pred"] = test["sequence_anchor_pred"]
    daily["graphpatch_residual"] = test["graphpatch_residual"]
    daily["sequence_graphpatch_pred"] = test["sequence_graphpatch_pred"]
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
    anchor = summary[summary["model"] != "Sequence-anchored GraphPatch residual"][
        ["zone", "regime", "rmse", "mae"]
    ].rename(columns={"rmse": "anchor_rmse", "mae": "anchor_mae"})
    summary = summary.merge(anchor, on=["zone", "regime"], how="left")
    summary["rmse_improvement_pct_vs_anchor"] = (summary["anchor_rmse"] - summary["rmse"]) / summary["anchor_rmse"] * 100
    summary["mae_improvement_pct_vs_anchor"] = (summary["anchor_mae"] - summary["mae"]) / summary["anchor_mae"] * 100
    paired = pd.DataFrame(paired_rows)
    diag = pd.DataFrame(diagnostics)
    daily = pd.concat(daily_parts, ignore_index=True)
    summary.to_csv(RESULTS / "opsd_sequence_anchor_graphpatch_price_summary.csv", index=False)
    paired.to_csv(RESULTS / "opsd_sequence_anchor_graphpatch_price_paired_tests.csv", index=False)
    diag.to_csv(RESULTS / "opsd_sequence_anchor_graphpatch_diagnostics.csv", index=False)
    daily.to_csv(RESULTS / "opsd_sequence_anchor_graphpatch_price_daily.csv", index=False)
    return summary, paired, diag


def plot_sequence_anchor(summary):
    spike = summary[summary["regime"] == "spike"].copy()
    zones = sorted(spike["zone"].unique())
    models = ["anchor", "graphpatch"]
    colors = {"anchor": "#F58518", "graphpatch": "#54A24B"}
    img = Image.new("RGB", (1500, 1120), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "Sequence-anchored GraphPatch residual on OPSD spikes", fill="#172033", font=font(40, True))
    d.text((88, 112), "Residual shrinkage is selected on each zone's calibration split", fill="#5f6b7a", font=font(23))
    left, top, width, height = 150, 190, 1180, 690
    vals = []
    for zone in zones:
        zone_sp = spike[spike["zone"] == zone]
        vals.append(float(zone_sp[zone_sp["model"] != "Sequence-anchored GraphPatch residual"]["rmse"].iloc[0]))
        vals.append(float(zone_sp[zone_sp["model"] == "Sequence-anchored GraphPatch residual"]["rmse"].iloc[0]))
    ymax = max(vals) * 1.18
    for i in range(6):
        y = top + height - i * height / 5
        d.line((left, y, left + width, y), fill="#e5e7eb", width=1)
        d.text((left - 76, y - 14), f"{ymax * i / 5:.1f}", fill="#657385", font=font(20))
    group_w = width / len(zones)
    bar_w = min(70, group_w / 5)
    for zi, zone in enumerate(zones):
        zone_sp = spike[spike["zone"] == zone]
        anchor_row = zone_sp[zone_sp["model"] != "Sequence-anchored GraphPatch residual"].iloc[0]
        graph_row = zone_sp[zone_sp["model"] == "Sequence-anchored GraphPatch residual"].iloc[0]
        for mi, (key, row) in enumerate([("anchor", anchor_row), ("graphpatch", graph_row)]):
            val = float(row["rmse"])
            bh = val / ymax * height
            x0 = left + zi * group_w + 70 + mi * (bar_w + 16)
            y0 = top + height - bh
            d.rounded_rectangle((x0, y0, x0 + bar_w, top + height), radius=6, fill=colors[key])
            d.text((x0 - 4, y0 - 28), f"{val:.2f}", fill="#1f2937", font=font(18, True))
        imp = float(graph_row["rmse_improvement_pct_vs_anchor"])
        d.text((left + zi * group_w + 110, top + height + 36), zone, fill="#1f2937", font=font(23, True), anchor="mm")
        d.text((left + zi * group_w + 110, top + height + 70), f"{imp:+.2f}%", fill="#526070", font=font(18), anchor="mm")
    d.line((left, top + height, left + width, top + height), fill="#94a3b8", width=2)
    d.rectangle((360, 985, 392, 1013), fill=colors["anchor"])
    d.text((406, 979), "Best DLinear/NLinear anchor", fill="#526070", font=font(22))
    d.rectangle((760, 985, 792, 1013), fill=colors["graphpatch"])
    d.text((806, 979), "Sequence-anchored GraphPatch", fill="#526070", font=font(22))
    d.text((90, 1032), "Source: OPSD public day-ahead prices; final 20% chronological test split; anchor and shrinkage selected on calibration data.", fill="#6b7280", font=font(20))
    out = FIGURES / "paper1_fig15_opsd_sequence_anchor_graphpatch.png"
    img.save(out)
    return out


def main():
    summary, paired, diag = run()
    print(RESULTS / "opsd_sequence_anchor_graphpatch_price_summary.csv")
    print(RESULTS / "opsd_sequence_anchor_graphpatch_price_paired_tests.csv")
    print(RESULTS / "opsd_sequence_anchor_graphpatch_diagnostics.csv")
    print(RESULTS / "opsd_sequence_anchor_graphpatch_price_daily.csv")
    print(plot_sequence_anchor(summary))


if __name__ == "__main__":
    main()
