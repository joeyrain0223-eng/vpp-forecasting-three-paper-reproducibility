from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_uci_ssl_representation_prototype import (
    ADAPT_28_START,
    HISTORY,
    RESULTS,
    FIGURES,
    TEST_START,
    build_scales,
    calendar_features,
    client_supervised_arrays,
    collect_source_windows,
    encode,
    feature_matrix,
    fit_adapter,
    fit_masked_reconstruction_basis,
    fit_ridge,
    load_data,
    metrics,
    predict_adapter,
    predict_ridge,
    source_training_matrix,
)


ADAPT_DAYS = [1, 3, 7, 28]


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


def lag_calendar_features(windows, times):
    lag_features = np.column_stack([windows[:, -1], windows[:, -24], windows[:, 0]])
    return np.column_stack([lag_features, calendar_features(times)])


def evaluate_target_linear(windows, y_norm, times, test_mask, days):
    start = TEST_START - pd.Timedelta(days=days)
    adapt_mask = (times >= start) & (times < TEST_START)
    if adapt_mask.sum() < max(12, int(24 * days * 0.6)):
        return None
    train_x = lag_calendar_features(windows[adapt_mask], times[adapt_mask])
    train_y = y_norm[adapt_mask]
    beta = fit_ridge(train_x, train_y, l2=1e-3)
    test_x = lag_calendar_features(windows[test_mask], times[test_mask])
    return predict_ridge(test_x, beta)


def evaluate_client(hourly, client, scale, mean_vec, components, source_beta, source_centroid):
    windows, y_norm, times = client_supervised_arrays(hourly, client, scale)
    test_mask = times >= TEST_START
    if test_mask.sum() < 24 * 30:
        return [], None

    test_y_norm = y_norm[test_mask]
    test_times = times[test_mask]
    test_windows = windows[test_mask]
    x_all = feature_matrix(windows, times, mean_vec, components, include_lags=True)
    source_pred_norm_all = predict_ridge(x_all, source_beta)
    source_pred_norm = source_pred_norm_all[test_mask]

    rows = []
    for name, pred_norm, protocol, days in [
        ("Seasonal-24h", test_windows[:, -24], "zero-label seasonal", 0),
        ("Seasonal-168h", test_windows[:, 0], "zero-label seasonal", 0),
        ("SSL-MR-lag-source-head", source_pred_norm, "zero-label source representation", 0),
    ]:
        rows.append(
            {
                "dataset": "UCI Electricity Load Diagrams 2011-2014",
                "target_client": client,
                "model": name,
                "protocol": protocol,
                "adapt_days": days,
                **metrics(test_y_norm * scale, pred_norm * scale),
            }
        )

    for days in ADAPT_DAYS:
        start = TEST_START - pd.Timedelta(days=days)
        adapt_mask = (times >= start) & (times < TEST_START)
        if adapt_mask.sum() < max(12, int(24 * days * 0.6)):
            continue
        adapt_pred_norm = source_pred_norm_all[adapt_mask]
        adapter_beta = fit_adapter(adapt_pred_norm, y_norm[adapt_mask], times[adapt_mask])
        adapter_pred_norm = predict_adapter(source_pred_norm, test_times, adapter_beta)
        rows.append(
            {
                "dataset": "UCI Electricity Load Diagrams 2011-2014",
                "target_client": client,
                "model": f"SSL-MR-lag+adapter-{days}d",
                "protocol": "frozen source representation with target adapter",
                "adapt_days": days,
                **metrics(test_y_norm * scale, adapter_pred_norm * scale),
            }
        )

        target_pred_norm = evaluate_target_linear(windows, y_norm, times, test_mask, days)
        if target_pred_norm is not None:
            rows.append(
                {
                    "dataset": "UCI Electricity Load Diagrams 2011-2014",
                    "target_client": client,
                    "model": f"Target-linear-{days}d",
                    "protocol": "target-only lag-calendar ridge",
                    "adapt_days": days,
                    **metrics(test_y_norm * scale, target_pred_norm * scale),
                }
            )

    adapt_28_mask = (times >= ADAPT_28_START) & (times < TEST_START)
    adapt_latent = encode(windows[adapt_28_mask], mean_vec, components)
    target_centroid = adapt_latent.mean(axis=0)
    source_distance = float(np.linalg.norm(target_centroid - source_centroid))
    target_scale = float(np.nanmedian(hourly.loc[(hourly.index >= ADAPT_28_START) & (hourly.index < TEST_START), client]))
    source_rmse = next(row["rmse"] for row in rows if row["model"] == "SSL-MR-lag-source-head")
    adapter_28 = next((row["rmse"] for row in rows if row["model"] == "SSL-MR-lag+adapter-28d"), np.nan)
    target_28 = next((row["rmse"] for row in rows if row["model"] == "Target-linear-28d"), np.nan)
    diagnostic = {
        "target_client": client,
        "source_target_latent_distance": source_distance,
        "target_28d_median_load": target_scale,
        "ssl_source_head_rmse": source_rmse,
        "ssl_adapter_28d_rmse": adapter_28,
        "target_linear_28d_rmse": target_28,
        "adapter_gain_vs_source_head": source_rmse - adapter_28,
        "adapter_gain_pct": (source_rmse - adapter_28) / source_rmse * 100,
    }
    return rows, diagnostic


def summarize(rows):
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "uci_ssl_cold_start_results.csv", index=False)
    summary = (
        result.groupby(["model", "protocol", "adapt_days"], as_index=False)
        .agg(
            target_clients=("target_client", "nunique"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_smape=("smape", "mean"),
            total_n=("n", "sum"),
        )
        .sort_values(["adapt_days", "mean_rmse", "model"])
    )
    summary.to_csv(RESULTS / "uci_ssl_cold_start_summary.csv", index=False)
    return result, summary


def summarize_domain(diagnostics):
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(RESULTS / "uci_ssl_domain_shift_diagnostics.csv", index=False)
    corr_distance_source = float(diag["source_target_latent_distance"].corr(diag["ssl_source_head_rmse"]))
    corr_distance_gain = float(diag["source_target_latent_distance"].corr(diag["adapter_gain_pct"]))
    summary = pd.DataFrame(
        [
            {
                "target_clients": int(len(diag)),
                "mean_latent_distance": float(diag["source_target_latent_distance"].mean()),
                "mean_source_head_rmse": float(diag["ssl_source_head_rmse"].mean()),
                "mean_adapter_28d_rmse": float(diag["ssl_adapter_28d_rmse"].mean()),
                "mean_adapter_gain_pct": float(diag["adapter_gain_pct"].mean()),
                "corr_distance_source_rmse": corr_distance_source,
                "corr_distance_adapter_gain_pct": corr_distance_gain,
            }
        ]
    )
    summary.to_csv(RESULTS / "uci_ssl_domain_shift_summary.csv", index=False)
    return diag, summary


def plot_cold_start(summary):
    width, height = 1900, 1180
    left, top, right, bottom = 180, 190, 1660, 900
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "UCI label-scarce adaptation curve", fill="#172033", font=font(44, True))
    d.text((88, 112), "Mean RMSE across ten target clients; lower is better", fill="#5f6b7a", font=font(24))

    ssl = summary[summary["model"].str.startswith("SSL-MR-lag")].copy()
    target = summary[summary["model"].str.startswith("Target-linear")].copy()
    seasonal = summary[summary["model"].isin(["Seasonal-24h", "Seasonal-168h"])].copy()
    xs = [0, 1, 3, 7, 28]
    all_values = list(ssl["mean_rmse"]) + list(target["mean_rmse"]) + list(seasonal["mean_rmse"])
    ymin = max(0, min(all_values) * 0.82)
    ymax = max(all_values) * 1.08

    def x_pos(days):
        return left + xs.index(days) * (right - left) / (len(xs) - 1)

    def y_pos(val):
        return bottom - (val - ymin) / (ymax - ymin) * (bottom - top)

    for i in range(6):
        val = ymin + (ymax - ymin) * i / 5
        y = y_pos(val)
        d.line((left, y, right, y), fill="#D7DCE2", width=1)
        d.text((left - 105, y - 13), f"{val:.0f}", fill="#526070", font=font(22))
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    for days in xs:
        x = x_pos(days)
        d.line((x, top, x, bottom), fill="#EEF1F5", width=1)
        label = "0" if days == 0 else str(days)
        tw = d.textlength(label, font=font(24, True))
        d.text((x - tw / 2, bottom + 25), label, fill="#1f2937", font=font(24, True))
    d.text((right + 18, bottom + 25), "days", fill="#526070", font=font(22))

    seasonal_colors = {"Seasonal-24h": "#B279A2", "Seasonal-168h": "#9D755D"}
    for _, row in seasonal.iterrows():
        y = y_pos(float(row["mean_rmse"]))
        d.line((left, y, right, y), fill=seasonal_colors[row["model"]], width=4)
        d.text((right + 20, y - 13), f"{row['model']} {row['mean_rmse']:.1f}", fill=seasonal_colors[row["model"]], font=font(20, True))

    series_specs = [
        ("SSL-MR-lag", "#E45756"),
        ("Target-linear", "#4C78A8"),
    ]
    for prefix, color in series_specs:
        pts = []
        if prefix == "SSL-MR-lag":
            source = summary[summary["model"] == "SSL-MR-lag-source-head"]
            if not source.empty:
                pts.append((0, float(source["mean_rmse"].iloc[0])))
            for days in ADAPT_DAYS:
                row = summary[summary["model"] == f"SSL-MR-lag+adapter-{days}d"]
                if not row.empty:
                    pts.append((days, float(row["mean_rmse"].iloc[0])))
        else:
            for days in ADAPT_DAYS:
                row = summary[summary["model"] == f"Target-linear-{days}d"]
                if not row.empty:
                    pts.append((days, float(row["mean_rmse"].iloc[0])))
        for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
            d.line((x_pos(d0), y_pos(v0), x_pos(d1), y_pos(v1)), fill=color, width=5)
        for days, val in pts:
            x, y = x_pos(days), y_pos(val)
            d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color)
            d.text((x + 12, y - 28), f"{val:.1f}", fill="#1f2937", font=font(20, True))

    d.rectangle((350, 980, 384, 1008), fill="#E45756")
    d.text((398, 974), "SSL source representation + adapter", fill="#526070", font=font(22))
    d.rectangle((830, 980, 864, 1008), fill="#4C78A8")
    d.text((878, 974), "Target-only lag-calendar ridge", fill="#526070", font=font(22))
    d.text((90, 1060), "Source: UCI Electricity Load Diagrams 2011-2014; target test period is Oct-Dec 2014.", fill="#6b7280", font=font(21))
    out = FIGURES / "paper2_fig6_uci_cold_start_curve.png"
    img.save(out)
    return out


def plot_domain_shift(diag):
    width, height = 1800, 1080
    left, top, right, bottom = 190, 180, 1580, 850
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((85, 55), "UCI representation-domain shift diagnostic", fill="#172033", font=font(44, True))
    d.text((88, 112), "Target latent distance versus source-head RMSE; labels identify held-out clients", fill="#5f6b7a", font=font(24))
    xmin = float(diag["source_target_latent_distance"].min()) * 0.92
    xmax = float(diag["source_target_latent_distance"].max()) * 1.08
    ymin = float(diag["ssl_source_head_rmse"].min()) * 0.88
    ymax = float(diag["ssl_source_head_rmse"].max()) * 1.12

    def x_pos(v):
        return left + (v - xmin) / (xmax - xmin) * (right - left)

    def y_pos(v):
        return bottom - (v - ymin) / (ymax - ymin) * (bottom - top)

    for i in range(6):
        x_val = xmin + (xmax - xmin) * i / 5
        x = x_pos(x_val)
        d.line((x, top, x, bottom), fill="#EEF1F5", width=1)
        d.text((x - 24, bottom + 20), f"{x_val:.1f}", fill="#526070", font=font(21))
        y_val = ymin + (ymax - ymin) * i / 5
        y = y_pos(y_val)
        d.line((left, y, right, y), fill="#D7DCE2", width=1)
        d.text((left - 90, y - 13), f"{y_val:.0f}", fill="#526070", font=font(21))
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    d.line((left, top, left, bottom), fill="#8792a2", width=2)
    for _, row in diag.iterrows():
        x = x_pos(float(row["source_target_latent_distance"]))
        y = y_pos(float(row["ssl_source_head_rmse"]))
        gain = float(row["adapter_gain_pct"])
        color = "#54A24B" if gain >= 0 else "#E45756"
        r = 10 + min(14, abs(gain) * 0.5)
        d.ellipse((x - r, y - r, x + r, y + r), fill=color, outline="#1f2937", width=1)
        d.text((x + r + 5, y - 10), str(row["target_client"]).replace("MT_", ""), fill="#1f2937", font=font(18, True))
    d.text((left, bottom + 66), "Source-target latent distance", fill="#1f2937", font=font(24, True))
    d.text((65, top - 42), "RMSE", fill="#1f2937", font=font(24, True))
    d.rectangle((455, 960, 489, 988), fill="#54A24B")
    d.text((503, 954), "28-day adapter improves source head", fill="#526070", font=font(22))
    d.rectangle((925, 960, 959, 988), fill="#E45756")
    d.text((973, 954), "28-day adapter degrades source head", fill="#526070", font=font(22))
    out = FIGURES / "paper2_fig7_uci_domain_shift_diagnostic.png"
    img.save(out)
    return out


def main():
    hourly, source_clients, target_clients = load_data()
    scales = build_scales(hourly, source_clients, target_clients)
    source_windows = collect_source_windows(hourly, source_clients, scales)
    mean_vec, components, _, _ = fit_masked_reconstruction_basis(source_windows)
    source_x, source_y = source_training_matrix(hourly, source_clients, scales, mean_vec, components, include_lags=True)
    source_beta = fit_ridge(source_x, source_y, l2=1e-3)
    source_centroid = encode(source_windows, mean_vec, components).mean(axis=0)

    rows = []
    diagnostics = []
    for client in target_clients:
        client_rows, diagnostic = evaluate_client(hourly, client, scales[client], mean_vec, components, source_beta, source_centroid)
        rows.extend(client_rows)
        if diagnostic is not None:
            diagnostics.append(diagnostic)

    _, summary = summarize(rows)
    diag, domain_summary = summarize_domain(diagnostics)
    fig1 = plot_cold_start(summary)
    fig2 = plot_domain_shift(diag)
    print(RESULTS / "uci_ssl_cold_start_results.csv")
    print(RESULTS / "uci_ssl_cold_start_summary.csv")
    print(RESULTS / "uci_ssl_domain_shift_diagnostics.csv")
    print(RESULTS / "uci_ssl_domain_shift_summary.csv")
    print(fig1)
    print(fig2)
    print(summary.to_string(index=False))
    print(domain_summary.to_string(index=False))


if __name__ == "__main__":
    main()
