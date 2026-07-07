from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import run_opsd_vpp_risk_simulator as sim


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TIDY = PROCESSED / "opsd_hourly_price_load_renewables_tidy.csv"
PRICE_FORECAST = RESULTS / "opsd_price_probabilistic_conformal_daily.csv"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

WIDTH_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]
NET_GRID = [-8.0, -4.0, 0.0, 4.0, 8.0]
RISK_WEIGHT = 0.50


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


def fit_linear(train, feature_cols, target_col, l2=1e-6):
    x = train[feature_cols].to_numpy(float)
    y = train[target_col].to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    eye = np.eye(x.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + l2 * eye, x.T @ y)


def predict_linear(frame, feature_cols, beta):
    x = frame[feature_cols].to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    return x @ beta


def net_shape(values):
    values = np.asarray(values, dtype=float)
    std = float(values.std())
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def prepare_net_forecast(tidy, price_forecast):
    tidy = tidy.sort_values(["zone", "timestamp_utc"]).copy()
    tidy["timestamp_utc"] = pd.to_datetime(tidy["timestamp_utc"], utc=True)
    tidy["net_load_mw"] = tidy["load_mw"] - tidy["wind_mw"].fillna(0) - tidy["solar_mw"].fillna(0)
    for lag in [1, 24, 168]:
        tidy[f"net_lag_{lag}"] = tidy.groupby("zone")["net_load_mw"].shift(lag)
    feature_cols = [
        "net_lag_1",
        "net_lag_24",
        "net_lag_168",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month",
    ]
    rows = []
    forecast_ts = price_forecast[["zone", "timestamp_utc"]].drop_duplicates()
    for zone, zdf in tidy.groupby("zone", sort=True):
        zone_ts = forecast_ts[forecast_ts["zone"] == zone]["timestamp_utc"]
        if zone_ts.empty:
            continue
        first_forecast_ts = zone_ts.min()
        work = zdf[["timestamp_utc", "zone", "net_load_mw", "load_mw"] + feature_cols].dropna().copy()
        train = work[work["timestamp_utc"] < first_forecast_ts].copy()
        test = work[work["timestamp_utc"].isin(set(zone_ts))].copy()
        if len(train) < 1000 or len(test) < 24:
            continue
        beta = fit_linear(train, feature_cols, "net_load_mw")
        test["net_pred_mw"] = predict_linear(test, feature_cols, beta)
        scale = max(float(zdf["load_mw"].median()), 1.0)
        test["net_error_ratio"] = (test["net_load_mw"] - test["net_pred_mw"]) / scale
        rows.append(test[["timestamp_utc", "zone", "net_load_mw", "net_pred_mw", "net_error_ratio"]])
    if not rows:
        raise RuntimeError("No net-load forecasts could be generated")
    return pd.concat(rows, ignore_index=True)


def daily_contexts():
    if not PRICE_FORECAST.exists():
        raise SystemExit(f"Missing Paper 1 price forecast file: {PRICE_FORECAST}")
    if not TIDY.exists():
        raise SystemExit(f"Missing OPSD processed data: {TIDY}")
    price = pd.read_csv(PRICE_FORECAST, parse_dates=["timestamp_utc"])
    price["timestamp_utc"] = pd.to_datetime(price["timestamp_utc"], utc=True)
    tidy = pd.read_csv(TIDY, parse_dates=["timestamp_utc"])
    tidy["timestamp_utc"] = pd.to_datetime(tidy["timestamp_utc"], utc=True)
    net_pred = prepare_net_forecast(tidy, price)
    merged = price.merge(net_pred, on=["zone", "timestamp_utc"], how="inner")
    merged["date"] = merged["timestamp_utc"].dt.date.astype(str)
    days_by_zone = sim.prepare_days()
    contexts = {}
    for zone, zdf in merged.groupby("zone", sort=True):
        sim_days = days_by_zone.get(zone, [])
        date_to_idx = {d["date"].date().isoformat(): idx for idx, d in enumerate(sim_days)}
        rows = []
        for date, day in zdf.groupby("date", sort=True):
            day = day.sort_values("timestamp_utc")
            if len(day) != 24 or date not in date_to_idx:
                continue
            idx = date_to_idx[date]
            if idx < sim.ROLLING_WINDOW_DAYS:
                continue
            hist = sim.historical_profiles(sim_days, idx)
            prev_price, prev_net, mean_price, std_price, mean_net = hist
            realized_price = day["price_eur_mwh"].to_numpy(float)
            pred_price = day["pred"].to_numpy(float)
            lower = day["regime_lower"].to_numpy(float)
            upper = day["regime_upper"].to_numpy(float)
            net_error_model = day["net_error_ratio"].to_numpy(float)
            realized_net = day["net_load_mw"].to_numpy(float)
            pred_net = day["net_pred_mw"].to_numpy(float)
            scale = sim_days[idx]["scale"]
            net_error_mean = (realized_net - mean_net) / scale
            h_charge, h_discharge = sim.schedule_same_score(realized_price)
            hindsight, _, _, _ = sim.daily_revenue(
                realized_price, h_charge, h_discharge, np.zeros_like(realized_price)
            )
            rows.append(
                {
                    "date": date,
                    "price": realized_price,
                    "pred_price": pred_price,
                    "lower": lower,
                    "upper": upper,
                    "interval_half_width": (upper - lower) / 2,
                    "pred_net": pred_net,
                    "net_error_model": net_error_model,
                    "mean_price": mean_price,
                    "std_price": std_price,
                    "net_error_mean": net_error_mean,
                    "hindsight": hindsight,
                    "h_charge": h_charge,
                    "h_discharge": h_discharge,
                }
            )
        contexts[zone] = rows
    return contexts


def evaluate_schedule(ctx, charge_idx, discharge_idx, net_error, method, zone, split):
    revenue, battery_revenue, flex_value, penalty = sim.daily_revenue(
        ctx["price"], charge_idx, discharge_idx, net_error
    )
    return {
        "dataset": "OPSD",
        "zone": zone,
        "date": ctx["date"],
        "split": split,
        "method": method,
        "revenue": revenue,
        "hindsight_revenue": ctx["hindsight"],
        "regret": ctx["hindsight"] - revenue,
        "battery_revenue": battery_revenue,
        "flex_value": flex_value,
        "imbalance_penalty": penalty,
        "negative_revenue": int(revenue < 0),
    }


def candidate_scores(ctx, coef):
    c_width, d_width, c_net, d_net = coef
    shape = net_shape(ctx["pred_net"])
    charge_score = ctx["pred_price"] + c_width * ctx["interval_half_width"] + c_net * shape
    discharge_score = ctx["pred_price"] + d_width * ctx["interval_half_width"] + d_net * shape
    return charge_score, discharge_score


def evaluate_coef(contexts, coef):
    revenues = []
    for ctx in contexts:
        charge_score, discharge_score = candidate_scores(ctx, coef)
        charge, discharge = sim.schedule_two_scores(charge_score, discharge_score)
        revenue, _, _, _ = sim.daily_revenue(ctx["price"], charge, discharge, ctx["net_error_model"])
        revenues.append(revenue)
    return np.asarray(revenues, dtype=float)


def select_policy(contexts, objective):
    best = None
    for coef in product(WIDTH_GRID, WIDTH_GRID, NET_GRID, NET_GRID):
        revenue = evaluate_coef(contexts, coef)
        if objective == "revenue":
            score = float(np.mean(revenue))
        elif objective == "risk_adjusted":
            score = float(np.mean(revenue) + RISK_WEIGHT * sim.cvar(revenue, 0.10))
        else:
            raise ValueError(objective)
        if best is None or score > best["score"]:
            best = {
                "coef": coef,
                "score": score,
                "train_mean_revenue": float(np.mean(revenue)),
                "train_cvar_10": sim.cvar(revenue, 0.10),
            }
    return best


def split_zone_contexts(contexts):
    n = len(contexts)
    if n < 60:
        return [], contexts
    cut = max(1, int(n * 0.50))
    return contexts[:cut], contexts[cut:]


def baseline_rows(ctx, zone, split):
    rows = []
    h_charge, h_discharge = ctx["h_charge"], ctx["h_discharge"]
    rows.append(
        evaluate_schedule(
            ctx, h_charge, h_discharge, np.zeros_like(ctx["price"]), "Hindsight optimum", zone, split
        )
    )
    mean_charge, mean_discharge = sim.schedule_same_score(ctx["mean_price"])
    rows.append(
        evaluate_schedule(
            ctx, mean_charge, mean_discharge, ctx["net_error_mean"], "Rolling-28d mean FTO", zone, split
        )
    )
    robust_charge_score = ctx["mean_price"] + sim.ROBUST_GAMMA * ctx["std_price"]
    robust_discharge_score = ctx["mean_price"] - sim.ROBUST_GAMMA * ctx["std_price"]
    robust_charge, robust_discharge = sim.schedule_two_scores(robust_charge_score, robust_discharge_score)
    rows.append(
        evaluate_schedule(
            ctx, robust_charge, robust_discharge, ctx["net_error_mean"], "Robust quantile FTO", zone, split
        )
    )
    point_charge, point_discharge = sim.schedule_same_score(ctx["pred_price"])
    rows.append(
        evaluate_schedule(
            ctx,
            point_charge,
            point_discharge,
            ctx["net_error_model"],
            "Paper1 point + Paper2 net FTO",
            zone,
            split,
        )
    )
    interval_charge, interval_discharge = sim.schedule_two_scores(ctx["upper"], ctx["lower"])
    rows.append(
        evaluate_schedule(
            ctx,
            interval_charge,
            interval_discharge,
            ctx["net_error_model"],
            "Paper1 interval + Paper2 net FTO",
            zone,
            split,
        )
    )
    return rows


def policy_rows(ctx, zone, split, policies):
    rows = []
    for label, selected in policies.items():
        charge_score, discharge_score = candidate_scores(ctx, selected["coef"])
        charge, discharge = sim.schedule_two_scores(charge_score, discharge_score)
        rows.append(evaluate_schedule(ctx, charge, discharge, ctx["net_error_model"], label, zone, split))
    return rows


def summarize(daily):
    rows = []
    for (split, zone, method), g in daily.groupby(["split", "zone", "method"], sort=True):
        revenue = g["revenue"].to_numpy(float)
        regret = g["regret"].to_numpy(float)
        rows.append(
            {
                "split": split,
                "zone": zone,
                "method": method,
                "days": int(len(g)),
                "mean_revenue": float(np.mean(revenue)),
                "mean_regret": float(np.mean(regret)),
                "median_regret": float(np.median(regret)),
                "negative_revenue_days": int(g["negative_revenue"].sum()),
                "revenue_p05": float(np.quantile(revenue, 0.05)),
                "cvar_10": sim.cvar(revenue, 0.10),
                "mean_imbalance_penalty": float(g["imbalance_penalty"].mean()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_test(summary):
    test = summary[summary["split"] == "test"].copy()
    return (
        test.groupby("method", as_index=False)
        .agg(
            mean_revenue=("mean_revenue", "mean"),
            mean_regret=("mean_regret", "mean"),
            cvar_10=("cvar_10", "mean"),
            negative_revenue_days=("negative_revenue_days", "sum"),
            mean_imbalance_penalty=("mean_imbalance_penalty", "mean"),
        )
        .sort_values("mean_revenue", ascending=False)
    )


def plot_forecast_coupled(summary):
    plot = aggregate_test(summary)
    plot = plot[plot["method"] != "Hindsight optimum"].copy()
    order = [
        "Forecast-coupled DF (revenue)",
        "Forecast-coupled DF (risk-adjusted)",
        "Paper1 interval + Paper2 net FTO",
        "Paper1 point + Paper2 net FTO",
        "Rolling-28d mean FTO",
        "Robust quantile FTO",
    ]
    plot["order"] = plot["method"].map({m: i for i, m in enumerate(order)})
    plot = plot.dropna(subset=["order"]).sort_values("order")
    width, height = 1900, 1160
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "Forecast-coupled VPP decision test", fill="#172033", font=font(44, True))
    d.text(
        (92, 114),
        "Paper 1 price intervals and Paper 2-style net-load forecasts are evaluated inside the VPP simulator",
        fill="#5f6b7a",
        font=font(23),
    )
    left, top, right, bottom = 650, 215, 1705, 900
    max_value = max(float(plot["mean_revenue"].max()) * 1.15, 1.0)
    for i in range(6):
        x = left + i * (right - left) / 5
        d.line((x, top, x, bottom), fill="#e5e9f0", width=1)
        d.text((x - 18, bottom + 25), f"{max_value * i / 5:.0f}", fill="#6b7280", font=font(21))
    colors = ["#2f6f9f", "#e45756", "#72b7b2", "#f2a541", "#4f7cac", "#8a8f98"]
    row_h = 96
    for i, row in enumerate(plot.itertuples(index=False)):
        y = top + 30 + i * row_h
        label = row.method.replace("Paper1 ", "P1 ").replace("Paper2 ", "P2 ")
        d.text((90, y + 15), label, fill="#263142", font=font(23, True))
        bar_len = (row.mean_revenue / max_value) * (right - left)
        d.rounded_rectangle((left, y, left + bar_len, y + 54), radius=8, fill=colors[i % len(colors)])
        d.text((left + bar_len + 12, y + 12), f"{row.mean_revenue:.2f}", fill="#263142", font=font(23, True))
        d.text((left + 430, y + 62), f"regret {row.mean_regret:.2f}; CVaR10 {row.cvar_10:.2f}", fill="#5f6b7a", font=font(19))
    d.text((left, bottom + 72), "Mean daily revenue on forecast-coupled held-out test split (EUR/day proxy)", fill="#526070", font=font(23, True))
    out = FIGURES / "paper3_fig7_opsd_forecast_coupled_vpp.png"
    img.save(out)
    return out


def run():
    contexts_by_zone = daily_contexts()
    daily_rows = []
    coef_rows = []
    for zone, contexts in contexts_by_zone.items():
        adaptation, test = split_zone_contexts(contexts)
        if not adaptation or not test:
            continue
        policies = {
            "Forecast-coupled DF (revenue)": select_policy(adaptation, "revenue"),
            "Forecast-coupled DF (risk-adjusted)": select_policy(adaptation, "risk_adjusted"),
        }
        for label, selected in policies.items():
            c_width, d_width, c_net, d_net = selected["coef"]
            coef_rows.append(
                {
                    "zone": zone,
                    "method": label,
                    "charge_width_coef": c_width,
                    "discharge_width_coef": d_width,
                    "charge_net_coef": c_net,
                    "discharge_net_coef": d_net,
                    "selection_score": selected["score"],
                    "selection_mean_revenue": selected["train_mean_revenue"],
                    "selection_cvar_10": selected["train_cvar_10"],
                    "adaptation_days": len(adaptation),
                    "test_days": len(test),
                }
            )
        for split, split_contexts in [("adaptation", adaptation), ("test", test)]:
            for ctx in split_contexts:
                daily_rows.extend(baseline_rows(ctx, zone, split))
                daily_rows.extend(policy_rows(ctx, zone, split, policies))
    daily = pd.DataFrame(daily_rows)
    summary = summarize(daily)
    coefs = pd.DataFrame(coef_rows)
    aggregate = aggregate_test(summary)
    daily.to_csv(RESULTS / "opsd_forecast_coupled_vpp_daily.csv", index=False)
    summary.to_csv(RESULTS / "opsd_forecast_coupled_vpp_summary.csv", index=False)
    coefs.to_csv(RESULTS / "opsd_forecast_coupled_vpp_coefficients.csv", index=False)
    aggregate.to_csv(RESULTS / "opsd_forecast_coupled_vpp_test_aggregate.csv", index=False)
    fig = plot_forecast_coupled(summary)
    return daily, summary, coefs, aggregate, fig


if __name__ == "__main__":
    _, _, _, aggregate, fig = run()
    print(RESULTS / "opsd_forecast_coupled_vpp_daily.csv")
    print(RESULTS / "opsd_forecast_coupled_vpp_summary.csv")
    print(RESULTS / "opsd_forecast_coupled_vpp_coefficients.csv")
    print(RESULTS / "opsd_forecast_coupled_vpp_test_aggregate.csv")
    print(fig)
    print(aggregate.to_string(index=False))
