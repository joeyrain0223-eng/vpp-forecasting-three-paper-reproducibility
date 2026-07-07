from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TIDY = PROCESSED / "opsd_hourly_price_load_renewables_tidy.csv"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

EFFICIENCY = 0.92
BATTERY_CAPACITY_MWH = 1.0
FLEX_LOAD_MWH = 0.60
ACTIVE_HOURS = 4
IMBALANCE_PENALTY_RATE = 45.0
ROBUST_GAMMA = 0.60
ROLLING_WINDOW_DAYS = 28


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


def cvar(values, alpha=0.10):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan
    cutoff = np.quantile(values, alpha)
    tail = values[values <= cutoff]
    return float(np.mean(tail)) if len(tail) else float(cutoff)


def daily_revenue(realized_price, charge_idx, discharge_idx, netload_error_ratio, capacity=BATTERY_CAPACITY_MWH):
    realized_price = np.asarray(realized_price, dtype=float)
    charge_idx = np.asarray(charge_idx, dtype=int)
    discharge_idx = np.asarray(discharge_idx, dtype=int)
    battery_power = capacity / ACTIVE_HOURS
    flex_power = FLEX_LOAD_MWH / ACTIVE_HOURS
    battery_revenue = battery_power * (
        realized_price[discharge_idx].sum() * EFFICIENCY
        - realized_price[charge_idx].sum() / EFFICIENCY
    )
    flex_value = flex_power * (
        realized_price[discharge_idx].sum() - realized_price[charge_idx].sum()
    )
    active_idx = np.concatenate([charge_idx, discharge_idx])
    imbalance_penalty = float(
        IMBALANCE_PENALTY_RATE
        * (battery_power + flex_power)
        * np.mean(np.abs(netload_error_ratio[active_idx]))
    )
    revenue = float(battery_revenue + flex_value - imbalance_penalty)
    return revenue, float(battery_revenue), float(flex_value), imbalance_penalty


def schedule_same_score(score, n=ACTIVE_HOURS):
    score = np.asarray(score, dtype=float)
    charge = np.argsort(score)[:n]
    discharge = np.argsort(score)[-n:]
    return charge, discharge


def schedule_two_scores(charge_score, discharge_score, n=ACTIVE_HOURS):
    charge = np.argsort(np.asarray(charge_score, dtype=float))[:n]
    discharge = np.argsort(np.asarray(discharge_score, dtype=float))[-n:]
    overlap = set(charge).intersection(set(discharge))
    if overlap:
        discharge_candidates = [idx for idx in np.argsort(discharge_score)[::-1] if idx not in set(charge)]
        discharge = np.asarray(discharge_candidates[:n], dtype=int)
    return np.asarray(charge, dtype=int), np.asarray(discharge, dtype=int)


def prepare_days():
    tidy = pd.read_csv(TIDY, parse_dates=["timestamp_utc"])
    tidy["date"] = tidy["timestamp_utc"].dt.date
    tidy["hour"] = tidy["timestamp_utc"].dt.hour
    tidy["net_load_mw"] = tidy["load_mw"] - tidy["wind_mw"].fillna(0) - tidy["solar_mw"].fillna(0)
    days = {}
    for zone, zdf in tidy.groupby("zone", sort=True):
        zone_days = []
        scale = max(float(zdf["load_mw"].median()), 1.0)
        for date, day in zdf.groupby("date", sort=True):
            day = day.sort_values("hour")
            if len(day) < 24:
                continue
            price = day["price_eur_mwh"].to_numpy(float)
            net = day["net_load_mw"].to_numpy(float)
            if np.isfinite(price).sum() < 24 or np.isfinite(net).sum() < 24:
                continue
            zone_days.append({"date": pd.to_datetime(str(date)), "price": price, "net_load": net, "scale": scale})
        days[zone] = zone_days
    return days


def historical_profiles(zone_days, idx):
    prev = zone_days[idx - 1]
    prev_price = prev["price"]
    prev_net = prev["net_load"]
    start = max(0, idx - ROLLING_WINDOW_DAYS)
    hist = zone_days[start:idx]
    hist_prices = np.vstack([d["price"] for d in hist])
    hist_net = np.vstack([d["net_load"] for d in hist])
    mean_price = hist_prices.mean(axis=0)
    std_price = hist_prices.std(axis=0)
    mean_net = hist_net.mean(axis=0)
    return prev_price, prev_net, mean_price, std_price, mean_net


def evaluate_zone(zone, zone_days, capacity=BATTERY_CAPACITY_MWH):
    rows = []
    start_idx = max(ROLLING_WINDOW_DAYS, 1)
    for idx in range(start_idx, len(zone_days)):
        day = zone_days[idx]
        price = day["price"]
        net_load = day["net_load"]
        scale = day["scale"]
        prev_price, prev_net, mean_price, std_price, mean_net = historical_profiles(zone_days, idx)
        net_error_prev = (net_load - prev_net) / scale
        net_error_mean = (net_load - mean_net) / scale

        h_charge, h_discharge = schedule_same_score(price)
        hindsight, h_batt, h_flex, h_penalty = daily_revenue(price, h_charge, h_discharge, np.zeros_like(price), capacity)

        methods = []
        prev_charge, prev_discharge = schedule_same_score(prev_price)
        methods.append(("Prev-day FTO", prev_charge, prev_discharge, net_error_prev))

        mean_charge, mean_discharge = schedule_same_score(mean_price)
        methods.append(("Rolling-28d mean FTO", mean_charge, mean_discharge, net_error_mean))

        robust_charge_score = mean_price + ROBUST_GAMMA * std_price
        robust_discharge_score = mean_price - ROBUST_GAMMA * std_price
        robust_charge, robust_discharge = schedule_two_scores(robust_charge_score, robust_discharge_score)
        methods.append(("Robust quantile FTO", robust_charge, robust_discharge, net_error_mean))

        methods.append(("Hindsight optimum", h_charge, h_discharge, np.zeros_like(price)))

        for method, charge_idx, discharge_idx, net_error in methods:
            revenue, battery_revenue, flex_value, penalty = daily_revenue(price, charge_idx, discharge_idx, net_error, capacity)
            rows.append(
                {
                    "dataset": "OPSD",
                    "zone": zone,
                    "date": day["date"].date().isoformat(),
                    "capacity_mwh": capacity,
                    "method": method,
                    "revenue": revenue,
                    "hindsight_revenue": hindsight,
                    "regret": hindsight - revenue,
                    "battery_revenue": battery_revenue,
                    "flex_value": flex_value,
                    "imbalance_penalty": penalty,
                    "negative_revenue": int(revenue < 0),
                }
            )
    return rows


def summarize(daily):
    rows = []
    for (zone, method), g in daily.groupby(["zone", "method"], sort=True):
        revenue = g["revenue"].to_numpy(float)
        regret = g["regret"].to_numpy(float)
        rows.append(
            {
                "zone": zone,
                "method": method,
                "days": int(len(g)),
                "mean_revenue": float(np.mean(revenue)),
                "mean_regret": float(np.mean(regret)),
                "median_regret": float(np.median(regret)),
                "negative_revenue_days": int(g["negative_revenue"].sum()),
                "revenue_p05": float(np.quantile(revenue, 0.05)),
                "cvar_10": cvar(revenue, 0.10),
                "mean_battery_revenue": float(g["battery_revenue"].mean()),
                "mean_flex_value": float(g["flex_value"].mean()),
                "mean_imbalance_penalty": float(g["imbalance_penalty"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    method_order = {
        "Hindsight optimum": 0,
        "Robust quantile FTO": 1,
        "Rolling-28d mean FTO": 2,
        "Prev-day FTO": 3,
    }
    summary["order"] = summary["method"].map(method_order)
    return summary.sort_values(["zone", "order"]).drop(columns=["order"])


def sensitivity(days_by_zone):
    rows = []
    for capacity in [0.5, 1.0, 2.0]:
        daily_rows = []
        for zone, zone_days in days_by_zone.items():
            daily_rows.extend(evaluate_zone(zone, zone_days, capacity=capacity))
        daily = pd.DataFrame(daily_rows)
        summary = summarize(daily)
        for _, row in summary.iterrows():
            if row["method"] in {"Prev-day FTO", "Robust quantile FTO"}:
                rows.append({"capacity_mwh": capacity, **row.to_dict()})
    return pd.DataFrame(rows)


def run():
    if not TIDY.exists():
        raise SystemExit(f"Missing processed OPSD data: {TIDY}")
    days_by_zone = prepare_days()
    daily_rows = []
    for zone, zone_days in days_by_zone.items():
        daily_rows.extend(evaluate_zone(zone, zone_days, capacity=BATTERY_CAPACITY_MWH))
    daily = pd.DataFrame(daily_rows)
    summary = summarize(daily)
    sens = sensitivity(days_by_zone)
    daily.to_csv(RESULTS / "opsd_vpp_risk_extended_daily.csv", index=False)
    summary.to_csv(RESULTS / "opsd_vpp_risk_extended_summary.csv", index=False)
    sens.to_csv(RESULTS / "opsd_vpp_risk_capacity_sensitivity.csv", index=False)
    return daily, summary, sens


def average_by_method(summary):
    return (
        summary.groupby("method", as_index=False)
        .agg(
            mean_revenue=("mean_revenue", "mean"),
            mean_regret=("mean_regret", "mean"),
            cvar_10=("cvar_10", "mean"),
            negative_revenue_days=("negative_revenue_days", "sum"),
            mean_imbalance_penalty=("mean_imbalance_penalty", "mean"),
        )
        .sort_values("mean_revenue", ascending=False)
    )


def plot_revenue(summary):
    plot = average_by_method(summary)
    plot = plot[plot["method"] != "Hindsight optimum"].copy()
    width, height = 1800, 1050
    left, top, right, bottom = 520, 185, 1600, 850
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "OPSD VPP risk-aware decision benchmark", fill="#172033", font=font(44, True))
    d.text((92, 112), "Mean daily revenue across four public zones; includes battery, flexible load, and imbalance penalty", fill="#5f6b7a", font=font(23))
    max_v = max(float(plot["mean_revenue"].max()) * 1.18, 1.0)
    for i in range(6):
        x = left + i * (right - left) / 5
        d.line((x, top, x, bottom), fill="#D7DCE2", width=1)
        label = f"{max_v * i / 5:.0f}"
        d.text((x - d.textlength(label, font=font(21)) / 2, bottom + 18), label, fill="#526070", font=font(21))
    row_h = (bottom - top) / len(plot)
    colors = {"Robust quantile FTO": "#E45756", "Rolling-28d mean FTO": "#4C78A8", "Prev-day FTO": "#72B7B2"}
    for idx, row in enumerate(plot.itertuples(index=False)):
        y = top + idx * row_h + row_h * 0.22
        bar_h = row_h * 0.50
        x1 = left + float(row.mean_revenue) / max_v * (right - left)
        d.rounded_rectangle((left, y, x1, y + bar_h), radius=8, fill=colors.get(row.method, "#4C78A8"))
        d.text((90, y + 4), row.method, fill="#1f2937", font=font(25, True))
        d.text((x1 + 14, y + 4), f"{float(row.mean_revenue):.2f}", fill="#1f2937", font=font(24, True))
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    out = FIGURES / "paper3_fig4_opsd_vpp_risk_revenue.png"
    img.save(out)
    return out


def plot_cvar(summary):
    plot = average_by_method(summary)
    plot = plot[plot["method"] != "Hindsight optimum"].copy().sort_values("cvar_10", ascending=False)
    width, height = 1800, 1050
    left, top, right, bottom = 520, 185, 1600, 850
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 55), "Downside revenue and CVaR", fill="#172033", font=font(44, True))
    d.text((92, 112), "Mean worst-decile revenue across public zones; higher is better", fill="#5f6b7a", font=font(24))
    min_v = min(float(plot["cvar_10"].min()), 0.0)
    max_v = max(float(plot["cvar_10"].max()), 1.0)
    span = max_v - min_v + 1e-6
    zero_x = left + (0 - min_v) / span * (right - left)
    for i in range(6):
        x = left + i * (right - left) / 5
        d.line((x, top, x, bottom), fill="#D7DCE2", width=1)
        label = f"{min_v + span * i / 5:.0f}"
        d.text((x - d.textlength(label, font=font(21)) / 2, bottom + 18), label, fill="#526070", font=font(21))
    d.line((zero_x, top, zero_x, bottom), fill="#8792a2", width=3)
    row_h = (bottom - top) / len(plot)
    colors = {"Robust quantile FTO": "#E45756", "Rolling-28d mean FTO": "#4C78A8", "Prev-day FTO": "#72B7B2"}
    for idx, row in enumerate(plot.itertuples(index=False)):
        y = top + idx * row_h + row_h * 0.22
        bar_h = row_h * 0.50
        val = float(row.cvar_10)
        x_val = left + (val - min_v) / span * (right - left)
        x0, x1 = sorted([zero_x, x_val])
        d.rounded_rectangle((x0, y, x1, y + bar_h), radius=8, fill=colors.get(row.method, "#4C78A8"))
        d.text((90, y + 4), row.method, fill="#1f2937", font=font(25, True))
        d.text((x1 + 14, y + 4), f"{val:.2f}", fill="#1f2937", font=font(24, True))
    d.line((left, bottom, right, bottom), fill="#8792a2", width=2)
    out = FIGURES / "paper3_fig5_opsd_vpp_cvar.png"
    img.save(out)
    return out


def main():
    daily, summary, sens = run()
    fig4 = plot_revenue(summary)
    fig5 = plot_cvar(summary)
    print(RESULTS / "opsd_vpp_risk_extended_daily.csv")
    print(RESULTS / "opsd_vpp_risk_extended_summary.csv")
    print(RESULTS / "opsd_vpp_risk_capacity_sensitivity.csv")
    print(fig4)
    print(fig5)
    print(summary.to_string(index=False))
    print(sens.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
