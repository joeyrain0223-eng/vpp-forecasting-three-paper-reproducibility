from __future__ import annotations

from pathlib import Path

from create_submission_candidate_manuscripts import rebuild_docx


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
PKG = ROOT / "manuscript"

PAPER_FILES = [
    PKG / "paper_3_decision_focused_vpp_bidding.md",
    ROOT / "manuscript" / "submission_candidate" / "paper_3_decision_focused_vpp_bidding.md",
]

APPENDIX_ASSET = PKG / "paper3_simulator_reproducibility_appendix.md"


NOTATION_BLOCK = """
The notation used in the reproducible simulator is summarized below.

|Symbol|Meaning in the public OPSD experiment|
|---|---|
|d|Decision day after the rolling-history warm-up period|
|h|Hourly index within the 24-hour decision horizon|
|p_d,h|Realized day-ahead price in EUR/MWh|
|p_hat_d,h|Forecast price or historical price score used for scheduling|
|n_d,h|Net load, computed as load minus wind and solar generation|
|epsilon_d,h|Scaled net-load forecast error used for imbalance settlement|
|C_d, D_d|Charge-hour and discharge-hour index sets selected for day d|
|B|Battery capacity in MWh; default 1.0, sensitivity 0.5/1.0/2.0|
|eta|Round-trip efficiency proxy used in storage revenue, fixed at 0.92|
|F|Daily flexible-load shifting volume, fixed at 0.60 MWh|
|pi_imb|Imbalance-penalty rate, fixed at 45 EUR/MWh in the base settlement|
|J(u; y)|Realized daily decision value under schedule u and realized outcomes y|
|CVaR10|Mean revenue in the worst 10 percent of realized daily revenues|
""".strip()


REPRO_PROTOCOL = """
### 5.5 Reproducibility protocol and held-out evaluation

All public experiments use chronological evaluation. The simulator first removes days without complete 24-hour price and net-load observations, then starts evaluation only after a 28-day rolling-history warm-up. The transparent historical policies use only information available before the evaluated day: previous-day prices, rolling 28-day mean prices, rolling 28-day price dispersion, and rolling 28-day net-load profiles.

The extended risk simulator reports all eligible post-warm-up public days. The decision-focused policy-search experiment uses a chronological 60/20/20 split after the warm-up: the first 60 percent of eligible days are used as training days, the next 20 percent as validation days, and the final 20 percent as the held-out test split reported in the main tables. Policy coefficients are selected on the combined training and validation period and then frozen before test evaluation. The forecast-coupled experiment uses the public Paper 1 price-forecast file and a Paper 2-style lag-calendar net-load model; where the forecast-coupled window is shorter, policy selection uses the earlier half of the coupled window and reports the later half as held-out evaluation.

This protocol is intentionally simple. It makes the decision layer auditable, avoids leakage from future prices or loads, and keeps the contribution focused on prediction-to-decision alignment rather than on a proprietary market simulator.
""".strip()


APPENDIX_BLOCK = """
## Appendix A. Simulator and Reproducibility Details

The public VPP simulator is implemented as a daily scoring and settlement engine over the OPSD hourly time-series data. For each zone and day, candidate policies select four charge hours and four discharge hours. If a two-score policy creates overlap between charge and discharge hours, discharge hours are reselected from the highest remaining discharge scores so that no hour is both charged and discharged.

### A.1 Daily settlement equation

For a day d, let C_d be the selected charge-hour set and D_d be the selected discharge-hour set. The default active storage power is q_B = B / 4 because four charge and four discharge hours are used. The flexible-load power proxy is q_F = F / 4, where F = 0.60 MWh. The base daily value is

R_d = q_B * (eta * sum_{h in D_d} p_d,h - eta^{-1} * sum_{h in C_d} p_d,h)
    + q_F * (sum_{h in D_d} p_d,h - sum_{h in C_d} p_d,h)
    minus pi_imb * (q_B + q_F) * mean_{h in C_d union D_d} |epsilon_d,h|.

The base setting uses eta = 0.92, B = 1.0 MWh, F = 0.60 MWh, pi_imb = 45 EUR/MWh, and four active charge/discharge hours. Capacity sensitivity repeats the same calculation for B in {0.5, 1.0, 2.0}. The reviewer stress test changes pi_imb to 70 EUR/MWh and adds a transaction-cost term c_tx * (q_B + q_F) * |C_d union D_d| with c_tx = 3 EUR/MWh.

### A.2 Baseline policies

The hindsight optimum uses the realized same-day price vector only as an upper-reference schedule under the same battery, flexible-load, and penalty definitions. The previous-day FTO policy scores hours with the previous day's price profile. The rolling-28d mean FTO policy scores hours with the rolling mean profile. The robust quantile FTO policy uses separate charge and discharge scores:

charge_score_h = mean_price_h + gamma * std_price_h,
discharge_score_h = mean_price_h - gamma * std_price_h,

with gamma = 0.60. This robust rule intentionally charges more conservatively in high-volatility hours and discharges against a lower-bound price proxy.

### A.3 Decision-focused policy search

The decision-focused policy search does not claim to be a black-box neural optimizer. It is a transparent surrogate layer used to test the predict-and-optimize hypothesis under held-out evaluation. Candidate policies use four coefficients:

charge_score_h = mean_price_h + a_c * std_price_h + b_c * shape(net_load)_h,
discharge_score_h = mean_price_h + a_d * std_price_h + b_d * shape(net_load)_h.

The grid is a_c, a_d in {-1.0, -0.5, 0.0, 0.5, 1.0} and b_c, b_d in {-8.0, -4.0, 0.0, 4.0, 8.0}. The revenue-selected policy maximizes mean training revenue. The risk-adjusted policy maximizes mean training revenue plus 0.50 times training CVaR10. The risk-aversion sweep repeats the same selection with lambda_r in {0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00}.

### A.4 Forecast-coupled policy search

The forecast-coupled experiment replaces historical price scores with Paper 1 price forecasts and uses a Paper 2-style lag-calendar net-load model to estimate imbalance exposure. Candidate policies score charge and discharge hours with forecast price, conformal interval half-width, and predicted net-load shape:

charge_score_h = p_hat_h + a_c * interval_width_h + b_c * shape(net_hat)_h,
discharge_score_h = p_hat_h + a_d * interval_width_h + b_d * shape(net_hat)_h.

The coefficient grid is the same width/net-load grid used by the decision-search scripts. The policy is selected on the earlier part of the coupled forecast window and evaluated on held-out later days.

### A.5 Metrics and scripts

The reported revenue is the mean of R_d across test days and zones. Regret is J(u_star; y) - J(u; y), where u_star is the same-day hindsight reference under identical settlement assumptions. CVaR10 is the mean of the worst 10 percent of realized daily revenues. Loss days are days with negative realized revenue.

The public results are reproduced by `run_opsd_vpp_risk_simulator.py`, `run_opsd_decision_focused_policy_search.py`, `run_opsd_forecast_coupled_vpp.py`, `run_opsd_risk_aversion_sensitivity.py`, and `run_opsd_vpp_reviewer_robustness.py`. The scripts write daily-level CSV files, summary tables, coefficient registers, and manuscript figures under `public_experiment_results/` and `figures/`.
""".strip()


def insert_after(text: str, anchor: str, addition: str) -> str:
    if addition in text:
        return text
    if anchor not in text:
        raise RuntimeError(f"Anchor not found: {anchor[:80]}")
    return text.replace(anchor, anchor + "\n\n" + addition, 1)


def insert_before(text: str, anchor: str, addition: str) -> str:
    if addition in text:
        return text
    if anchor not in text:
        raise RuntimeError(f"Anchor not found: {anchor[:80]}")
    return text.replace(anchor, addition + "\n\n" + anchor, 1)


def main() -> None:
    APPENDIX_ASSET.write_text(APPENDIX_BLOCK + "\n", encoding="utf-8")
    notation_anchor = "where u_star is the hindsight-optimal decision under realized outcomes y."
    protocol_anchor = "Diagnostic comparisons remove or vary the main decision ingredients: forecast source, robust risk adjustment, storage capacity, flexible-load modeling, and the risk-aversion coefficient in decision-focused policy selection."
    appendix_anchor = "## 7. Conclusion"
    for path in PAPER_FILES:
        text = path.read_text(encoding="utf-8")
        text = insert_after(text, notation_anchor, NOTATION_BLOCK)
        text = insert_after(text, protocol_anchor, REPRO_PROTOCOL)
        text = insert_before(text, appendix_anchor, APPENDIX_BLOCK)
        path.write_text(text, encoding="utf-8")
        rebuild_docx(path, text)
        print(path)
        print(path.with_suffix(".docx"))
    print(APPENDIX_ASSET)


if __name__ == "__main__":
    main()
