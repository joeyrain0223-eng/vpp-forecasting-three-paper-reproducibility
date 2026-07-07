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
