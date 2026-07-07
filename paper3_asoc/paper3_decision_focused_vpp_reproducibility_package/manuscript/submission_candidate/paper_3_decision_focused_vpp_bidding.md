# Decision-Focused Learning for Virtual Power Plant Bidding under Electricity Price and Load Uncertainty



## Abstract

Virtual power plant bidding decisions depend on forecasts of electricity price, load, renewable generation, and resource flexibility. Standard forecasting pipelines optimize prediction accuracy first and then pass forecasts into a separate optimization module. This forecast-then-optimize paradigm can be suboptimal because the forecasting loss may not align with downstream market profit, risk exposure, or operational penalties. This paper studies a decision-focused, public-data virtual power plant bidding protocol under electricity price and load uncertainty. Rather than claiming a fully differentiable neural market engine, the implemented framework uses auditable forecast-to-decision simulators, trainable policy-search coefficients selected on historical decision value, forecast-coupled policies, and robustness/risk-aversion checks. Price and net-load uncertainty enter through public forecast signals, rolling robust baselines, fuzzy risk-aware scoring, and CVaR-oriented policy selection. The empirical evaluation compares previous-day forecast-then-optimize, rolling-mean forecast-then-optimize, robust quantile forecast-then-optimize, fuzzy risk-aware forecast-then-optimize, forecast-coupled policies, held-out decision-focused policy search, risk-aversion sweeps, and settlement-stress checks on public OPSD market scenarios. The study frames virtual power plant operation as a machine-learning problem of aligning prediction signals with downstream decision quality under transparent reproducibility constraints.

Keywords: virtual power plant; electricity market bidding; decision-focused learning; predict-and-optimize; uncertainty; robust optimization; fuzzy risk-aware baseline; time-series forecasting.

## 1. Introduction

Virtual power plants are expected to participate in electricity markets by coordinating distributed energy resources, flexible loads, storage, and renewable generation. Their bidding decisions require forecasts of prices and resource availability. A common engineering pipeline first trains forecasting models to minimize MAE or RMSE and then feeds the forecasts into an optimization model. Although intuitive, this pipeline can fail when small prediction errors near decision boundaries produce large financial losses or infeasible schedules.

For example, an electricity price forecast that is slightly inaccurate during a high-spread period can reverse the value of charging or discharging storage. A load forecast that is accurate on average may still lead to reserve shortfalls if its uncertainty is miscalibrated. These examples show that forecasting accuracy and decision quality are related but not equivalent. A virtual power plant needs predictions that are useful for decisions, not merely predictions that minimize symmetric statistical error.

Decision-focused learning provides a principled framework for this problem. Instead of treating optimization as a downstream black box, decision-aware pipelines incorporate decision loss, regret, or differentiable optimization layers when the implementation supports such feedback, following prescriptive analytics, smart predict-then-optimize, and task-based learning work [1]-[5]. In the present public implementation, this idea is evaluated through frozen forecast signals and transparent policy-search coefficients rather than through an end-to-end neural optimizer. This perspective is well aligned with computer-science research on predict-and-optimize, differentiable optimization, and task-based learning [2]-[5].

This paper studies decision-focused learning for virtual power plant bidding through a transparent public-data simulator and trainable policy-search layer. The implemented method combines probabilistic price and net-load forecast signals with a bidding simulator that accounts for storage constraints, flexible-load limits, imbalance penalties, and risk preferences. The paper compares this auditable decision-focused protocol with standard forecast-then-optimize methods and evaluates not only prediction error but also revenue, regret, risk, and operational penalties.

The contribution is not a new power-system dispatch formulation. Rather, the contribution is a learning framework that connects uncertain time-series prediction with downstream optimization objectives in a virtual power plant market setting.

## 2. Related Work

### 2.1 Virtual power plant market operation

Virtual power plants aggregate distributed resources to provide market services such as energy trading, demand response, reserve support, and renewable integration [12], [13]. Recent surveys emphasize that VPP operation is no longer a single dispatch problem; it is a resource-coordination problem involving distributed generation, flexible demand, storage, market rules, and multidimensional interactions among aggregators, system operators, and customers [32]. Market participation therefore requires decisions under uncertainty rather than only deterministic scheduling.

Existing VPP bidding studies usually formulate market participation through stochastic programming, robust optimization, bilevel bidding, or strategic reserve co-optimization. Stochastic offering models explicitly represent price and production uncertainty [25], energy and spinning-reserve bidding models formalize market participation constraints [26], and industrial VPP scheduling work connects stochastic profit objectives with demand response [27]. Other work studies intraday demand-response exchange markets and strategic energy-market bidding for VPPs [14], [15]. These studies are valuable, but they usually treat forecast models as exogenous inputs. The present paper instead asks how forecast signals and policy-selection rules should be evaluated by downstream market value.

### 2.2 Forecast-then-optimize and decision-focused learning

Predictive models are commonly evaluated with statistical error metrics, but electricity-price forecasting benchmarks show that statistical accuracy alone is only one layer of market-facing evaluation [10], [11], [23], [24]. Prescriptive analytics argues that prediction quality should be judged by the decision it supports when the operational objective is known [1]. Smart predict-then-optimize methods, differentiable optimization layers, task-based end-to-end learning, and decision-focused combinatorial optimization provide tools for aligning model training with optimization performance [2]-[5]. A recent decision-focused learning survey further clarifies that practical systems must separate differentiable end-to-end claims from auditable surrogate decision protocols, benchmark design, and deployment constraints [31].

This paper is positioned in that second, auditable branch. It does not claim that every market-settlement detail is differentiable. Instead, it connects public price and net-load forecast signals to a transparent VPP decision simulator and evaluates policies by realized revenue, regret, CVaR, loss days, and settlement stress. This keeps the contribution in the computer-science predict-and-optimize literature while avoiding an unsupported claim of a full neural market engine.

### 2.3 Uncertainty-aware energy decision making

Robust and stochastic optimization account for uncertain variables by considering worst-case sets, scenarios, or probability distributions. Their foundations are well established in robust optimization and stochastic programming [34], [35], and risk-sensitive market models often use coherent downside measures such as CVaR [6]. In machine-learning-based energy pipelines, uncertainty can be generated by quantile forecasts, ensembles, Bayesian methods, conformal prediction, or sequence models that provide multi-horizon uncertainty information [8], [9], [11].

The implemented protocol uses uncertainty not only for reporting prediction intervals but also as an input to risk-aware bidding decisions and policy selection. This design is deliberately narrower than a full stochastic market-clearing model: it keeps the public experiment reproducible, freezes policy coefficients before test evaluation, and reports downside behavior explicitly instead of replacing uncertainty with a single point forecast.

### 2.4 Soft-computing and fuzzy decision support

Applied Soft Computing reviewers will expect a visible connection to soft-computing methods rather than only a power-market simulation. Fuzzy-set theory and fuzzy system identification provide the classical basis for interpretable membership-based reasoning [17], [18], and ANFIS extends that reasoning into adaptive neuro-fuzzy inference [19]. Reviews of neuro-fuzzy systems and bio-inspired computational intelligence show that soft-computing methods are often selected when interpretability, nonlinear uncertainty handling, and heuristic search are more important than closed-form optimality [20], [28].

Energy systems provide many examples of this motivation. Fuzzy optimization has been used for VPP demand-response bidding and pricing [21], while fuzzy logic and genetic optimization have been used for uncertain microgrid energy management [29], [30]. These studies motivate the fuzzy risk-aware baseline in this paper. The baseline is intentionally interpretable and testable: it uses rolling price-level, price-volatility, and net-load-shape memberships, then evaluates the resulting schedule under the same public settlement simulator as the other policies. Its mixed result is important because it prevents a superficial soft-computing claim: fuzzy reasoning improves interpretability and some downside behavior, but it does not automatically dominate decision-value-selected policies.

### 2.5 Reinforcement learning and policy-learning boundary

Reinforcement learning is another natural candidate for energy-market decision making. Demand-response surveys show that RL can learn operational policies under delayed rewards and user-response uncertainty [22], and recent smart-grid reviews summarize deep reinforcement learning applications for dispatch, control, and market operation [33]. However, RL evidence can be difficult to compare fairly when simulators, reward shaping, market rules, and training horizons differ. For this reason, the present manuscript does not present an RL agent as the main contribution. It uses frozen, auditable policy-search coefficients as a conservative decision-focused layer and treats stronger neural or RL policy learning as a future extension once a public benchmark and reproducible settlement protocol are fixed.

## 3. Problem Formulation

Consider a virtual power plant that participates in a day-ahead or intraday market. At each decision time, it observes historical prices, load, renewable generation, storage state, and resource constraints. It chooses a bid or schedule u for future horizons h = 1, ..., H. The realized revenue depends on market price p_h, delivered energy e_h, imbalance penalties, and operating constraints.

A simplified objective is based on a convex or surrogate decision layer, using standard optimization notation [7]:

maximize_u E[sum_h p_h e_h(u) - C(u) - penalty(u, y)] - rho * Risk(u),

subject to storage dynamics, power limits, state-of-charge limits, load flexibility constraints, and market bid constraints.

The ideal decision-focused learning problem can be written as selecting forecast information z(X) and a policy pi so that the final decision u = pi(z(X)) reduces decision regret:

Regret = J(u_star; y) - J(pi(z(X)); y),

where u_star is the hindsight-optimal decision under realized outcomes y. The reproducible experiments instantiate this notation conservatively: z(X) is supplied by public forecast signals or rolling historical summaries, pi is selected through chronological coefficient search, and the selected policy is frozen before held-out evaluation. Thus the current evidence supports auditable policy selection, not end-to-end representation training through a differentiable market layer.

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

## 4. Proposed Method

### 4.1 Probabilistic forecasting module

The forecasting module outputs price and load quantiles or scenarios. It may reuse the models developed in the previous two manuscripts. For each horizon, the model generates a set of scenarios S = {p_h^s, l_h^s, r_h^s}, representing price, load, and renewable generation uncertainty.

### 4.2 Bidding optimization layer

The optimization layer maps scenarios into a market bid or operating schedule. A basic formulation includes:

- energy balance constraints;
- battery charge/discharge limits;
- state-of-charge dynamics;
- flexible-load shifting bounds;
- renewable generation availability;
- bid quantity and price constraints;
- risk constraints such as CVaR or downside revenue limits.

The exact formulation remains simple enough for clear experimentation. Overly complex electricity-market rules can obscure the machine-learning contribution.

### 4.3 Auditable decision-focused policy search

The reproducible implementation uses a transparent surrogate decision-focused layer rather than an end-to-end neural optimizer. Candidate policies score charge and discharge hours with historical price mean, price volatility, and net-load shape features. Policy coefficients are selected on chronological training and validation days by maximizing realized decision value or a revenue-plus-CVaR objective, and the selected coefficients are then frozen before held-out test evaluation.

This design tests the predict-and-optimize hypothesis without relying on an opaque proprietary market engine. The decision objective is:

L_decision = -J(u_theta; y) - lambda_r CVaR10(u_theta),

where J is realized daily decision value in the public simulator and lambda_r controls risk aversion. Differentiable optimization layers remain a natural extension, but the current manuscript claims only the public, auditable surrogate policy-search evidence reported in the experiments.


### 4.4 Risk-aware scenario selection

The model does not optimize only expected revenue. A risk-aware decision layer uses CVaR or quantile-based downside constraints:

CVaR_alpha(loss(u)) <= tau.

This allows the virtual power plant to trade expected profit against downside exposure. The empirical risk analysis reports capacity sensitivity, revenue-versus-CVaR policy selection, and a held-out risk-aversion sweep over the decision-search objective.

## 5. Experimental Design

### 5.1 Simulation environment

The public decision substrate uses OPSD hourly day-ahead price, load, wind, and solar series for DE-LU, DK1, DK2, and Great Britain [16]. The first transparent decision layer keeps the original storage-arbitrage benchmark, while the extended simulator adds battery capacity, flexible load shifting, net-load forecast error, imbalance penalties, downside revenue, and CVaR. The final forecast-coupled layer connects the Paper 1 conformal electricity-price forecasts and a Paper 2-style lag-calendar net-load forecast to the same VPP simulator, so probabilistic forecasting evidence is evaluated through downstream decision value [8]-[11]. The goal is not to claim a production market engine; it is to provide a reproducible predict-then-optimize testbed for decision-focused learning.

Table 1 reports the public simulator assumptions used by the extended OPSD experiment.

|Component|Setting|
|---|---|
|Market data|OPSD hourly day-ahead prices, load, wind, and solar|
|Zones|DE-LU, DK1, DK2, Great Britain|
|Decision horizon|Daily 24-hour horizon|
|Battery capacity|1.0 MWh default, with 0.5/1.0/2.0 MWh sensitivity|
|Charge/discharge power proxy|Four active charge hours and four active discharge hours per day|
|Battery efficiency|0.92 round-trip proxy in revenue calculation|
|Flexible load|0.60 MWh daily shift from high-price to low-price hours|
|Imbalance penalty|45 EUR/MWh proxy scaled by active energy and net-load forecast error|
|Forecast inputs|Previous-day profile, rolling 28-day mean profile, rolling 28-day robust price bands, Paper 1 price forecasts, and Paper 2-style net-load forecasts|
|Benchmark|Same-day hindsight optimum under the same battery, flexible-load, and imbalance-penalty definitions|

### 5.2 Baselines

The empirical comparison uses transparent historical baselines, decision-focused policy search, and forecast-coupled policies. The historical baselines include same-day hindsight, previous-day forecast-then-optimize, rolling 28-day mean forecast-then-optimize, and robust quantile forecast-then-optimize [1], [2]. The decision-focused variants choose charge and discharge scoring coefficients on a training or adaptation period by directly optimizing realized revenue or a revenue-plus-CVaR objective [1], [2], [6]. The forecast-coupled policies connect Paper 1 price forecasts and Paper 2-style net-load forecasts to the same simulator. The local disclosure-data pilot is retained as application evidence rather than the main reproducibility claim.

The decision-focused policy search is deliberately lightweight. It does not claim to be a full neural optimizer; it tests the core predict-and-optimize hypothesis under a fully auditable setting. Candidate policies use historical price mean, price volatility, and net-load shape features to score charging and discharging hours. The selected policy is then frozen and evaluated on the final 20% held-out public days.

### 5.3 Metrics

Decision metrics: total revenue, average daily profit, regret against hindsight optimum, imbalance penalty, downside revenue, CVaR, loss days, and risk-adjusted return [1], [2], [6]. Forecasting metrics are still reported but do not dominate the paper.

### 5.4 Ablation study

Diagnostic comparisons remove or vary the main decision ingredients: forecast source, robust risk adjustment, fuzzy risk-aware scoring, storage capacity, flexible-load modeling, and the risk-aversion coefficient in decision-focused policy selection.

### 5.5 Reproducibility protocol and held-out evaluation

All public experiments use chronological evaluation. The simulator first removes days without complete 24-hour price and net-load observations, then starts evaluation only after a 28-day rolling-history warm-up. The transparent historical policies use only information available before the evaluated day: previous-day prices, rolling 28-day mean prices, rolling 28-day price dispersion, and rolling 28-day net-load profiles.

The extended risk simulator reports all eligible post-warm-up public days. The decision-focused policy-search experiment uses a chronological 60/20/20 split after the warm-up: the first 60 percent of eligible days are used as training days, the next 20 percent as validation days, and the final 20 percent as the held-out test split reported in the main tables. Policy coefficients are selected on the combined training and validation period and then frozen before test evaluation. The forecast-coupled experiment uses the public Paper 1 price-forecast file and a Paper 2-style lag-calendar net-load model; where the forecast-coupled window is shorter, policy selection uses the earlier half of the coupled window and reports the later half as held-out evaluation.

This protocol is intentionally simple. It makes the decision layer auditable, avoids leakage from future prices or loads, and keeps the contribution focused on prediction-to-decision alignment rather than on a proprietary market simulator.

## 6. Results and Discussion

Table 2 reports the public OPSD decision benchmark. The result shows a clear gap between hindsight decision value and a simple previous-day forecast-then-optimize rule.

|Zone|Days|Hindsight|Prev-day FTO|Mean regret|Median regret|Loss days|
|---|---|---|---|---|---|---|
|DE_LU|720|82.040|46.960|35.080|17.751|65|
|DK_1|2085|54.316|25.897|28.419|15.832|546|
|DK_2|2085|59.810|35.211|24.600|12.629|413|
|GB_GBN|2074|86.977|64.703|22.274|14.930|57|

The public benchmark strengthens the central argument of this paper. Even when the decision model is intentionally simple, previous-day forecast-then-optimize leaves material regret on every public zone and produces negative revenue on many days in DK1 and DK2. This supports the need for a decision-focused learning objective that values errors according to their operational consequences rather than only their contribution to MAE or RMSE.

Table 3 reports the extended public VPP risk simulation averaged across the four OPSD zones. Unlike the earlier storage-only benchmark, this simulator includes a 1.0 MWh battery, 0.60 MWh flexible load shifting, and imbalance penalties derived from net-load forecast error. The hindsight row is a same-day reference under the same resource and penalty definitions, not an unconstrained revenue upper bound.

|Method|Revenue|Regret|CVaR10|Loss days|Penalty|
|---|---|---|---|---|---|
|Hindsight optimum|31.791|0|9.027|87|0|
|Rolling-28d mean FTO|20.881|10.911|-1.184|901|3.370|
|Robust quantile FTO|20.101|11.690|-0.954|938|3.315|
|Prev-day FTO|17.389|14.402|-7.179|1310|3.468|

The rolling 28-day mean FTO policy obtains the highest average revenue among implementable policies, while the robust quantile FTO policy slightly improves worst-decile revenue relative to rolling-mean FTO and materially improves downside revenue relative to previous-day FTO. This pattern is important for a decision-focused learning paper: the risk-aware policy is not universally superior on average revenue, but it changes the revenue-risk trade-off in the direction expected from a CVaR-aware objective.

Table 4 reports the capacity sensitivity for the two main implementable policies. Larger battery capacity increases upside revenue but also amplifies exposure to forecast and net-load mismatch, so regret and downside risk do not scale linearly.

|Capacity|Method|Revenue|Regret|CVaR10|Loss days|
|---|---|---|---|---|---|
|0.500|Robust quantile FTO|14.940|8.045|0.297|769|
|0.500|Prev-day FTO|13.089|9.896|-3.932|1109|
|1|Robust quantile FTO|20.101|11.690|-0.954|938|
|1|Prev-day FTO|17.389|14.402|-7.179|1310|
|2|Robust quantile FTO|30.422|18.981|-3.521|1097|
|2|Prev-day FTO|25.990|23.413|-13.719|1469|

The extended simulator therefore supplies the public risk layer for the third paper: revenue, regret, imbalance penalty, loss days, CVaR, and capacity sensitivity are all measurable on public data. It also gives the manuscript a direct computational interface to the probabilistic price forecasts from Paper 1 and the transferable load forecasts from Paper 2, so downstream learning objectives can be evaluated against regret and downside revenue rather than only against point-forecast error.

Table 5 reports the held-out decision-focused policy-search experiment. The revenue-optimized policy obtains the highest average revenue among implementable policies on the final 20% public test split. The risk-adjusted policy sacrifices a small amount of average revenue but improves CVaR relative to the revenue-only decision search. This result is the first trainable decision layer in the paper: policy parameters are selected on historical decision value and then evaluated out of sample.

|Method|Revenue|Regret|CVaR10|Loss days|Penalty|
|---|---|---|---|---|---|
|Hindsight optimum|33.470|0|10.829|5|0|
|DF policy search (revenue)|21.639|11.831|-1.853|115|3.392|
|DF policy search (risk-adjusted)|21.487|11.983|-1.276|110|3.402|
|Rolling-28d mean FTO|21.381|12.089|-1.500|110|3.442|
|Robust quantile FTO|21.128|12.342|-0.585|97|3.295|
|Prev-day FTO|18.266|15.204|-7.197|184|3.654|

![Figure 4. OPSD VPP extended risk simulation: average daily revenue by implementable policy.](figures/paper3_fig4_opsd_vpp_risk_revenue.png)

![Figure 5. OPSD VPP extended risk simulation: worst-decile revenue CVaR by implementable policy.](figures/paper3_fig5_opsd_vpp_cvar.png)

![Figure 6. OPSD decision-focused policy search on held-out public test days.](figures/paper3_fig6_opsd_decision_focused_policy_search.png)

Table 6 reports the forecast-coupled experiment that connects the first two manuscripts to the VPP decision simulator. Paper 1 supplies the hourly price point forecast and regime-conformal interval. Paper 2 supplies the load-forecasting interface through a lag-calendar net-load model trained before the price-forecast test window. On the held-out forecast-coupled split, Paper1 point plus Paper2 net-load forecast-then-optimize substantially improves average revenue and regret relative to historical rolling and robust baselines. The forecast-coupled decision-search policy gives the highest average revenue, while the conformal-interval policy is more conservative and is not the best average-revenue choice on this split. This result supports the dissertation-level linkage: better probabilistic price and load forecasts can be evaluated directly by downstream VPP value.

|Method|Revenue|Regret|CVaR10|Loss days|Penalty|
|---|---|---|---|---|---|
|Hindsight optimum|36.104|0|14.151|0|0|
|Forecast-coupled DF (revenue)|31.435|4.669|10.385|2|0.799|
|Paper1 point + Paper2 net FTO|31.368|4.736|10.571|2|0.798|
|Forecast-coupled DF (risk-adjusted)|31.250|4.854|10.128|2|0.793|
|Paper1 interval + Paper2 net FTO|30.659|5.445|9.488|4|0.780|
|Rolling-28d mean FTO|22.783|13.321|-0.274|49|3.375|
|Robust quantile FTO|22.318|13.787|-0.387|43|3.244|

![Figure 7. Forecast-coupled OPSD VPP decision test linking Paper 1 and Paper 2 forecasts to Paper 3.](figures/paper3_fig7_opsd_forecast_coupled_vpp.png)

Table 7 reports the risk-aversion sensitivity of the decision-focused policy-search objective. The policy is selected on the training and validation period by maximizing mean revenue plus lambda_r times CVaR10, then evaluated on the final held-out public test split. A moderate risk weight, lambda_r=0.25 or 0.50, reduces loss days from 115 to 110 and improves CVaR10 from -1.85 to -1.28 relative to revenue-only selection, at the cost of about 0.15 EUR/day proxy average revenue. Larger weights do not continue improving downside risk under the current coarse policy grid, so the paper reports the observed trade-off rather than claiming monotone risk improvement.

|Lambda_r|Revenue|Regret|CVaR10|Loss days|Penalty|
|---|---|---|---|---|---|
|0|21.639|11.831|-1.853|115|3.392|
|0.250|21.487|11.983|-1.276|110|3.402|
|0.500|21.487|11.983|-1.276|110|3.402|
|0.750|21.371|12.099|-1.741|112|3.424|
|1|21.371|12.099|-1.741|112|3.424|
|1.500|21.371|12.099|-1.741|112|3.424|
|2|21.371|12.099|-1.741|112|3.424|

![Figure 8. OPSD risk-aversion sensitivity of decision-focused VPP policy search.](figures/paper3_fig8_opsd_risk_aversion_sensitivity.png)

Table 8 reports reviewer-facing settlement robustness checks. The base-settlement rows match the held-out decision-search test split in Table 5. The stress rows raise the imbalance-penalty rate from 45 to 70 EUR/MWh and add a 3 EUR/MWh transaction-cost proxy to all active charge/discharge and flexible-load operations. Under this harsher settlement, all implementable policies lose revenue, but the revenue-selected decision-focused policy remains the strongest average-revenue policy. The robust quantile FTO policy retains fewer loss days and a less negative CVaR10 than the decision-focused variants in the high-penalty stress case, which is useful reviewer evidence: the proposed decision-focused layer improves value, but explicit robust scheduling still matters under severe settlement penalties.

|Scenario|Method|Revenue|Regret|CVaR10|Loss days|Txn penalty|
|---|---|---|---|---|---|---|
|Base settlement|DF policy search (revenue)|21.639|11.831|-1.853|115|0|
|Base settlement|DF policy search (risk-adjusted)|21.487|11.983|-1.276|110|0|
|Base settlement|Rolling-28d mean FTO|21.381|12.089|-1.500|110|0|
|Base settlement|Robust quantile FTO|21.128|12.342|-0.585|97|0|
|High penalty + transaction cost|DF policy search (revenue)|10.154|13.716|-14.125|386|9.600|
|High penalty + transaction cost|DF policy search (risk-adjusted)|9.997|13.873|-13.488|388|9.600|
|High penalty + transaction cost|Rolling-28d mean FTO|9.869|14.001|-13.805|386|9.600|
|High penalty + transaction cost|Robust quantile FTO|9.697|14.173|-12.558|365|9.600|

Table 9 reports the fine-grid check around the selected decision-focused policies. The refinement grid does not overturn the coarse-grid ranking: the coarse revenue-selected policy remains the best average-revenue variant, while the coarse risk-adjusted policy remains the strongest CVaR10 variant among the decision-focused grid checks. This reduces the risk that the paper's decision-search conclusion is an artifact of an overly coarse coefficient grid.

|Grid|Revenue|Regret|CVaR10|Loss days|Penalty|
|---|---|---|---|---|---|
|Coarse revenue grid|21.639|11.831|-1.853|115|3.392|
|Fine revenue grid|21.553|11.916|-1.693|116|3.394|
|Coarse risk-adjusted grid|21.487|11.983|-1.276|110|3.402|
|Fine risk-adjusted grid|21.441|12.029|-1.777|113|3.416|

![Figure 9. OPSD reviewer robustness checks for settlement stress and fine policy-grid refinement.](figures/paper3_fig9_opsd_reviewer_robustness.png)

### 6.7 Fuzzy soft-computing baseline check

To make the Applied Soft Computing fit explicit, an additional fuzzy risk-aware FTO comparator is evaluated on the same OPSD held-out test split. The comparator uses three transparent membership signals: rolling historical price level, rolling price volatility, and normalized net-load shape. It increases discharge priority when price and net-load memberships are high, increases charge priority when price membership is low, and penalizes high-volatility hours in both directions. This baseline is deliberately simple and fully auditable; it is not tuned after seeing the test set.

The fuzzy risk-aware comparator reaches 11.342 EUR/day proxy revenue, 22.128 regret, -4.968 CVaR10, 259 loss days, and 3.295 average imbalance penalty. Rolling-28d mean FTO and robust quantile FTO remain stronger on average revenue, at 21.381 and 21.128 EUR/day respectively, while fuzzy risk-aware FTO improves downside revenue relative to previous-day FTO (-4.968 versus -7.197 CVaR10). The result is therefore negative but informative: a soft-computing heuristic can encode interpretable risk preference, yet interpretability alone does not guarantee superior market value. This finding strengthens the manuscript's central claim that VPP bidding should be evaluated by downstream decision quality rather than by the presence of a particular forecasting or heuristic family.

The decision results are deterministic chronological hold-out simulation metrics rather than inferential p-values. The evidence should be read through out-of-sample ranking, downside-risk behavior, and settlement-stress robustness, not through significance testing across exchangeable samples. This is appropriate for the public simulator because the daily market scenarios are temporally ordered, policy coefficients are frozen before held-out evaluation, and the goal is to compare auditable decision protocols under identical settlement rules.

## 7. Conclusion

This paper studies decision-focused learning for virtual power plant bidding under electricity price and load uncertainty through a public, auditable forecast-to-decision simulator. By connecting probabilistic price and net-load forecast signals to policy-search objectives, the method evaluates predictive representations by market-operation outcomes rather than by forecasting error alone. The empirical evaluation emphasizes revenue, regret, downside risk, imbalance penalties, risk-aversion sensitivity, and settlement robustness. The work contributes bounded evidence for computer-science research on predict-and-optimize learning while addressing a practical virtual power plant decision problem.

## Data Availability Statement

The OPSD public decision benchmark [16] is reproducible from `run_public_opsd_baselines.py`; the extended risk simulator is reproducible from `run_opsd_vpp_risk_simulator.py`; the held-out decision-focused policy search is reproducible from `run_opsd_decision_focused_policy_search.py`; the forecast-coupled VPP experiment is reproducible from `run_opsd_forecast_coupled_vpp.py`; the risk-aversion sensitivity experiment is reproducible from `run_opsd_risk_aversion_sensitivity.py`; the reviewer-facing robustness checks are reproducible from `run_opsd_vpp_reviewer_robustness.py`; and the fuzzy soft-computing comparator is reproducible from `run_opsd_fuzzy_risk_vpp_baseline.py`. Local Hunan and Shandong operational records are retained only as non-public application-context evidence and are not part of the manuscript's public reproducibility claim.

## Local and China Real-World Data Assets

The newly inventoried local authorized real-world data directory gives the decision-focused virtual power plant paper a stronger China-market application layer. The audit file `paper_package/master_submission_control/real_world_data_inventory_and_paper_mapping.md` records Liaoning and Eastern Inner Mongolia quarter-hour market disclosure data, Shandong market/load/weather/coal files, and open-license China grid/resource metadata.

|family|files|total_mib|status|main_use|risk|
|---|---|---|---|---|---|
|China power grid multi-year transmission network|6|0.226|Figshare CC BY 4.0 for core network dataset; GEM UHV reference folder has non-commercial caveat|Dissertation background, graph construction motivation, resource/network metadata appendix; optional Paper 1 graph-prior discussion|mixed license: keep GEM NC materials out of journal supplements unless permission is confirmed|
|Eastern Inner Mongolia public market disclosure|5|4.063|public-market-disclosure/local-copy; verify portal license before redistribution|Paper 1 cross-market price forecasting; Paper 3 decision-focused bidding; dissertation regional comparison|low confidentiality if sourced from public disclosure, but source-page citation and license check are required|
|Liaoning public market disclosure|12|26.385|public-market-disclosure/local-copy; verify portal license before redistribution|Paper 1 price forecasting; Paper 2 regional load forecasting; Paper 3 VPP decision case; dissertation China market chapter|low confidentiality if sourced from public disclosure, but final supplement redistribution still needs source-page citation and license check|
|OSM China power grid GIS extraction|13|598.263|OpenStreetMap-derived local extraction; several source files are incomplete .downloading files|Dissertation background only unless extraction is completed and ODbL attribution/share-alike handling is reviewed|license and incomplete-download risk; do not use as a primary journal experiment yet|
|Shandong market, load, weather, coal and consumption data|8|52.183|local real-world market data; treat as non-public until source-page and redistribution rights are confirmed|Paper 1 local price/weather/fuel case; Paper 2 load-weather case; Paper 3 settlement and VPP decision case; dissertation applied validation|medium: may include market-operation details; use aggregated or anonymized statistics in manuscripts until permission is clear|
|WRI/GEM China power plant metadata and time-series summaries|18|20.214|WRI Global Power Plant Database is CC BY 4.0; GEM links require final use-permission review|Dissertation resource-mix context, VPP portfolio scenario design, Paper 3 resource assumptions appendix|WRI is usable with attribution; GEM commercial/non-commercial caveats must be separated|

For Paper 3, the most relevant variables are bidding space, renewable output and forecasts, tie-line plans, non-market unit output, day-ahead price, real-time price, actual load and forecast load. These variables can define a realistic forecast-then-optimize environment for storage, flexible load and market bidding. The public OPSD simulator remains the reviewable benchmark; the China data should be used for anonymized stress tests, parameter calibration, market-state motivation and dissertation case-study discussion until permission and source licenses are confirmed.

The immediate modeling update is to build a normalized decision table at quarter-hour or hourly resolution with market, timestamp, price, load, renewable, tie-line and bidding-space features. That table can drive a China application version of the revenue, regret, loss-day and CVaR evaluation already used in the public simulator.

This update has started: `china_real_world_data/china_market_disclosure_quarterhour_wide.csv` normalizes 92,576 quarter-hour records, and `china_real_world_data/china_market_disclosure_baseline_report.md` reports a simplified storage-arbitrage decision baseline for Liaoning and Eastern Inner Mongolia. The decision evidence is currently a local application baseline, not a public supplement result.

The Shandong price layer adds a longer day-ahead/real-time decision baseline in `china_real_world_data/shandong_real_data_baseline_report.md`. Over 1,364 daily price-spread proxy days, hindsight daily arbitrage reaches 449.885 yuan/MWh-spread units on average, while day-ahead forecast-then-optimize reaches 393.369, leaving mean regret 56.517 and 14 negative forecast-then-optimize days. This gives Paper 3 a concrete China-market motivation for decision-focused learning: even when the day-ahead signal is useful, downstream market value still depends on forecast error shape and risk exposure.

A separate GIS infrastructure audit, `paper_package/master_submission_control/gis_energy_infrastructure_evidence.md`, profiles China transmission-network snapshots, OSM mainland power-facility extractions, WRI China power-plant metadata and the GEM integrated China power-facility table. The current derived summaries contain 12,839 2025 transmission-line records, 2,041 substation records, 9,444 grid-link records, 4,870,459 OSM mainland power records, 4,274 WRI China plant records and approximately 3,108.905 GW of represented GEM operating capacity. For Paper 3, these records are used as infrastructure-context evidence and graph/resource heterogeneity motivation only. They are not treated as target labels or as a model-training dataset for the public benchmark experiments, and raw OSM/GEM/SHP/PBF files remain excluded from public supplements until attribution, license and redistribution checks are closed.

![Figure 10. China real-market decision baseline. Daily price-spread proxy revenue is compared between hindsight and day-ahead forecast-then-optimize policies for Eastern Inner Mongolia, Liaoning and Shandong.](figures/paper3_fig10_china_real_vpp_decision_baselines.png)

![Figure 11. China GIS infrastructure context for network-scale virtual power plant decision making. The figure summarizes 2015/2020/2025 grid snapshots, OSM mainland power-facility extractions and open plant/resource metadata as scenario-context evidence rather than as a training target.](figures/paper3_fig14_china_grid_gis_externality.png)


## References

[1] Dimitris Bertsimas, Nathan Kallus, "From Predictive to Prescriptive Analytics", Management Science, vol. 66, no. 3, pp. 1025-1044, 2020, https://doi.org/10.1287/mnsc.2018.3253.
[2] Adam N. Elmachtoub, Paul Grigas, "Smart “Predict, then Optimize”", Management Science, vol. 68, no. 1, pp. 9-26, 2022, https://doi.org/10.1287/mnsc.2020.3922.
[3] Amos, Brandon, Kolter, J. Zico, "OptNet: Differentiable Optimization as a Layer in Neural Networks," Proceedings of the 34th International Conference on Machine Learning, 2017. https://proceedings.mlr.press/v70/amos17a.html.
[4] Donti, Priya L., Amos, Brandon, Kolter, J. Zico, "Task-based End-to-end Model Learning in Stochastic Optimization," Advances in Neural Information Processing Systems, 2017. https://papers.neurips.cc/paper/7132-task-based-end-to-end-model-learning-in-stochastic-optimization.
[5] Bryan Wilder, Bistra Dilkina, Milind Tambe, "Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization", Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 01, pp. 1658-1665, 2019, https://doi.org/10.1609/aaai.v33i01.33011658.
[6] R. Tyrrell Rockafellar, Stanislav Uryasev, "Optimization of conditional value-at-risk", The Journal of Risk, vol. 2, no. 3, pp. 21-41, 2000, https://doi.org/10.21314/JOR.2000.038.
[7] Stephen Boyd, Lieven Vandenberghe, "Convex Optimization", 2004, https://doi.org/10.1017/CBO9780511804441.
[8] Anastasios N. Angelopoulos, Stephen Bates, "Conformal Prediction: A Gentle Introduction", Foundations and Trends® in Machine Learning, vol. 16, no. 4, pp. 494-591, 2023, https://doi.org/10.1561/2200000101.
[9] Bryan Lim, Sercan Ö. Arık, Nicolas Loeff, Tomas Pfister, "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting", International Journal of Forecasting, vol. 37, no. 4, pp. 1748-1764, 2021, https://doi.org/10.1016/j.ijforecast.2021.03.012.
[10] Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron, "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark", Applied Energy, vol. 293, pp. 116983, 2021, https://doi.org/10.1016/j.apenergy.2021.116983.
[11] Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli, Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol. 32, no. 3, pp. 896-913, 2016, https://doi.org/10.1016/j.ijforecast.2016.02.001.
[12] Hossein Mohammadi Rouzbahani, Hadis Karimipour, Lei Lei, "A review on virtual power plant for energy management", Sustainable Energy Technologies and Assessments, vol. 47, pp. 101370, 2021, https://doi.org/10.1016/j.seta.2021.101370.
[13] Seyyed Mostafa Nosratabadi, Rahmat-Allah Hooshmand, Eskandar Gholipour, "A comprehensive review on microgrid and virtual power plant concepts employed for distributed energy resources scheduling in power systems", Renewable and Sustainable Energy Reviews, vol. 67, pp. 341-363, 2017, https://doi.org/10.1016/j.rser.2016.09.025.
[14] Hieu Trung Nguyen, Long Bao Le, Zhaoyu Wang, "A Bidding Strategy for Virtual Power Plants With the Intraday Demand Response Exchange Market Using the Stochastic Programming", IEEE Transactions on Industry Applications, vol. 54, no. 4, pp. 3044-3055, 2018, https://doi.org/10.1109/TIA.2018.2828379.
[15] Morteza Shafiekhani, Ali Badri, Miadreza Shafie-khah, João P.S. Catalão, "Strategic bidding of virtual power plant in energy markets: A bi-level multi-objective approach", International Journal of Electrical Power & Energy Systems, vol. 113, pp. 208-219, 2019, https://doi.org/10.1016/j.ijepes.2019.05.023.
[16] {Open Power System Data}, "Time series data package," Open Power System Data, 2020. https://data.open-power-system-data.org/time_series/2020-10-06/.
[17] L.A. Zadeh, "Fuzzy sets", Information and Control, vol. 8, no. 3, pp. 338-353, 1965, https://doi.org/10.1016/S0019-9958(65)90241-X.
[18] Tomohiro Takagi, Michio Sugeno, "Fuzzy identification of systems and its applications to modeling and control", IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-15, no. 1, pp. 116-132, 1985, https://doi.org/10.1109/TSMC.1985.6313399.
[19] J.-S.R. Jang, "ANFIS: adaptive-network-based fuzzy inference system", IEEE Transactions on Systems, Man, and Cybernetics, vol. 23, no. 3, pp. 665-685, 1993, https://doi.org/10.1109/21.256541.
[20] Samarjit Kar, Sujit Das, Pijush Kanti Ghosh, "Applications of neuro fuzzy systems: A brief review and future outline", Applied Soft Computing, vol. 15, pp. 243-259, 2014, https://doi.org/10.1016/j.asoc.2013.10.014.
[21] Ali T. Al-Awami, Nemer A. Amleh, Ammar M. Muqbel, "Optimal Demand Response Bidding and Pricing Mechanism With Fuzzy Optimization: Application for a Virtual Power Plant", IEEE Transactions on Industry Applications, vol. 53, no. 5, pp. 5051-5061, 2017, https://doi.org/10.1109/TIA.2017.2723338.
[22] José R. Vázquez-Canteli, Zoltán Nagy, "Reinforcement learning for demand response: A review of algorithms and modeling techniques", Applied Energy, vol. 235, pp. 1072-1089, 2019, https://doi.org/10.1016/j.apenergy.2018.11.002.
[23] Rafał Weron, "Electricity price forecasting: A review of the state-of-the-art with a look into the future", International Journal of Forecasting, vol. 30, no. 4, pp. 1030-1081, 2014, https://doi.org/10.1016/j.ijforecast.2014.08.008.
[24] Jakub Nowotarski, Rafał Weron, "Recent advances in electricity price forecasting: A review of probabilistic forecasting", Renewable and Sustainable Energy Reviews, vol. 81, pp. 1548-1568, 2018, https://doi.org/10.1016/j.rser.2017.05.234.
[25] Hrvoje Pandžić, Juan M. Morales, Antonio J. Conejo, Igor Kuzle, "Offering model for a virtual power plant based on stochastic programming", Applied Energy, vol. 105, pp. 282-292, 2013, https://doi.org/10.1016/j.apenergy.2012.12.077.
[26] Elaheh Mashhour, Seyed Masoud Moghaddas-Tafreshi, "Bidding Strategy of Virtual Power Plant for Participating in Energy and Spinning Reserve Markets—Part I: Problem Formulation", IEEE Transactions on Power Systems, vol. 26, no. 2, pp. 949-956, 2011, https://doi.org/10.1109/TPWRS.2010.2070884.
[27] Seyyed Mostafa Nosratabadi, Rahmat-Allah Hooshmand, Eskandar Gholipour, "Stochastic profit-based scheduling of industrial virtual power plant using the best demand response strategy", Applied Energy, vol. 164, pp. 590-606, 2016, https://doi.org/10.1016/j.apenergy.2015.12.024.
[28] Imran Rahman, Junita Mohamad-Saleh, "Hybrid bio-Inspired computational intelligence techniques for solving power system optimization problems: A comprehensive survey", Applied Soft Computing, vol. 69, pp. 72-130, 2018, https://doi.org/10.1016/j.asoc.2018.04.051.
[29] Wei Dong, Qiang Yang, Xinli Fang, Wei Ruan, "Adaptive optimal fuzzy logic based energy management in multi-energy microgrid considering operational uncertainties", Applied Soft Computing, vol. 98, pp. 106882, 2021, https://doi.org/10.1016/j.asoc.2020.106882.
[30] Enrico De Santis, Antonello Rizzi, Alireza Sadeghian, "Hierarchical genetic optimization of a fuzzy logic system for energy flows management in microgrids", Applied Soft Computing, vol. 60, pp. 135-149, 2017, https://doi.org/10.1016/j.asoc.2017.05.059.
[31] Jayanta Mandi, James Kotary, Senne Berden, Maxime Mulamba, Victor Bucarey, Tias Guns, Ferdinando Fioretto, "Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities", Journal of Artificial Intelligence Research, vol. 80, pp. 1623-1701, 2024, https://doi.org/10.1613/jair.1.15320.
[32] Hongchao Gao, Tai Jin, Cheng Feng, Chuyi Li, Qixin Chen, Chongqing Kang, "Review of virtual power plant operations: Resource coordination and multidimensional interaction", Applied Energy, vol. 357, pp. 122284, 2024, https://doi.org/10.1016/j.apenergy.2023.122284.
[33] Yuanzheng Li, Chaofan Yu, Mohammad Shahidehpour, Tao Yang, Zhigang Zeng, Tianyou Chai, "Deep Reinforcement Learning for Smart Grid Operations: Algorithms, Applications, and Prospects", Proceedings of the IEEE, vol. 111, no. 9, pp. 1055-1096, 2023, https://doi.org/10.1109/JPROC.2023.3303358.
[34] Aharon Ben-Tal, Laurent El Ghaoui, Arkadi Nemirovski, "Robust Optimization", 2009, https://doi.org/10.1515/9781400831050.
[35] John R. Birge, François Louveaux, "Introduction to Stochastic Programming", Springer Series in Operations Research and Financial Engineering, 2011, https://doi.org/10.1007/978-1-4614-0237-4.
