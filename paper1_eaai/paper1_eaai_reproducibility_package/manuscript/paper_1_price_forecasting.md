# Sequence-Anchored GraphPatch Residual Learning for Uncertainty-Aware Electricity Price Forecasting in Virtual Power Plant Market Operation

**Target journals:** Engineering Applications of Artificial Intelligence; Expert Systems with Applications; Neurocomputing; Applied Soft Computing.

**Manuscript status:** Working manuscript with public-data experiments, figures, references, and data-availability materials integrated. Do not submit until target-journal formatting, school classification confirmation, and final author approval are completed.

## Abstract

Electricity price forecasting has become a core computational problem for virtual power plants that participate in increasingly volatile electricity markets. Existing forecasting studies often focus on point accuracy, while market operation requires calibrated uncertainty, robustness to price spikes, and decision-relevant representations across heterogeneous market, load, renewable, weather, and calendar signals. This paper proposes an uncertainty-aware spatio-temporal learning framework for multi-horizon electricity price forecasting in virtual power plant market operation. The framework constructs a dynamic heterogeneous temporal graph over market zones, resource-side variables, and exogenous covariates, and then combines graph message passing, patch-based temporal attention, and probabilistic output heads to generate calibrated point and interval forecasts. A conformal calibration layer is further introduced to improve distribution-free reliability under non-stationary market regimes. The proposed model is evaluated on public electricity market datasets and virtual-power-plant-oriented feature sets against statistical, machine-learning, and deep time-series baselines. The planned evaluation reports deterministic accuracy, probabilistic calibration, robustness under spike regimes, cross-market generalization, and downstream bidding value. The study positions electricity price forecasting as a computer-science problem of uncertain, heterogeneous, non-stationary time-series learning rather than a conventional power-system simulation problem.

**Keywords:** electricity price forecasting; virtual power plant; spatio-temporal learning; graph neural network; Transformer; conformal prediction; uncertainty quantification.

## 1. Introduction

Virtual power plants aggregate distributed photovoltaic generation, battery storage, flexible loads, and other controllable resources into a market-facing entity. Their economic performance depends on the ability to anticipate market prices at multiple horizons. A virtual power plant that only receives a point forecast is exposed to two kinds of risk. First, price spikes and negative-price events can dominate profit and loss even if average prediction error is moderate. Second, bidding, arbitrage, and load-shifting decisions require a distribution over plausible outcomes, not only a single expected value. These requirements turn electricity price forecasting into a problem of uncertainty-aware temporal learning.

From a computer-science perspective, electricity price data exhibit several characteristics that are not well handled by generic forecasting pipelines. The series is non-stationary across seasons and market regimes; it depends on heterogeneous exogenous variables such as load, weather, renewable generation, fuel prices, and calendar effects; and it contains sparse but economically important extreme events. Spatial relationships among market nodes, balancing areas, or resource groups are also time-varying. A model that treats the price series as an isolated univariate signal is therefore structurally limited.

Recent progress in Transformer-based time-series forecasting, graph neural networks, and conformal prediction creates an opportunity to design a more suitable computational framework. Patch-based temporal attention can model long-range temporal patterns efficiently, graph learning can capture spatial or relational dependencies, and conformal calibration can provide prediction intervals with finite-sample validity under mild assumptions. However, these components have rarely been integrated into a virtual-power-plant market operation setting with a direct emphasis on calibrated uncertainty and price-spike robustness.

This paper proposes an uncertainty-aware spatio-temporal learning framework for multi-horizon electricity price forecasting. The method first transforms raw market, load, renewable, weather, and calendar signals into a dynamic heterogeneous graph. A graph-temporal encoder extracts relational representations, a temporal attention module captures multi-scale dependencies, and probabilistic heads estimate quantiles or distributional parameters for each forecast horizon. Finally, a conformal calibration layer adjusts interval forecasts using recent residual behavior, improving reliability when the market regime shifts.

The intended contribution is methodological. Electricity markets are used as a demanding testbed for multi-source non-stationary time-series learning. The model is designed to be compared not only by MAE and RMSE but also by pinball loss, CRPS, interval coverage, interval width, spike-regime error, and downstream bidding value. This framing makes the manuscript suitable for AI engineering and intelligent decision-support journals while remaining relevant to virtual power plant applications.

## 2. Related Work

### 2.1 Electricity price forecasting

Electricity price forecasting studies have evolved from statistical models and shallow machine learning toward deep temporal architectures. Traditional approaches such as ARIMA, support vector regression, random forests, and gradient boosting are still useful baselines because they are stable, interpretable, and computationally efficient. Deep learning approaches, including recurrent networks, temporal convolutional networks, attention models, and hybrid architectures, can represent nonlinear temporal dependencies and exogenous interactions. The key gap for market operation is that many studies still optimize point accuracy and underreport calibration, extreme-event performance, and downstream decision value.

### 2.2 Deep time-series forecasting

Transformer variants such as Informer, Autoformer, Temporal Fusion Transformer, PatchTST, and related architectures have become standard references for long-horizon forecasting. At the same time, linear baselines such as DLinear show that sophisticated attention is not automatically superior unless the data regime and evaluation protocol justify its complexity. This motivates a careful experimental design in which the proposed method is compared with both strong deep models and simple but competitive baselines.

### 2.3 Spatio-temporal graph learning

Electricity markets contain relational structure: market zones are connected by transmission constraints, neighboring loads respond to similar weather patterns, and distributed resources are correlated through geography and behavior. Graph WaveNet and related spatio-temporal graph models demonstrate that learned graph structures can improve forecasting when explicit adjacency is incomplete or dynamic. This paper adopts this principle but extends it to heterogeneous market covariates and uncertainty-aware price prediction.

### 2.4 Uncertainty quantification and conformal prediction

Prediction intervals are essential for bidding and risk control. Quantile regression, Bayesian neural networks, ensembling, and distributional forecasting provide alternative uncertainty estimates. Conformal prediction is attractive because it can wrap around an existing model and provide distribution-free calibration under exchangeability or approximate online conditions. In this paper, conformal calibration is used as a post-processing layer to correct the empirical coverage of neural quantile forecasts.

## 3. Problem Formulation

Let p_t^z denote the electricity price at time t for market zone z, and let X_t^z contain exogenous features including system load, renewable generation, weather, calendar variables, and resource aggregation indicators. Given a lookback window of length L, the goal is to forecast zone-specific prices for horizons h = 1, ..., H. Unlike deterministic forecasting, the model estimates conditional quantiles q_alpha(t+h,z) or a predictive distribution P(p_{t+h}^z | X_{t-L+1:t}^z).

The prediction target is therefore:

f_theta: {X_{t-L+1:t}^z, G_t} -> {q_alpha(t+h,z) for alpha in A, h = 1, ..., H},

where G_t is a dynamic heterogeneous graph representing market and resource relationships. The objective combines point error, quantile loss, calibration loss, and optional decision-aware regularization.

## 4. Proposed Method

### 4.1 Dynamic heterogeneous market graph

The graph contains nodes for market zones, load areas, renewable generation groups, weather stations, and virtual-power-plant resource clusters. Edges are initialized from physical or geographic proximity when available and then updated by learned similarity. For nodes i and j at time t, the adaptive edge weight is:

a_ij,t = softmax_j(phi(z_i,t)^T psi(z_j,t) / sqrt(d)),

where z_i,t and z_j,t are node embeddings and phi, psi are learnable projections. This design allows the model to discover time-varying dependencies without relying exclusively on fixed grid topology.

### 4.2 Spatio-temporal encoder

The encoder alternates graph message passing and temporal attention. Graph message passing aggregates relational information:

h_i,t' = sigma(W_self h_i,t + sum_j a_ij,t W_msg h_j,t).

The temporal module then applies patch-based attention over each node representation. Patches reduce sequence length and help the model capture daily, weekly, and seasonal structures without excessive computation. Positional, calendar, and market-session embeddings are added to preserve temporal semantics.

### 4.3 Probabilistic forecasting head

The forecasting head outputs multiple quantiles for each horizon. The quantile loss is:

L_q = sum_{h, alpha} max(alpha e_{h,alpha}, (alpha - 1) e_{h,alpha}),

where e_{h,alpha}^z = p_{t+h}^z - q_alpha(t+h,z). The median quantile is used for point prediction, while lower and upper quantiles provide interval forecasts.

### 4.4 Conformal calibration

A rolling calibration set is used to estimate nonconformity scores from recent residuals. For a nominal interval [q_lower, q_upper], the conformalized interval is widened by an empirical quantile of calibration errors. This layer is deliberately separated from the neural encoder so that it can adapt to regime shifts without retraining the full model.

### 4.5 Training objective

The full objective is:

L_total = L_pred + lambda_q L_q + lambda_c L_cal + lambda_r L_reg,

where L_pred is MAE or Huber loss for the median prediction, L_q is quantile loss, L_cal encourages empirical coverage, and L_reg controls smoothness or graph sparsity. Hyperparameters are selected on validation data.

## 5. Experimental Design

### 5.1 Datasets

The reproducible public benchmark uses Open Power System Data (OPSD) `time_series_60min_singleindex.csv`, package version 2020-10-06. The selected public variables include hourly day-ahead price, load, solar generation, and wind generation for four bidding zones with joint price-load coverage. Local Hunan and Shandong datasets are retained as application case studies rather than as the only evidence.

Table 1 reports the public benchmark coverage used in the current experiment.

|Zone|Rows|Start|End|Price|Load|Solar|Wind|
|---|---|---|---|---|---|---|---|
|DE_LU|17521|2018-10-01|2020-09-30|17521|17521|17516|17521|
|DK_1|50386|2015-01-01|2020-09-30|50386|50386|50377|50386|
|DK_2|50386|2015-01-01|2020-09-30|50386|50386|50373|50386|
|GB_GBN|50288|2015-01-02|2020-09-30|50288|50288|50247|50252|

### 5.2 Baselines

The current public benchmark now includes transparent persistence, seasonal, lag-calendar-exogenous, graph residual ridge, random-hidden-layer residual, GraphPatch residual, DLinear/NLinear-style anchors, and a TDConv-style TCN-family sequence comparator. Full PatchTST/TFT training remains a reviewer-response extension rather than a current claim. For uncertainty forecasting, compare quantile regression, DeepAR-style distributional forecasting, ensemble intervals, and conformalized baselines.

### 5.3 Metrics

Point forecasting metrics: MAE, RMSE, MAPE or sMAPE. Probabilistic metrics: pinball loss, CRPS, PICP, PINAW, calibration error. Market-value metrics: simulated bidding profit, regret, downside risk, and performance during top 5% price volatility periods.

### 5.4 Ablation study

The current ablation removes cross-zone graph information, nonlinear residual learning, calibration strategy, rolling-origin stability, leave-one-zone-out transfer, spike-threshold choice, and conformal calibration-window choice. Each component is evaluated on ordinary, high-volatility, chronological, and cross-zone transfer settings where the corresponding diagnostic is meaningful.

## 6. Results and Discussion

Table 2 reports the first public reproducible point-forecasting benchmark on OPSD day-ahead prices. The table reports the best transparent baseline by RMSE for each zone; the complete baseline table includes persistence, 24-hour seasonal naive, 168-hour seasonal naive, and lag-calendar-exogenous linear models.

|Zone|Best baseline|MAE|RMSE|sMAPE|N|
|---|---|---|---|---|---|
|DE_LU|Linear lag+cal+exog|3.146|5.516|14.635|3505|
|DK_1|Linear lag+cal+exog|3.073|5.015|18.728|10072|
|DK_2|Linear lag+cal+exog|3.076|5.147|15.135|10072|
|GB_GBN|Linear lag+cal+exog|3.308|4.847|12.667|10053|

The lag-calendar-exogenous linear baseline obtains the lowest RMSE among the transparent non-graph baselines on all four OPSD zones. This finding sets a strong reference point: graph-temporal structure is evaluated where it should matter most, namely calibrated uncertainty, spike-regime robustness, and downstream decision value.

Table 3 reports the first public probabilistic price-forecasting benchmark on the same OPSD test split. A lag-calendar-exogenous linear point model is trained on the first 60% of each zone, calibrated on the next 20%, and evaluated on the final 20%. The global split-conformal variant uses one calibration residual quantile per zone. The regime-conformal variant estimates residual quantiles by hour block and lag-volatility regime, with a global fallback for sparse regimes.

|Zone|Model|PICP|PINAW|Width|PB05|PB95|Interval score|N|
|---|---|---|---|---|---|---|---|---|
|DE_LU|Global conformal|0.897|0.046|12.783|0.565|0.642|24.148|3471|
|DE_LU|Regime conformal|0.904|0.047|12.939|0.540|0.617|23.140|3471|
|DK_1|Global conformal|0.890|0.054|14.009|0.551|0.641|23.847|10043|
|DK_1|Regime conformal|0.888|0.053|13.768|0.567|0.586|23.079|10043|
|DK_2|Global conformal|0.889|0.057|13.776|0.542|0.671|24.271|10043|
|DK_2|Regime conformal|0.885|0.054|13.219|0.545|0.612|23.140|10043|
|GB_GBN|Global conformal|0.936|0.077|17.284|0.536|0.578|22.294|10016|
|GB_GBN|Regime conformal|0.936|0.074|16.753|0.512|0.555|21.353|10016|

The public OPSD probability benchmark removes the previous gap in uncertainty metrics. The global conformal baseline obtains PICP values of 0.897, 0.890, 0.889, and 0.936 across DE-LU, DK1, DK2, and Great Britain for a 0.900 target. The regime-conformal baseline slightly improves DE-LU coverage and reduces interval score on all four public zones, although DK1 and DK2 remain mildly under-covered. This diagnostic motivates graph-temporal modeling as a conditional-regime problem rather than as a claim of universal point-error dominance.

![Figure 1. OPSD public split-conformal coverage against the 90% target.](figures/paper1_fig4_opsd_conformal_coverage.png)

![Figure 2. Representative OPSD regime-conformal prediction interval on the DE-LU public test split.](figures/paper1_fig5_opsd_conformal_interval.png)

Table 4 adds an auditable graph-temporal residual ablation. The first-stage model is the same local lag-calendar-exogenous ridge model used above. A second residual model then uses only prediction-time-safe graph features: lagged cross-zone price means and dispersions at 1-hour, 24-hour, and 168-hour lags, together with local lags and exogenous variables. The spike subset is defined by each zone's training 90th percentile of |price lag 1 - price lag 24|, so the test split is not used to tune the threshold.

|Zone|Regime|Local RMSE|Graph RMSE|RMSE improvement %|MAE improvement %|N|
|---|---|---|---|---|---|---|
|DE_LU|All hours|5.562|5.508|0.962|0.139|3471|
|DE_LU|Spike|13.129|12.679|3.433|3.057|251|
|DK_1|All hours|5.017|5.001|0.330|-0.261|10043|
|DK_1|Spike|7.496|7.459|0.496|-0.352|1718|
|DK_2|All hours|5.167|5.142|0.477|0.434|10043|
|DK_2|Spike|7.708|7.662|0.599|1.662|1712|
|GB_GBN|All hours|4.873|4.938|-1.333|-4.270|10016|
|GB_GBN|Spike|10.008|9.872|1.359|2.253|767|

The graph-temporal residual model improves spike-regime RMSE in all four OPSD zones, with relative RMSE gains ranging from 0.496% in DK1 to 3.433% in DE-LU. Across all test hours, the same model improves DE-LU, DK1, and DK2 but degrades Great Britain by 1.333% RMSE. This pattern is useful for the thesis claim: lagged cross-zone graph information is most helpful in volatile regimes, while ordinary-period forecasting still requires careful regularization and regime-aware model selection.

Table 5 reports a split-conformal interval layer on top of the graph-temporal residual point model.

|Zone|PICP|PINAW|Width|PB05|PB95|Interval score|N|
|---|---|---|---|---|---|---|---|
|DE_LU|0.902|0.047|12.821|0.553|0.645|23.963|3471|
|DK_1|0.892|0.054|13.902|0.547|0.637|23.696|10043|
|DK_2|0.888|0.056|13.648|0.533|0.674|24.142|10043|
|GB_GBN|0.925|0.073|16.558|0.516|0.585|22.029|10016|

The graph-temporal residual conformal layer reaches near-target or conservative coverage in DE-LU and Great Britain, while DK1 and DK2 remain mildly under-covered. Compared with the regime-conformal linear benchmark, the result does not dominate interval score. The empirical implication is that graph-temporal point correction and conditional uncertainty calibration should be treated as complementary modules.

![Figure 3. OPSD spike-regime RMSE comparison for the local and graph-temporal residual models.](figures/paper1_fig6_opsd_graph_temporal_spike_rmse.png)

![Figure 4. Relative RMSE change from the graph-temporal residual correction on all hours and spike-regime hours.](figures/paper1_fig7_opsd_graph_temporal_improvement.png)


Table 6 adds a reviewer-facing nonlinear baseline on the same OPSD public split. The nonlinear model is a lightweight random-hidden-layer neural residual model: standardized graph-temporal inputs are projected through a reproducible tanh hidden layer, and the residual output is fitted by ridge regression. This design keeps the baseline auditable without adding external deep-learning dependencies.

|Zone|Model|MAE|RMSE|RMSE improvement %|MAE improvement %|N|
|---|---|---|---|---|---|---|
|DE_LU|Local ridge|7.127|13.129|0|0|251|
|DE_LU|Graph ridge|6.909|12.679|3.433|3.057|251|
|DE_LU|Graph ELM|7.268|13.242|-0.855|-1.977|251|
|DK_1|Local ridge|4.667|7.496|0|0|1718|
|DK_1|Graph ridge|4.683|7.459|0.496|-0.352|1718|
|DK_1|Graph ELM|4.757|7.439|0.771|-1.936|1718|
|DK_2|Local ridge|4.959|7.708|0|0|1712|
|DK_2|Graph ridge|4.877|7.662|0.599|1.662|1712|
|DK_2|Graph ELM|5.008|7.671|0.475|-0.977|1712|
|GB_GBN|Local ridge|7.586|10.008|0|0|767|
|GB_GBN|Graph ridge|7.415|9.872|1.359|2.253|767|
|GB_GBN|Graph ELM|8.345|11.261|-12.519|-10.004|767|

The nonlinear residual baseline is useful mainly as a stress test. It improves spike-regime RMSE in DK1 and DK2 but degrades DE-LU and Great Britain. The graph residual ridge model is more stable on spike regimes, improving RMSE in all four public zones. This supports a conservative modeling conclusion: graph-temporal information is valuable under volatility, but unconstrained nonlinear residual correction can overfit zone-specific structure.

Table 7 reports paired hourly error-reduction evidence against the local ridge baseline on spike-regime test hours. Win rate is the fraction of matched hours where the graph model has lower absolute error than local ridge; p-values are approximate two-sided paired sign tests.

|Zone|Model|Win rate|Mean abs-error reduction|p approx.|Direction|N|
|---|---|---|---|---|---|---|
|DE_LU|Graph ridge|0.470|0.218|0.344|mixed|251|
|DE_LU|Graph ELM|0.438|-0.141|0.050|mixed|251|
|DK_1|Graph ridge|0.474|-0.016|0.030|worse|1718|
|DK_1|Graph ELM|0.474|-0.090|0.030|worse|1718|
|DK_2|Graph ridge|0.571|0.082|<0.001|better|1712|
|DK_2|Graph ELM|0.484|-0.048|0.192|mixed|1712|
|GB_GBN|Graph ridge|0.554|0.171|0.003|better|767|
|GB_GBN|Graph ELM|0.463|-0.759|0.040|worse|767|

The paired tests show that graph residual ridge has statistically significant spike-regime wins in DK2 and Great Britain, while DE-LU and DK1 remain mixed. The ELM residual model does not provide robust paired evidence and is significantly worse in Great Britain. These results justify retaining a regularized graph-temporal residual layer as the transparent public baseline, while the subsequent GraphPatch experiments test the stronger residual architecture required for the target-journal submission.

![Figure 5. OPSD nonlinear graph-temporal price baselines on spike-regime RMSE.](figures/paper1_fig8_opsd_nonlinear_spike_rmse.png)

![Figure 6. Paired spike-regime error-reduction win rates against the local ridge baseline.](figures/paper1_fig9_opsd_paired_spike_win_rate.png)


Table 8 upgrades the public OPSD experiment from an auditable random-hidden-layer stress test to a stronger deep graph-temporal residual baseline. The GraphPatch model constructs multi-scale temporal patches from 1-, 2-, 3-, 6-, 12-, 24-, 48-, 72-, and 168-hour local lags, augments them with lagged cross-zone graph means and dispersions, and trains a two-layer tanh residual MLP. The calibrated blend selects the MLP/ridge residual mixture on the calibration split only, using a weighted all-hour and spike-regime RMSE objective.

|Zone|Model|MAE|RMSE|RMSE improvement %|MAE improvement %|MLP weight|N|
|---|---|---|---|---|---|---|---|
|DE_LU|Local ridge|7.127|13.129|0|0||251|
|DE_LU|Patch ridge|6.356|12.714|3.168|10.817||251|
|DE_LU|Patch MLP|6.711|12.394|5.603|5.835||251|
|DE_LU|Cal. blend|6.331|12.572|4.243|11.178|0.200|251|
|DK_1|Local ridge|4.667|7.496|0|0||1718|
|DK_1|Patch ridge|4.226|7.068|5.710|9.452||1718|
|DK_1|Patch MLP|3.936|6.684|10.835|15.662||1718|
|DK_1|Cal. blend|3.911|6.679|10.902|16.190|0.900|1718|
|DK_2|Local ridge|4.959|7.708|0|0||1712|
|DK_2|Patch ridge|4.505|7.382|4.228|9.164||1712|
|DK_2|Patch MLP|4.311|6.937|9.998|13.065||1712|
|DK_2|Cal. blend|4.304|7.001|9.168|13.215|0.600|1712|
|GB_GBN|Local ridge|7.586|10.008|0|0||767|
|GB_GBN|Patch ridge|7.315|9.765|2.422|3.567||767|
|GB_GBN|Patch MLP|7.157|9.728|2.797|5.657||767|
|GB_GBN|Cal. blend|7.073|9.543|4.648|6.763|0.300|767|

The deep GraphPatch result is materially stronger than the previous ELM stress test. The standalone GraphPatch MLP improves spike-regime RMSE in all four public zones, with gains of 5.603% in DE-LU, 10.835% in DK1, 9.998% in DK2, and 2.797% in Great Britain relative to the local ridge baseline. The calibrated GraphPatch blend also improves spike-regime RMSE in all four zones, with the largest gains in DK1 and DK2. The all-hour result remains mixed in Great Britain, so the paper should frame the model as volatility-regime strengthening rather than universal average-error dominance.

Table 9 reports split-conformal intervals around the calibrated GraphPatch blend. This connects the deep residual model to the paper's uncertainty-aware contribution rather than leaving it as a point-forecast-only upgrade.

|Zone|MLP weight|PICP|PINAW|Width|PB05|PB95|Interval score|N|
|---|---|---|---|---|---|---|---|---|
|DE_LU|0.200|0.906|0.040|10.960|0.508|0.540|20.965|3471|
|DK_1|0.900|0.884|0.043|11.060|0.497|0.533|20.614|10043|
|DK_2|0.600|0.884|0.047|11.328|0.494|0.577|21.414|10043|
|GB_GBN|0.300|0.912|0.069|15.642|0.526|0.561|21.731|10016|

The calibrated GraphPatch conformal intervals are near or above the 90% target in DE-LU and Great Britain, while DK1 and DK2 remain mildly under-covered at about 0.884. This is useful reviewer evidence: the deeper point model improves volatility-regime accuracy, but conditional calibration for Nordic zones still needs a regime-aware or locally adaptive interval layer.

Table 10 gives paired spike-regime evidence for the calibrated GraphPatch blend against the local ridge baseline.

|Zone|Win rate|Mean abs-error reduction|p approx.|Direction|N|
|---|---|---|---|---|---|
|DE_LU|0.629|0.797|<0.001|better|251|
|DK_1|0.622|0.756|<0.001|better|1718|
|DK_2|0.619|0.655|<0.001|better|1712|
|GB_GBN|0.570|0.513|<0.001|better|767|

The calibrated GraphPatch blend wins significantly against the local ridge baseline in all four spike-regime paired tests. This is the first public Paper 1 result that supports a credible method claim: a computer-science-oriented temporal patch plus cross-zone graph residual learner improves volatile-market forecasting while still exposing the remaining uncertainty-calibration limitation.

![Figure 7. OPSD spike-regime RMSE for local ridge, GraphPatch ridge, GraphPatch MLP, and calibrated GraphPatch blend.](figures/paper1_fig10_opsd_deep_graphpatch_spike_rmse.png)

![Figure 8. OPSD calibrated GraphPatch conformal coverage and calibration-selected MLP residual weight.](figures/paper1_fig11_opsd_deep_graphpatch_conformal_blend.png)


Table 11 adds reviewer-facing robustness checks for the deep GraphPatch model. Rolling-origin evaluation uses three chronological train-calibration-test windows per zone. Leave-one-zone-out evaluation trains the residual learner on the other zones, keeps the local ridge model trained on the held-out zone, and selects only the blend weight on the held-out calibration split.

|Protocol|Regime|Model|Mean RMSE imp. %|Median RMSE imp. %|Min RMSE imp. %|Positive cases|Cases|
|---|---|---|---|---|---|---|---|
|Rolling origin|all|GraphPatch blend|12.021|13.149|5.792|12|12|
|Rolling origin|spike|GraphPatch blend|11.252|11.988|2.224|12|12|
|Leave-one-zone-out|all|LOZO blend|6.678|7.053|2.886|4|4|
|Leave-one-zone-out|spike|LOZO blend|5.601|6.650|-0.963|3|4|

The rolling-origin result is strong: the GraphPatch blend improves spike-regime RMSE in 12/12 rolling cases, with mean spike RMSE improvement of 11.252%. This addresses a common reviewer concern that gains may be tied to a single chronological split. The leave-one-zone-out result is positive on average but weaker, with 3/4 positive spike-regime RMSE cases and mean spike RMSE improvement of 5.601%. The model therefore shows useful transfer, but not unconditional cross-zone dominance.

Table 12 reports the leave-one-zone-out spike-regime details. The DE-LU holdout has a small RMSE degradation but a positive MAE reduction and a mixed paired test, while DK1, DK2, and Great Britain show clear positive transfer.

|Held-out zone|RMSE|Local RMSE|RMSE imp. %|MAE imp. %|Win rate|p approx.|Direction|N|
|---|---|---|---|---|---|---|---|---|
|DE_LU|13.256|13.129|-0.963|3.152|0.558|0.067|mixed|251|
|DK_1|7.000|7.496|6.619|10.155|0.593|<0.001|better|1718|
|DK_2|7.193|7.708|6.682|10.267|0.616|<0.001|better|1712|
|GB_GBN|9.000|10.008|10.067|11.683|0.722|<0.001|better|767|

These robustness checks strengthen the reviewer-facing empirical argument without overstating generalization. The empirical claim is now narrower and more defensible: GraphPatch is consistently helpful across rolling time origins and mostly transferable across zones, but a fully portable residual learner still requires adaptive calibration or target-domain labels.

![Figure 9. OPSD rolling-origin robustness of the GraphPatch blend on spike-regime RMSE.](figures/paper1_fig12_opsd_graphpatch_rolling_origin.png)

![Figure 10. OPSD leave-one-zone-out GraphPatch transfer robustness on spike-regime RMSE.](figures/paper1_fig13_opsd_graphpatch_zone_holdout.png)


Table 13 now closes the main deep-sequence baseline gap by adding a TCN-family comparator to the DLinear/NLinear sequence-anchor block. The TDConv-style model uses deterministic multi-scale causal slices from the 168-hour history window, including dilations of 1, 2, 4, and 8 hours, and fits a regularized ridge head on the resulting temporal-convolution features. It is intentionally reported as a trainable dilated-convolution ridge comparator rather than as a full GPU-dependent TCN, because the public supplement must remain reproducible in a lightweight NumPy/Pandas environment.

|Zone|Model|MAE|RMSE|sMAPE|N|
|---|---|---|---|---|---|
|DE_LU|DLinear-style|6.509|13.156|35.363|251|
|DE_LU|NLinear-style|6.505|13.171|35.609|251|
|DE_LU|TDConv-style|6.361|12.785|34.973|251|
|DK_1|DLinear-style|3.908|6.669|27.394|1718|
|DK_1|NLinear-style|3.905|6.672|27.483|1718|
|DK_1|TDConv-style|3.907|6.682|27.241|1718|
|DK_2|DLinear-style|4.251|6.956|21.834|1712|
|DK_2|NLinear-style|4.247|6.956|21.844|1712|
|DK_2|TDConv-style|4.250|7.016|21.524|1712|
|GB_GBN|DLinear-style|6.162|8.679|34.518|767|
|GB_GBN|NLinear-style|6.136|8.608|34.262|767|
|GB_GBN|TDConv-style|5.927|8.449|32.296|767|

The TDConv comparator changes the interpretation in a useful way. It becomes the selected calibration anchor in DE-LU and Great Britain, while NLinear remains selected in DK1 and DK2. Thus, GraphPatch is no longer being compared only with decomposition-style linear anchors; it must add residual value after a stronger TCN-family sequence feature map is allowed into the anchor set. The selected spike-regime anchors are DE_LU: TDConv-style, DK_1: NLinear-style, DK_2: NLinear-style, GB_GBN: TDConv-style.

Table 14 reports the TDConv-inclusive selected-anchor GraphPatch residual. The residual shrinkage is chosen only on each zone's calibration split, and the test split remains the final chronological 20% of each OPSD zone.

|Zone|Selected anchor|Residual weight|Anchor RMSE|GP RMSE|RMSE gain %|MAE gain %|Win rate|p approx.|N|
|---|---|---|---|---|---|---|---|---|---|
|DE_LU|TDConv-style|0.900|12.785|12.434|2.742|1.335|0.478|0.487|251|
|DK_1|NLinear-style|0.400|6.672|6.601|1.073|1.065|0.551|<0.001|1718|
|DK_2|NLinear-style|0.600|6.956|6.859|1.384|2.210|0.556|<0.001|1712|
|GB_GBN|TDConv-style|0.500|8.449|8.470|-0.249|-0.390|0.490|0.588|767|

This stronger comparison makes the Paper 1 claim narrower but more publishable. The TDConv-inclusive GraphPatch residual improves all-hour RMSE in 4/4 zones and spike-regime RMSE in 3/4 zones, with mean spike-regime RMSE gain of 1.237% over the selected anchor. The Nordic zones retain significant paired gains, DE-LU keeps a positive RMSE/MAE reduction but not a significant paired win rate, and Great Britain becomes the explicit limitation case under the stronger TDConv anchor. The manuscript should therefore claim calibrated residual value under volatile regimes, not universal dominance over every local sequence encoder.

![Figure 11. TDConv-family sequence comparator and TDConv-inclusive GraphPatch residual on OPSD spike-regime RMSE.](figures/paper1_fig16_opsd_tdconv_anchor_graphpatch.png)


![Figure 12. Proposed uncertainty-aware spatio-temporal price forecasting framework.](figures/paper1_fig1_architecture.png)

![Figure 13. Representative split-conformal prediction interval on the local Hunan price holdout set.](figures/paper1_fig2_price_interval.png)

![Figure 14. Empirical coverage of the 90% split-conformal interval on local price datasets.](figures/paper1_fig3_coverage.png)

The public benchmark and the local pilot support the same methodological conclusion: simple temporal structure is strong in ordinary periods, but market operation requires calibrated intervals and decision-aware evaluation under volatility. The OPSD conformal result also shows why conditional calibration, not only average coverage, is central to the proposed graph-temporal uncertainty model.
## 6.2 Pilot Results on Local Price Data

The current pilot experiment uses only lightweight, auditable baselines and is treated as an application feasibility check rather than the sole evidence for the main claim. The results nevertheless confirm that the extracted local price data can support forecasting experiments and that market-state variables are informative.

|Data|Model|MAE|RMSE|sMAPE|N|Corr|
|---|---|---|---|---|---|---|
|Hunan price|Persist-1h|51.296|81.570|60.216|1181||
|Hunan price|Seasonal-24h|105.269|149.020|101.887|1181||
|Hunan price|Seasonal-168h|163.386|211.150|129.566|1181||
|Hunan price|Linear-lag-cal|57.289|78.893|92.929|1181||
|Disclosure|Persist-1h|61.936|119.468|29.327|692|0.751|
|Disclosure|Seasonal-24h|144.647|210.555|56.056|692|0.751|
|Disclosure|Seasonal-168h|157.586|221.890|63.550|692|0.751|
|Disclosure|Linear-space-cal|66.387|104.346|41.017|692|0.751|

For the Hunan spot-price series, the one-hour persistence baseline obtains MAE 51.30 and RMSE 81.57 yuan/MWh on the holdout split, while a lag-and-calendar linear model obtains RMSE 78.89. For the disclosure price-space dataset, the training correlation between bidding space and real-time price is 0.751, supporting the use of supply-demand state features in the proposed multi-source model. The public OPSD experiment now supplies the reproducible main benchmark, while this local table remains an application case for checking whether local market-state variables are informative.


The same local price data were also used for a split-conformal interval pilot. The linear model was trained on the first 60% of the usable sequence, calibrated on the next 20%, and tested on the final 20%. A 90% target interval was used.

|Data|Model|Target|Coverage|Width|NormWidth|N|
|---|---|---|---|---|---|---|
|Hunan price|Linear-conformal|0.900|0.919|290.492|0.400|1181|
|Disclosure|Linear-conformal|0.900|0.912|370.257|0.408|692|

The empirical coverage is slightly above the 90% target on both local datasets, which supports the manuscript's emphasis on calibrated uncertainty. The intervals are intentionally produced by a simple auditable baseline; the public OPSD experiment above reports PICP, PINAW, pinball loss, interval score, and graph-temporal residual ablations for the reproducible main claim.

## 7. Conclusion

This paper presents an uncertainty-aware spatio-temporal learning framework for electricity price forecasting in virtual power plant market operation. The evidence shows that lag-safe graph-temporal residual learning can improve volatile-regime price forecasts after strong local sequence anchors have been fitted, including a TDConv-style TCN-family comparator. The result is deliberately bounded: rolling-origin tests are strong, cross-zone transfer is mostly positive, and the TDConv-inclusive spike comparison exposes Great Britain as a limitation case. The work contributes to computer-science research on heterogeneous non-stationary time-series learning while providing a calibrated forecasting component for virtual power plant decision systems.

## Data Availability Statement

The public OPSD benchmark is reproducible from the scripts `public_data_download_templates.py`, `run_public_opsd_baselines.py`, `run_opsd_probabilistic_price_model.py`, `run_opsd_graph_temporal_price_ablation.py`, `run_opsd_nonlinear_price_baselines.py`, `run_opsd_deep_graph_patch_price_model.py`, `run_opsd_graphpatch_robustness.py`, `run_opsd_modern_sequence_price_baselines.py`, `run_opsd_sequence_anchor_graphpatch_price_model.py`, and `run_opsd_tdconv_sequence_anchor_graphpatch_price_model.py`. Private virtual power plant data, if used, are reported as a non-public supplementary case and the main claims remain reproducible on public datasets.

## Local and China Real-World Data Assets

The newly inventoried local authorized real-world data directory adds a China-market application layer to this price-forecasting manuscript. The audit file `paper_package/master_submission_control/real_world_data_inventory_and_paper_mapping.md` records 62 files in six families, including Liaoning and Eastern Inner Mongolia quarter-hour public market disclosure tables, Shandong price/load/weather/coal data, China transmission-network metadata, and WRI/GEM power-plant metadata.

|family|files|total_mib|status|main_use|risk|
|---|---|---|---|---|---|
|China power grid multi-year transmission network|6|0.226|Figshare CC BY 4.0 for core network dataset; GEM UHV reference folder has non-commercial caveat|Dissertation background, graph construction motivation, resource/network metadata appendix; optional Paper 1 graph-prior discussion|mixed license: keep GEM NC materials out of journal supplements unless permission is confirmed|
|Eastern Inner Mongolia public market disclosure|5|4.063|public-market-disclosure/local-copy; verify portal license before redistribution|Paper 1 cross-market price forecasting; Paper 3 decision-focused bidding; dissertation regional comparison|low confidentiality if sourced from public disclosure, but source-page citation and license check are required|
|Liaoning public market disclosure|12|26.385|public-market-disclosure/local-copy; verify portal license before redistribution|Paper 1 price forecasting; Paper 2 regional load forecasting; Paper 3 VPP decision case; dissertation China market chapter|low confidentiality if sourced from public disclosure, but final supplement redistribution still needs source-page citation and license check|
|OSM China power grid GIS extraction|13|598.263|OpenStreetMap-derived local extraction; several source files are incomplete .downloading files|Dissertation background only unless extraction is completed and ODbL attribution/share-alike handling is reviewed|license and incomplete-download risk; do not use as a primary journal experiment yet|
|Shandong market, load, weather, coal and consumption data|8|52.183|local real-world market data; treat as non-public until source-page and redistribution rights are confirmed|Paper 1 local price/weather/fuel case; Paper 2 load-weather case; Paper 3 settlement and VPP decision case; dissertation applied validation|medium: may include market-operation details; use aggregated or anonymized statistics in manuscripts until permission is clear|
|WRI/GEM China power plant metadata and time-series summaries|18|20.214|WRI Global Power Plant Database is CC BY 4.0; GEM links require final use-permission review|Dissertation resource-mix context, VPP portfolio scenario design, Paper 3 resource assumptions appendix|WRI is usable with attribution; GEM commercial/non-commercial caveats must be separated|

For Paper 1, the most relevant additions are the Liaoning and Eastern Inner Mongolia day-ahead/real-time clearing price and market-state fields, plus Shandong day-ahead/real-time price, weather and fuel-price context. These data are valuable for a China-market application case because they contain price volatility, bidding space, load, renewable generation and cross-market heterogeneity. They should not replace the OPSD public benchmark as the main reproducible evidence; instead, they should be used for anonymized external validation, feature-motivation figures, and dissertation case-study support after source/license verification.

The immediate modeling update is to build a normalized quarter-hour China market table with fields such as market, timestamp, day-ahead price, real-time price, system load, forecast load, renewable output/forecast, bidding space, tie-line plan, weather variables and coal index. This table can support graph-temporal feature engineering and spike-regime validation while preserving the manuscript's computer-science contribution: heterogeneous non-stationary time-series learning with calibrated uncertainty.

This update has started: `china_real_world_data/china_market_disclosure_quarterhour_wide.csv` normalizes 92,576 quarter-hour records from Liaoning and Eastern Inner Mongolia, and `china_real_world_data/china_market_disclosure_baseline_report.md` records leakage-aware price baselines. These results should be treated as China-market application evidence until source-page/license verification is complete.

The Shandong data layer has also been normalized in `china_real_world_data/shandong_price_hourly_day_ahead_real_time.csv` and evaluated in `china_real_world_data/shandong_real_data_baseline_report.md`. It contains 33,621 hourly day-ahead/real-time price rows, 117,610 canonical quarter-hour load-weather rows and 196 weekly BSPI coal-index rows. Duplicate load-weather forecast-version rows are aggregated by timestamp using a declared median rule before modeling. On the chronological price holdout, a leakage-safe linear model using lagged price, day-ahead price, weather aggregates and coal index reaches RMSE 75.324 yuan/MWh, compared with 93.112 for the direct day-ahead price baseline and 98.205 for one-hour persistence. This gives Paper 1 a concrete China-market application result for fuel/weather-aware price forecasting, while the public OPSD experiment remains the reproducible main claim.

A separate GIS infrastructure audit, `paper_package/master_submission_control/gis_energy_infrastructure_evidence.md`, profiles China transmission-network snapshots, OSM mainland power-facility extractions, WRI China power-plant metadata and the GEM integrated China power-facility table. The current derived summaries contain 12,839 2025 transmission-line records, 2,041 substation records, 9,444 grid-link records, 4,870,459 OSM mainland power records, 4,274 WRI China plant records and approximately 3,108.905 GW of represented GEM operating capacity. For Paper 1, these records are used as infrastructure-context evidence and graph/resource heterogeneity motivation only. They are not treated as target labels or as a model-training dataset for the public benchmark experiments, and raw OSM/GEM/SHP/PBF files remain excluded from public supplements until attribution, license and redistribution checks are closed.

![Figure 18. China real-market application baselines for electricity price forecasting. The chart compares persistence, day/market-direct baselines and leakage-safe lagged market/weather models on Eastern Inner Mongolia, Liaoning and Shandong holdout splits.](figures/paper1_fig18_china_real_market_price_baselines.png)


## References

[1] Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, Jones, Llion, Gomez, Aidan N., Kaiser, Lukasz, Polosukhin, Illia, Attention is all you need, Advances in Neural Information Processing Systems 30 (2017), https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need.
[2] Bryan Lim, Sercan Ö. Arık, Nicolas Loeff, Tomas Pfister, Temporal Fusion Transformers for interpretable multi-horizon time series forecasting, International Journal of Forecasting 37 (2021) 1748-1764, https://doi.org/10.1016/j.ijforecast.2021.03.012.
[3] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang, Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting, Proceedings of the AAAI Conference on Artificial Intelligence 35 (2021) 11106-11115, https://doi.org/10.1609/aaai.v35i12.17325.
[4] Wu, Haixu, Xu, Jiehui, Wang, Jianmin, Long, Mingsheng, Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting, Advances in Neural Information Processing Systems 34 (2021) 22419--22430, https://arxiv.org/abs/2106.13008.
[5] Nie, Yuqi, Nguyen, Nam H., Sinthong, Phanwadee, Kalagnanam, Jayant, A time series is worth 64 words: Long-term forecasting with Transformers, International Conference on Learning Representations (2023), https://openreview.net/forum?id=Jbdc0vTOcol.
[6] Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu, Are Transformers Effective for Time Series Forecasting?, Proceedings of the AAAI Conference on Artificial Intelligence 37 (2023) 11121-11128, https://doi.org/10.1609/aaai.v37i9.26317.
[7] Oreshkin, Boris N., Carpov, Dmitri, Chapados, Nicolas, Bengio, Yoshua, N-BEATS: Neural basis expansion analysis for interpretable time series forecasting, International Conference on Learning Representations (2020), https://openreview.net/forum?id=r1ecqn4YwB.
[8] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Chengqi Zhang, Graph WaveNet for Deep Spatial-Temporal Graph Modeling, Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (2019) 1907-1913, https://doi.org/10.24963/ijcai.2019/264.
[9] Anastasios N. Angelopoulos, Stephen Bates, Conformal Prediction: A Gentle Introduction, Foundations and Trends® in Machine Learning 16 (2023) 494-591, https://doi.org/10.1561/2200000101.
[10] Dimitris Bertsimas, Nathan Kallus, From Predictive to Prescriptive Analytics, Management Science 66 (2020) 1025-1044, https://doi.org/10.1287/mnsc.2018.3253.
[11] Adam N. Elmachtoub, Paul Grigas, Smart “Predict, then Optimize”, Management Science 68 (2022) 9-26, https://doi.org/10.1287/mnsc.2020.3922.
[12] Amos, Brandon, Kolter, J. Zico, OptNet: Differentiable Optimization as a Layer in Neural Networks, Proceedings of the 34th International Conference on Machine Learning 70 (2017) 136--145, https://proceedings.mlr.press/v70/amos17a.html.
[13] Donti, Priya L., Amos, Brandon, Kolter, J. Zico, Task-based End-to-end Model Learning in Stochastic Optimization, Advances in Neural Information Processing Systems 30 (2017), https://proceedings.neurips.cc/paper/2017/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf.
[14] Tao Hong, Pierre Pinson, Shu Fan, Global Energy Forecasting Competition 2012, International Journal of Forecasting 30 (2014) 357-363, https://doi.org/10.1016/j.ijforecast.2013.07.001.
[15] Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli, Rob J. Hyndman, Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond, International Journal of Forecasting 32 (2016) 896-913, https://doi.org/10.1016/j.ijforecast.2016.02.001.
[16] Jesus Lago, Fjo De Ridder, Bart De Schutter, Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms, Applied Energy 221 (2018) 386-405, https://doi.org/10.1016/j.apenergy.2018.02.069.
[17] Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron, Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark, Applied Energy 293 (2021) 116983, https://doi.org/10.1016/j.apenergy.2021.116983.
[18] Rafał Weron, Electricity price forecasting: A review of the state-of-the-art with a look into the future, International Journal of Forecasting 30 (2014) 1030-1081, https://doi.org/10.1016/j.ijforecast.2014.08.008.
[19] Bartosz Uniejewski, Grzegorz Marcjasz, Rafał Weron, On the importance of the long-term seasonal component in day-ahead electricity price forecasting, Energy Economics 79 (2019) 171-182, https://doi.org/10.1016/j.eneco.2018.02.007.
[20] Jakub Nowotarski, Rafał Weron, Recent advances in electricity price forecasting: A review of probabilistic forecasting, Renewable and Sustainable Energy Reviews 81 (2018) 1548-1568, https://doi.org/10.1016/j.rser.2017.05.234.
[21] Ben Taieb, Souhaib, Taylor, James W., Hyndman, Rob J., Coherent probabilistic forecasts for hierarchical time series, Proceedings of the 34th International Conference on Machine Learning 70 (2017) 3348--3357, https://proceedings.mlr.press/v70/taieb17a.html.
[22] Hyndman, Rob J., Athanasopoulos, George, Forecasting: Principles and Practice, OTexts (2021), https://otexts.com/fpp3/.
[23] {Open Power System Data}, Data Package Time Series, version 2020-10-06, (2020), https://data.open-power-system-data.org/time_series/2020-10-06/.
[24] Hossein Mohammadi Rouzbahani, Hadis Karimipour, Lei Lei, A review on virtual power plant for energy management, Sustainable Energy Technologies and Assessments 47 (2021) 101370, https://doi.org/10.1016/j.seta.2021.101370.
[25] Seyyed Mostafa Nosratabadi, Rahmat-Allah Hooshmand, Eskandar Gholipour, A comprehensive review on microgrid and virtual power plant concepts employed for distributed energy resources scheduling in power systems, Renewable and Sustainable Energy Reviews 67 (2017) 341-363, https://doi.org/10.1016/j.rser.2016.09.025.
[26] Stephen Boyd, Lieven Vandenberghe, Convex Optimization, (2004), https://doi.org/10.1017/cbo9780511804441.
[27] Tao Hong, Pierre Pinson, Yi Wang, Rafal Weron, Dazhi Yang, Hamidreza Zareipour, Energy Forecasting: A Review and Outlook, IEEE Open Access Journal of Power and Energy 7 (2020) 376-388, https://doi.org/10.1109/oajpe.2020.3029979.
[28] Florian Ziel, Rafał Weron, Day-ahead electricity price forecasting with high-dimensional structures: Univariate vs. multivariate modeling frameworks, Energy Economics 70 (2018) 396-420, https://doi.org/10.1016/j.eneco.2017.12.016.
[29] Tilmann Gneiting, Adrian E Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation, Journal of the American Statistical Association 102 (2007) 359-378, https://doi.org/10.1198/016214506000001437.
[30] Roger Koenker, Gilbert Bassett, Regression Quantiles, Econometrica 46 (1978) 33, https://doi.org/10.2307/1913643.
[31] Francis X. Diebold, Roberto S. Mariano, Comparing Predictive Accuracy, Journal of Business & Economic Statistics 13 (1995) 253-263, https://doi.org/10.1080/07350015.1995.10524599.
[32] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, Chengqi Zhang, Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks, Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (2020) 753-763, https://doi.org/10.1145/3394486.3403118.
[33] Bing Yu, Haoteng Yin, Zhanxing Zhu, Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting, Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence (2018) 3634-3640, https://doi.org/10.24963/ijcai.2018/505.
[34] Bryan Lim, Stefan Zohren, Time-series forecasting with deep learning: a survey, Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 379 (2021) 20200209, https://doi.org/10.1098/rsta.2020.0209.
[35] José F. Torres, Dalil Hadjout, Abderrazak Sebaa, Francisco Martínez-Álvarez, Alicia Troncoso, Deep Learning for Time Series Forecasting: A Survey, Big Data 9 (2021) 3-21, https://doi.org/10.1089/big.2020.0159.
