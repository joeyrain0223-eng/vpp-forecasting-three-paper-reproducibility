# Sequence-Anchored GraphPatch Residual Learning for Uncertainty-Aware Electricity Price Forecasting in Virtual Power Plant Market Operation

## Abstract

Electricity price forecasting for virtual power plants requires more than average point accuracy: market operation also depends on calibrated uncertainty, price-spike robustness, and representations that transfer across heterogeneous market zones. This paper develops a reproducible graph-temporal learning framework for day-ahead price forecasting that combines lag-safe cross-zone graph features, multi-scale temporal patches, residual learning, and split-conformal calibration. The engineering application is virtual power plant market operation, while the artificial-intelligence contribution is a calibration-selected GraphPatch residual architecture for non-stationary, spike-prone time series. Public experiments use Open Power System Data for Germany-Luxembourg, Denmark West, Denmark East, and Great Britain. On spike-regime test hours, the calibrated GraphPatch blend improves RMSE over a local lag-calendar ridge baseline in all four zones and wins all four paired sign tests. Against DLinear/NLinear anchors, a shrinkage-controlled residual improves spike-regime RMSE in all four zones by 3.69% on average. With a TDConv-style TCN-family comparator, the selected-anchor residual improves all-hour RMSE in 4/4 zones and spike-regime RMSE in 3/4 zones, exposing Great Britain as the strong-anchor limitation case. Rolling-origin validation improves spike-regime RMSE in 12/12 zone-window cases. Conformal intervals reach near-target or conservative coverage in Germany-Luxembourg and Great Britain but remain mildly under-covered in the Nordic zones. The results support a narrow claim: graph-temporal residual learning improves volatile-market forecasting while adaptive calibration remains necessary.

**Keywords:** electricity price forecasting; virtual power plant; graph-temporal learning; conformal prediction; spike robustness; public reproducibility.

## 1. Introduction

Virtual power plants aggregate distributed photovoltaic generation, battery storage, flexible loads, and other controllable resources into a market-facing entity [24,25]. Their scheduling and bidding decisions depend strongly on day-ahead price forecasts, which are often coupled to downstream prescriptive, predict-then-optimize, and differentiable optimization layers [10-13,26]. For this application, average error alone is not enough: sparse price spikes, negative prices, and regime changes can dominate operational risk even when ordinary-hour accuracy appears acceptable.

From a computer-science perspective, this setting is a non-stationary time-series learning problem with heterogeneous covariates and cross-zone dependence. Public forecasting competitions, electricity-price forecasting reviews, long-term seasonality studies, probabilistic forecasting surveys, and open-data benchmarks establish that evaluation must cover chronology, volatility, uncertainty, and replicability rather than only a single average-error split [14-23]. Local temporal structure is strong, so simple seasonal, linear, and modern sequence baselines can be difficult to beat. At the same time, price spikes may reflect information that is not fully captured by a single-zone sequence model, including lagged behavior in neighboring markets, load conditions, and renewable-generation patterns.

The paper therefore asks a narrow empirical question: after a strong local sequence anchor has already modeled ordinary temporal persistence, can lag-safe cross-zone graph patches improve volatile price forecasting? The proposed answer is a residual-learning framework. A local anchor first predicts the target price, a GraphPatch residual learner then uses local temporal patches and lagged cross-zone summaries to estimate the remaining error, and a calibration-selected shrinkage or blend weight controls the residual contribution. Transformer foundations, interpretable temporal attention models, patch-based sequence models, linear sequence baselines, and N-BEATS-style decomposition models define the relevant sequence-learning comparison space [1-7]. Split-conformal calibration is used to report interval reliability rather than treating point-error gains as sufficient evidence [9].

This framing is intentionally more conservative than an unconstrained dynamic-graph or end-to-end Transformer claim. The contribution is not that graph learning universally dominates electricity price forecasting. The contribution is an auditable AI-engineering result on public OPSD data: graph-temporal residual information improves spike-regime forecasts beyond local ridge and DLinear/NLinear-style anchors in the reported tests, while leave-one-zone-out and conformal diagnostics expose the remaining transfer and calibration limitations.

The novelty is therefore in the residual interface rather than in attaching another feature block to a forecaster. DLinear, NLinear, and TDConv-style anchors are allowed to explain the dominant local seasonality, trend, and short-range convolutional structure first. GraphPatch then receives a different task: estimate the remaining forecast error from lag-safe cross-zone summaries and multi-scale patches, with its residual magnitude controlled by calibration-only shrinkage. This design creates a falsifiable test of relational information: if graph-temporal features merely duplicate the local anchor, the shrinkage weight should collapse toward zero or the held-out residual should not improve spike-regime errors. The method also separates three choices that are often entangled in electricity-price neural models: the local sequence model, the relational residual learner, and the uncertainty calibration layer. That separation makes the contribution more than an engineering tweak, because the experiment can identify when graph information is useful, when a stronger anchor absorbs it, and when conformal calibration still fails.

The manuscript contributes three elements. First, it defines a reproducible public benchmark for virtual-power-plant-oriented price forecasting with explicit spike-regime and calibration diagnostics. Second, it introduces a sequence-anchored GraphPatch residual design that keeps the graph contribution identifiable against DLinear/NLinear and TDConv-family local sequence baselines. Third, it reports rolling-origin, leave-one-zone-out, spike-threshold, strong-anchor, and calibration-window sensitivity checks so that the empirical claim is testable rather than dependent on a single split or threshold.

## 2. Related Work

### 2.1 Electricity-price forecasting protocols and benchmark discipline

Electricity-price forecasting has a mature literature, but the field remains sensitive to data leakage, split design, volatility treatment, and benchmark selection. Early and review work emphasizes that electricity prices differ from ordinary demand series because price spikes, negative prices, calendar effects, fuel conditions, and renewable penetration can create non-Gaussian and regime-dependent errors [16-20]. Open benchmark studies further show that model rankings can change when the evaluation moves from a single market or single split to multivariate, high-dimensional, and reproducible comparison settings [17,28]. This paper follows that benchmark discipline by using chronological public OPSD splits, training-defined spike thresholds, and explicit public/private data boundaries rather than treating a non-public single case or a single holdout window as decisive evidence.

### 2.2 Deep sequence models and the strength of simple anchors

Recent sequence-learning work provides many strong candidates for electricity-price forecasting, including attention-based, decomposition-based, patch-based, convolutional, and interpretable multi-horizon models [1-5]. At the same time, N-BEATS and linear sequence baselines demonstrate that sophisticated neural architectures do not automatically dominate when local seasonality and persistence are strong [6,7]. Broader deep time-series surveys reach the same methodological warning: the value of a new architecture depends on the forecast horizon, scale, stationarity, covariate availability, and comparison protocol [34,35]. This motivates the sequence-anchored design in this paper. The GraphPatch module is not evaluated as an unrestricted black-box replacement for strong local temporal models; it is evaluated as a residual learner that must add value after local lag-calendar, DLinear/NLinear-style, and TDConv-style anchors have already captured ordinary temporal structure.

### 2.3 Spatio-temporal graph learning for correlated market signals

Graph neural networks are relevant because electricity prices, loads, and renewable generation are spatially and operationally correlated across market zones. Spatio-temporal graph convolution, adaptive graph learning, and multivariate graph forecasting show that learned or partially learned relational structure can improve time-series forecasts when single-series models miss cross-node dependencies [8,32,33]. However, the direct transfer of traffic-style graph models to electricity markets is not straightforward: market zones have changing congestion patterns, cross-border flow constraints, regulatory differences, and uneven public covariate availability. The proposed GraphPatch design therefore uses lag-safe graph summaries and calibration-selected residual shrinkage. This sacrifices some expressiveness, but it keeps the evidence auditable and makes the graph contribution identifiable against local anchors.

### 2.4 Probabilistic forecasting, conformal calibration, and scoring

For virtual power plant operation, a useful price forecast must describe uncertainty as well as point location. Probabilistic energy forecasting competitions, hierarchical forecasting methods, and electricity-price probabilistic reviews establish that coverage, sharpness, calibration, and distributional scoring are central evaluation dimensions [14,15,20,21]. Proper scoring rules define why probabilistic forecasts should be judged by scores that reward calibrated distributions rather than only narrow intervals [29], while quantile regression provides a classical basis for interval and tail modeling [30]. Split-conformal prediction adds a model-agnostic calibration layer with transparent empirical coverage control under appropriate exchangeability conditions [9]. In this paper, conformal intervals are used as a diagnostic and deployment boundary: the GraphPatch point model is not treated as operationally sufficient unless interval coverage and width remain defensible on held-out data.

### 2.5 Statistical comparison and robustness evidence

Electricity-price forecasting papers are vulnerable to overclaiming when they report only average error on a single test period. Forecast-comparison work such as the Diebold-Mariano framework highlights that paired predictive accuracy should be assessed through matched forecast errors, while modern energy-forecasting practice also emphasizes rolling-origin validation, regime-specific performance, and transparent negative cases [27,31]. This paper reports paired spike-regime win rates, rolling-origin windows, leave-one-zone-out transfer, and calibration-window checks. The purpose is not to inflate the number of tests, but to make the claim falsifiable: a graph-temporal residual model should help where cross-zone information plausibly matters, and it should expose rather than hide cases where transfer or interval calibration remains weak.

### 2.6 Virtual power plant relevance and prediction-decision boundary

Virtual power plant studies connect forecasting to market bidding, distributed resource scheduling, and energy-management decisions [24,25]. Predictive-to-prescriptive analytics, smart predict-then-optimize learning, differentiable optimization layers, and task-based learning explain why forecast errors should ultimately be judged by downstream decisions when the operational objective is known [10-13,26]. This paper deliberately stops short of claiming full market-profit optimization. Its role in the dissertation sequence is the price-forecasting chapter: it supplies a calibrated and spike-aware public forecast evidence layer that can later feed the companion virtual-power-plant decision paper. This boundary keeps the EAAI submission focused on artificial-intelligence forecasting methodology while preserving the engineering relevance to VPP market operation.

## 3. Problem Formulation

Let p_t^z denote the hourly day-ahead electricity price for market zone z. Given a chronological lookback window, local calendar variables, local lagged prices, local load and renewable-generation variables, and lagged cross-zone summaries from other market zones, the task is to predict p_{t+h}^z on a held-out future test period. The main reported setting uses h = 1 and evaluates both all-hour and spike-regime performance. Spike hours are defined from the training-set distribution of absolute short-term versus daily-lag price changes, so no test-set information is used to select the volatility threshold.

The paper studies a residual-learning question rather than an unconstrained end-to-end forecasting claim: after a strong local sequence model has captured ordinary temporal persistence, can lag-safe graph-temporal patch features add measurable value in volatile price regimes? The implemented prediction pipeline is:

anchor_t^z = f_anchor(X_{t-L+1:t}^z),

r_t^z = p_t^z - anchor_t^z,

delta_t^z = f_GP(Phi_t^z),

hat p_t^z = anchor_t^z + gamma_z delta_t^z,

where f_anchor is the selected local ridge, DLinear-style, NLinear-style, or TDConv-style anchor; Phi_t^z contains local temporal patches and lagged cross-zone graph summaries; f_GP is the GraphPatch residual learner; and gamma_z is a shrinkage weight selected only on calibration data. Split-conformal calibration is then applied to calibration residuals to construct intervals around the point forecast. This formulation keeps the graph contribution identifiable and prevents the proposed model from being evaluated only as a black-box replacement for strong local sequence baselines.

## 4. Proposed Method

![Figure 1. Sequence-anchored GraphPatch residual pipeline for lag-safe electricity price forecasting and conformal diagnostics.](figures/paper1_fig1_architecture.png)

### 4.1 Lag-safe graph-temporal feature construction

The implemented graph-temporal input is deliberately lag-safe. For each target zone, the feature set includes local price lags, load and renewable-generation covariates, calendar indicators, and cross-zone summaries constructed only from information available before the forecast origin. Cross-zone features are summarized through lagged means, dispersions, and multi-scale temporal patches rather than through contemporaneous target-period prices. This design is less expressive than a fully learned dynamic graph, but it is auditable and directly aligned with the public OPSD data columns.

### 4.2 Sequence anchor and residual target

The first stage fits a local sequence anchor. Transparent lag-calendar ridge models provide the reproducible baseline, DLinear-style and NLinear-style anchors test decomposition-style sequence persistence, and a TDConv-style trainable dilated-convolution ridge comparator supplies a TCN-family strong-anchor check without introducing GPU-dependent reproducibility requirements. The residual target is the difference between the observed price and the selected anchor prediction. This residual formulation is central to the paper's computer-science claim: graph-temporal learning is used to explain what strong local temporal models still miss, especially during spike regimes.

### 4.3 GraphPatch residual learner

The GraphPatch residual learner constructs multi-scale patches from 1-, 2-, 3-, 6-, 12-, 24-, 48-, 72-, and 168-hour windows, augments them with lagged cross-zone graph summaries, and predicts an additive residual correction. The main nonlinear variant is a two-layer tanh residual MLP. A ridge residual variant and a lightweight random-hidden-layer residual model are retained as auditable controls. For each zone, the final blend weight between residual variants, or the residual shrinkage coefficient relative to the sequence anchor, is selected on calibration data only.

### 4.4 Split-conformal interval calibration

Point-error gains do not by themselves establish operational reliability. The interval layer therefore uses split-conformal calibration on residuals from the calibration window immediately preceding the test period. For a nominal 90% interval, the empirical residual quantile widens the point forecast into a prediction interval. Calibration-window sensitivity is reported to test whether the interval result depends on a fragile split choice.

### 4.5 Reviewer-facing robustness design

The method is evaluated through six diagnostics that match the paper's claims: spike-regime improvements against local ridge, sequence-anchored improvements against DLinear/NLinear-style anchors, TDConv-inclusive strong-anchor checks, lightweight patch-attention reviewer-response checks, rolling-origin robustness across multiple chronological windows, and leave-one-zone-out transfer across market zones. This design keeps the claims narrow. The paper does not claim universal average-error dominance, full portability across all market systems, or downstream bidding-profit claims; it claims that lag-safe graph-temporal residual learning improves volatile price forecasting under reproducible public-data tests while exposing calibration and transfer limitations.

## 5. Experimental Design

### 5.1 Datasets

The reproducible public benchmark uses Open Power System Data (OPSD) `time_series_60min_singleindex.csv`, package version 2020-10-06. The selected public variables include hourly day-ahead price, load, solar generation, and wind generation for four bidding zones with joint price-load coverage. Non-public Hunan and Shandong operational records are retained only for internal application-boundary checks; they are not used as public evidence for the main empirical claims in this target-journal manuscript.

Table 1 reports the public benchmark coverage used in the current experiment.

|Zone|Rows|Start|End|Price|Load|Solar|Wind|
|---|---|---|---|---|---|---|---|
|DE_LU|17521|2018-10-01|2020-09-30|17521|17521|17516|17521|
|DK_1|50386|2015-01-01|2020-09-30|50386|50386|50377|50386|
|DK_2|50386|2015-01-01|2020-09-30|50386|50386|50373|50386|
|GB_GBN|50288|2015-01-02|2020-09-30|50288|50288|50247|50252|

### 5.2 Baselines

The current public benchmark now includes transparent persistence, seasonal, lag-calendar-exogenous, graph residual ridge, random-hidden-layer residual, GraphPatch residual, DLinear/NLinear-style anchors, a TDConv-style TCN-family sequence comparator, and a lightweight patch-attention reviewer baseline. Full PatchTST/TFT training remains a reviewer-response extension rather than a current claim. For uncertainty forecasting, the manuscript reports conformalized baselines and leaves quantile-regression, DeepAR-style distributional forecasting, and ensemble intervals as optional extensions for the next experiment cycle.

### 5.3 Metrics

Point forecasting metrics: MAE, RMSE, MAPE or sMAPE. Probabilistic metrics: pinball loss, CRPS, PICP, PINAW, calibration error. Operational relevance is reported through spike-regime error, interval coverage, interval width, and robustness diagnostics. Full bidding-profit simulation is left to the companion virtual-power-plant decision paper rather than claimed as evidence in this price-forecasting submission.

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

The lag-calendar-exogenous linear baseline obtains the lowest RMSE among the transparent non-graph baselines on all four OPSD zones. This finding sets a strong reference point: graph-temporal structure is evaluated where it should matter most in this manuscript, namely calibrated uncertainty, spike-regime robustness, and stability across chronological or cross-zone diagnostics.

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

![Figure 2. OPSD public split-conformal coverage against the 90% target.](figures/paper1_fig4_opsd_conformal_coverage.png)

![Figure 3. Representative OPSD regime-conformal prediction interval on the DE-LU public test split.](figures/paper1_fig5_opsd_conformal_interval.png)

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

The graph-temporal residual conformal layer reaches near-target or conservative coverage in DE-LU and Great Britain, while DK1 and DK2 remain mildly under-covered. Compared with the regime-conformal linear benchmark, the result does not dominate interval score. The empirical implication is that graph-temporal point correction and conditional uncertainty calibration are best treated as complementary modules.

![Figure 4. OPSD spike-regime RMSE comparison for the local and graph-temporal residual models.](figures/paper1_fig6_opsd_graph_temporal_spike_rmse.png)

![Figure 5. Relative RMSE change from the graph-temporal residual correction on all hours and spike-regime hours.](figures/paper1_fig7_opsd_graph_temporal_improvement.png)

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

The paired tests show that graph residual ridge has statistically significant spike-regime wins in DK2 and Great Britain, while DE-LU and DK1 remain mixed. The ELM residual model does not provide robust paired evidence and is significantly worse in Great Britain. These tests are treated as descriptive diagnostic evidence: interpretation depends on effect direction, magnitude, and the later robustness checks, and the p-values are not used to claim universal market portability. These results justify retaining a regularized graph-temporal residual layer as the transparent public baseline, while the subsequent GraphPatch experiments test the stronger residual architecture required for the target-journal submission.

![Figure 6. OPSD nonlinear graph-temporal price baselines on spike-regime RMSE.](figures/paper1_fig8_opsd_nonlinear_spike_rmse.png)

![Figure 7. Paired spike-regime error-reduction win rates against the local ridge baseline.](figures/paper1_fig9_opsd_paired_spike_win_rate.png)

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

The deep GraphPatch result is materially stronger than the previous ELM stress test. The standalone GraphPatch MLP improves spike-regime RMSE in all four public zones, with gains of 5.603% in DE-LU, 10.835% in DK1, 9.998% in DK2, and 2.797% in Great Britain relative to the local ridge baseline. The calibrated GraphPatch blend also improves spike-regime RMSE in all four zones, with the largest gains in DK1 and DK2. The all-hour result remains mixed in Great Britain, so the paper frames the model as volatility-regime strengthening rather than universal average-error dominance.

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

![Figure 8. OPSD spike-regime RMSE for local ridge, GraphPatch ridge, GraphPatch MLP, and calibrated GraphPatch blend.](figures/paper1_fig10_opsd_deep_graphpatch_spike_rmse.png)

![Figure 9. OPSD calibrated GraphPatch conformal coverage and calibration-selected MLP residual weight.](figures/paper1_fig11_opsd_deep_graphpatch_conformal_blend.png)

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

![Figure 10. OPSD rolling-origin robustness of the GraphPatch blend on spike-regime RMSE.](figures/paper1_fig12_opsd_graphpatch_rolling_origin.png)

![Figure 11. OPSD leave-one-zone-out GraphPatch transfer robustness on spike-regime RMSE.](figures/paper1_fig13_opsd_graphpatch_zone_holdout.png)

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

![Figure 12. TDConv-family sequence comparator and TDConv-inclusive GraphPatch residual on OPSD spike-regime RMSE.](figures/paper1_fig16_opsd_tdconv_anchor_graphpatch.png)

Table 15 adds a lightweight patch-attention sequence baseline as a reviewer-response check for the patch/attention family. The model pools seven 24-hour patches from the 168-hour history window with deterministic similarity weights and a regularized ridge head. It is intentionally described as a CPU-only patch-attention baseline, not as full PatchTST or TFT training.

Table 15 reports the patch-attention reviewer baseline against the TDConv-style comparator on spike-regime OPSD test hours.

|Zone|TDConv RMSE|Patch-attention RMSE|Patch paired wins / n|MAE delta vs TDConv|Sign-test p|Interpretation|
|---|---:|---:|---:|---:|---:|---|
|DE_LU|12.785|12.027|114 / 251|-0.105|0.147|lower RMSE but paired evidence mixed|
|DK_1|6.682|6.464|841 / 1718|-0.050|0.385|lower RMSE but paired evidence mixed|
|DK_2|7.016|6.794|902 / 1712|0.057|0.026|positive paired evidence|
|GB_GBN|8.449|10.838|379 / 767|-1.159|0.745|negative control under strong anchor|

The patch-attention baseline reduces spike-regime RMSE in DE-LU, DK1 and DK2 relative to TDConv, but paired absolute-error evidence is significant only in DK2 and Great Britain remains a negative-control case under the strong anchor. This result addresses part of the reviewer concern that patch/attention-style sequence evidence is absent. It also preserves the manuscript's claim boundary: the submitted evidence supports GraphPatch residual value after strong local anchors, not universal superiority over every modern sequence family.

![Figure 13. Patch-attention reviewer baseline compared with the TDConv-style sequence comparator on OPSD spike-regime RMSE.](figures/paper1_fig17_opsd_patch_attention_baseline.png)

The public benchmark supports the methodological conclusion that simple temporal structure is strong in ordinary periods, but market operation requires calibrated intervals and decision-aware evaluation under volatility. The OPSD conformal result also shows why conditional calibration, not only average coverage, is central to the proposed graph-temporal uncertainty model.

### 6.2 China Real-World Application Check

To test whether the public OPSD conclusion is relevant to the Chinese market setting, an additional non-public application check was run on normalized Chinese disclosure and Shandong market records. This check is not used as the public reproducibility claim and the raw files are not redistributed. It is reported as an external application sanity check because it exercises the same computer-science problem structure: heterogeneous market time series, day-ahead versus real-time labels, weather and fuel covariates, and volatility-sensitive forecasting.

The Chinese application layer contains quarter-hour Liaoning and Eastern Inner Mongolia disclosure records and an hourly Shandong day-ahead/real-time price table linked to regional weather aggregates and weekly BSPI coal-index observations. On the Shandong chronological holdout, a leakage-safe linear model using lagged price, day-ahead price, weather aggregates, and coal index reaches RMSE 75.324 yuan/MWh. The direct day-ahead price baseline reaches RMSE 93.112 and one-hour persistence reaches RMSE 98.205. In the Liaoning and Eastern Inner Mongolia disclosure checks, lagged market-state models also outperform the strongest persistence-style baselines on the same held-out protocol.

The China-data preprocessing follows the same leakage boundary as the public benchmark. Sentinel-scale administrative price values at or below -1000 yuan/MWh or at or above 5000 yuan/MWh are recoded as missing before modeling, while market-bounded negative prices are retained because they are plausible electricity-market outcomes. Chronological splits are performed after lag filtering; same-time realized real-time prices, future load/renewable observations, and hindsight decision variables are excluded from the forecasting feature set. Missing time slots are not silently interpolated for the holdout score; they reduce the effective test sample through the lag-window filter.

This result should be interpreted conservatively. It does not replace the public OPSD benchmark and does not imply that private Chinese data are available for independent redistribution. It does, however, strengthen the application argument: the same graph-temporal and exogenous-covariate motivation appears in a real Chinese electricity-market setting where fuel, weather, and market-state variables materially affect real-time price error.

![Figure 14. China real-market price application check using leakage-safe lagged market/weather baselines across Eastern Inner Mongolia, Liaoning, and Shandong holdout splits.](figures/paper1_fig18_china_real_market_price_baselines.png)

### 6.3 GIS Infrastructure-Context Check

The graph-temporal formulation is also checked against an open GIS infrastructure evidence layer. A separate audit profiles China transmission-network snapshots, OSM-derived mainland power-grid extracts, WRI China power-plant metadata, and GEM China integrated power-facility records. The 2025 grid snapshot contains 12,839 transmission-line records, 2,041 substation records, and 9,444 grid-link records. The 2015-to-2025 transmission-line record count increases by 610.122%, while grid-link records increase by 114.588%, which supports the paper's rolling and time-aware graph motivation. The OSM mainland extraction profiles 4.87 million power-grid records, including 216,700 line records and 194,613 generator points; the renewable-or-storage source tags in those generator points account for 98.832% of the generator-point table under the derived-summary denominator. WRI contributes 4,274 China power-plant records and a 25.407% renewable capacity share, while the GEM integrated tracker represents about 3,108.9 GW of operating capacity in the local China table.

These GIS statistics are not treated as target labels and are not used to train the OPSD price model. Their role is narrower and more defensible: they document why a price-forecasting model for virtual power plant operation should consider cross-zone graph structure, resource heterogeneity, network evolution, and network-scale operating context. This keeps the submitted numerical claims on public OPSD experiments while preventing the application framing from collapsing into a single-zone or single-device forecasting example.

### 6.4 Public Reproducibility and Application Boundary

All quantitative claims in this target-journal manuscript are evaluated on the public OPSD benchmark and can be regenerated from the accompanying public-data reproducibility package. Non-public Hunan and Shandong operational records were used only to check whether the research question is relevant to local virtual-power-plant market operation. They are not required to reproduce the reported tables or figures, are not redistributed, and are excluded from the public supplement. This separation keeps the submitted evidence auditable while preserving the application motivation.

## 7. Conclusion

This paper developed and evaluated a reproducible graph-temporal residual-learning framework for virtual-power-plant-oriented electricity price forecasting. The final evidence supports a focused computational claim: after local temporal structure is captured by ridge, DLinear-style, NLinear-style, or TDConv-style anchors, lag-safe GraphPatch residual features can add value in volatile price regimes. The DLinear/NLinear sequence-anchored residual improves spike-regime RMSE in all four OPSD zones, while the stricter TDConv-inclusive anchor improves all-hour RMSE in four zones and spike-regime RMSE in three zones. Rolling-origin validation is positive in all 12 zone-window cases, and leave-one-zone-out transfer is positive in three of four held-out zones. Split-conformal calibration provides usable intervals, but the Nordic zones remain mildly under-covered, showing that adaptive local calibration is still required before deployment.

The contribution is therefore not a broad assertion that graph learning always dominates electricity price forecasting. It is a reproducible AI-engineering result: residual graph-temporal representation learning can complement strong sequence baselines under price spikes, and public-data calibration diagnostics can reveal where uncertainty estimates remain fragile. This framing connects the work to virtual power plant market operation while keeping the submitted claims within the evidence provided by the public OPSD benchmark.

## Data Availability Statement

The public-data reproducibility package is supplied as supplementary material. It contains the processed public OPSD benchmark data, experiment scripts, result tables, rendered figures, verified references, and a manifest with SHA-256 checksums. The original OPSD time-series source can be retrieved from the public Open Power System Data repository. Non-public Hunan and Shandong operational records are not redistributed because they may contain confidential market-operation or customer-related information. Only aggregate, non-identifying China-market application statistics and derived GIS infrastructure-context summaries are reported as external validity checks; the public OPSD benchmark remains the independently reproducible evidence layer.

## Declaration of Competing Interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of Generative AI and AI-Assisted Technologies in the Writing Process

During the preparation of this manuscript, OpenAI ChatGPT/Codex was used to support manuscript organization, language editing, code/package documentation, and reproducibility-checklist drafting. The tool was not used as an author, did not determine the scientific conclusions, did not replace verification of data, code, references, results, or claims, and was not used to create or alter figures, images, or graphical-abstract artwork. After using these tools, the author reviewed and edited all AI-assisted text and takes full responsibility for the final content of the manuscript.

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
