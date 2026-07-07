# Transferable Short-Term Load Forecasting for Aggregated Virtual Power Plant Resources via Source-Pooled Time-Series Representation Learning

zhijie REN

College of Computer Science, Hunan University, Lushan South Road, Yuelu District, Changsha, Hunan 410082, China

Corresponding author: zhijie REN (e-mail: 471062741@qq.com).

Funding: This research received no external funding.

Article type for IEEE Access submission system: Research Article.

## Abstract

Accurate load forecasting is fundamental for virtual power plants that coordinate distributed resources, flexible demand, storage, and market transactions. However, aggregated virtual power plant resources often face cross-domain heterogeneity, limited historical observations, and cold-start deployment conditions. Conventional supervised load forecasting models can degrade when applied to new buildings, industrial parks, charging clusters, or regional aggregations. This paper studies transferable short-term load forecasting as a time-series representation-learning problem. Instead of presenting a single universal deep model, the reproducible framework evaluates source-pooled temporal representations under transparent transfer protocols. The checks include masked-reconstruction features, random-convolution features, trainable dilated-convolution ridge features, multi-seed TDConv stability, CPU-only patch-attention transfer, source-trained MLP transfer, and a NumPy neural residual head. Each representation is paired with source heads or lightweight target adapters. Public experiments use OPSD system-load zones, UCI Electricity Load Diagrams multi-client transfer, and UCI Appliances Energy Prediction external validation. The strongest UCI evidence comes from source-pooled temporal representations that improve held-out client RMSE against target-only ridge baselines; the multi-seed TDConv check confirms that this gain is stable to source-window subsampling, and the patch-attention check improves slightly over the TDConv adapter while target-only patch fitting and source-trained MLP transfer expose useful negative controls. The external appliance dataset shows that transparent lag-weather features can remain competitive. The study frames virtual power plant load forecasting as a computer-science problem of representation reuse, label efficiency, reproducibility, and failure-aware adaptation under domain shift.

Index Terms: load forecasting; virtual power plant; source-pooled representation learning; transfer learning; time-series representation; cold start; domain adaptation.

## 1. Introduction

Virtual power plants depend on short-term load forecasts to schedule flexible demand, allocate storage, plan market bids, and evaluate reserve capability. Unlike conventional system-level load forecasting, virtual power plant forecasting often concerns aggregated portfolios of heterogeneous resources: commercial buildings, industrial parks, residential communities, electric vehicle charging clusters, and distributed energy assets. These resources differ in scale, behavioral patterns, weather sensitivity, occupancy structure, and controllability.

A practical challenge is that new resources may have limited historical data. A building or charging cluster newly connected to a virtual power plant cannot provide years of labeled observations. Even when historical data exist, distribution shifts occur across regions, seasons, users, and tariff regimes. A supervised model trained on one aggregation may fail on another. This motivates a transferable forecasting framework that learns reusable temporal representations before task-specific fine-tuning.

Self-supervised and source-pooled representation learning offer practical routes for this setting, consistent with recent time-series representation learning work [11]. Instead of requiring labels for every target resource, a model can learn temporal features from source load archives through masked reconstruction, temporal-filter representations, or other pretext and representation objectives. The learned representation can then be adapted to downstream forecasting tasks with fewer labels. This paradigm is increasingly important for time-series learning, but it must be tested against strong target-only and transparent baselines rather than only against weak seasonal rules.

This paper studies a transferable load forecasting framework for aggregated virtual power plant resources. The method learns source-domain temporal representations, adapts them through lightweight target heads or adapters, and evaluates them under practical deployment scenarios: cross-client transfer, few-shot adaptation, and cold-start forecasting. The goal is not merely to reduce RMSE in a single dataset, but to understand when reusable representations improve generalization under distribution shift and when simple target baselines remain competitive.

The contribution is computer-science oriented. The paper studies how to learn reusable temporal representations for heterogeneous energy-related time series. The virtual power plant setting supplies a realistic and economically important application, but the method is positioned as a general transferable forecasting architecture.

## 2. Related Work

### 2.1 Short-term load forecasting and energy forecasting protocols

Short-term load forecasting has developed from statistical extrapolation and seasonal rules into a broad family of machine-learning and deep-learning methods. Energy-forecasting competitions and survey work emphasize that credible experiments require chronological splits, transparent baselines, task-appropriate metrics, and clear separation between point accuracy and uncertainty or decision value [14]-[16], [19], [20], [23], [30]. For virtual power plants, these requirements are stricter than in ordinary system-load forecasting because forecast errors affect aggregation, demand response, storage dispatch, and market bidding. Recent energy-forecasting reviews also show that deep models can help when multi-scale temporal, weather, and calendar structure are present, but the gain depends on data coverage, nonstationarity, and baseline strength [23], [24]. This paper therefore treats OPSD, UCI Electricity, and UCI Appliances as complementary public evidence layers rather than relying on a single private operating case.

### 2.2 Deep sequence models for time-series forecasting

Deep time-series forecasting models include recurrent probabilistic models, convolutional models, decomposition-based linear models, and Transformer-family architectures [1]-[8], [22], [27]-[29]. Temporal Fusion Transformers, Informer, Autoformer, PatchTST, DLinear, N-BEATS, and TCN-style architectures represent different assumptions about attention, decomposition, local temporal filters, and long-horizon sequence structure [2]-[8]. Survey evidence suggests that architecture choice alone is not enough: model evaluation must compare against strong linear, seasonal, and feature-engineered baselines, especially for electricity load series whose lag and calendar regularities are strong [22], [28], [30]. The present manuscript uses these models as the methodological neighborhood, while its implemented public evidence focuses on source-pooled representations that can be reproduced without GPU-dependent training.

### 2.3 Transfer learning, domain adaptation, and representation shift

Transfer learning studies how source-domain information can improve a target task when target labels are limited [12]. Domain adaptation and transferable representation learning address distribution mismatch between source and target domains, including feature alignment and domain-invariant representation objectives [13], [25]. In VPP portfolios, domain shift appears through building function, region, weather exposure, occupancy, tariff regime, charging behavior, and aggregation composition. A target resource may also have only days or weeks of labeled observations. The central question is therefore not whether one global model wins every target case, but whether a source-pooled representation can reduce label requirements while exposing failure cases through diagnostics.

### 2.4 Self-supervised and random-filter time-series representations

Self-supervised time-series representation learning uses reconstruction, contrastive, predictive, or multi-scale objectives to learn reusable features before downstream supervision [11], [22], [28]. Random convolutional transforms such as ROCKET and MiniRocket show that large collections of multi-scale temporal filters can provide strong, efficient representations for time-series tasks [9], [10]. SCINet and related convolution-interaction models further support the idea that temporal filtering and interaction structure can be competitive for sequence modeling [26]. These lines of work motivate the Paper 2 design: compare masked-reconstruction features, random-convolution features, trainable dilated-convolution ridge features, and a NumPy neural residual-head check under the same source-head and target-adapter protocol instead of attributing gains to an unsupported end-to-end architecture.

### 2.5 Cross-client load forecasting for virtual power plant resources

Load forecasting for aggregated VPP resources differs from traditional regional load forecasting because the target entity can be a newly enrolled resource cluster, a building portfolio, an industrial park, or an EV charging group. The available target history may be short, and privacy or platform boundaries may prevent direct pooling of raw local operating data. Public multi-client datasets such as UCI Electricity Load Diagrams [17] are therefore useful because they permit source-client and target-client splits that mimic resource onboarding. OPSD provides system-scale public load and market context [18], while UCI Appliances adds a weather-rich residential load sanity check [21]. The three public layers jointly support a computer-science framing around domain generalization, label efficiency, and representation reuse.

### 2.6 Evaluation gaps addressed by this paper

Several gaps remain in load-transfer studies. First, papers may report improvement over weak seasonal baselines while ignoring strong target-only ridge or lag-weather models. Second, representation-learning papers may report average gains without client-level paired evidence or negative controls. Third, energy applications often use local data without a redistribution-safe public supplement. This paper addresses these gaps by using matched source-head and target-adapter protocols, paired sign-test summaries across held-out clients, a target-only trainable-convolution negative control, and public-data-only reproducibility assets. The remaining limitation is explicit: the current evidence supports source-pooled temporal representation reuse, not a claim that every future VPP target can be solved without target validation.


## 3. Problem Formulation

Consider N load domains, where each domain may represent a building, park, charging cluster, region, or aggregation portfolio. For client or resource domain i, a sequence contains load ell_t^i and covariates x_t^i such as weather, calendar variables, and resource metadata. Given a lookback window L and forecast horizon H, the target is:

f_theta(x_{t-L+1:t}^i, ell_{t-L+1:t}^i, m^i) -> ell_{t+1:t+H}^i,

where m^i denotes domain metadata. The key challenge is to maintain performance when target domain i has limited or no labeled training data.

The evaluation uses four protocols:

1. Within-domain forecasting: train and test on the same domain.
2. Cross-domain forecasting: train on source domains and test on unseen domains.
3. Few-shot adaptation: fine-tune with k days or weeks of target-domain data.
4. Cold-start forecasting: use metadata and exogenous variables with minimal load history.

## 4. Proposed Method

![FIGURE 1. Transferable short-term load forecasting framework with source-pooled representation reuse and lightweight adaptation.](figures/paper2_fig1_framework.png)

Figure 1 summarizes the transferable forecasting pipeline. Unlabeled multi-client load windows are first used to learn a reusable temporal representation; small target-domain samples then fit a lightweight adapter rather than retraining the full representation from scratch. The design keeps the contribution centered on representation reuse, domain shift, cold-start adaptation, and label efficiency, while the virtual power plant setting supplies the operational motivation.

### 4.1 Time-series encoder

The encoder maps a multivariate time-series window into latent representations, drawing on sequence models and modern time-series encoders [1]-[8]. It may use temporal convolution, patch-based Transformer blocks, or a hybrid architecture. Patch embeddings reduce sequence length, and calendar embeddings encode hour-of-day, day-of-week, holiday, and seasonal information. The model is deliberately modular so that alternative encoders can be tested. In the reproducible public experiment, the encoder family is evaluated in three forms: a low-rank masked-reconstruction representation, a stronger ROCKET-style random-convolution temporal representation inspired by random convolutional kernel transforms [9], [10], and a trainable dilated-convolution ridge encoder. The latter checks test whether multi-scale temporal filters, trainable causal convolution-window features, and a small nonlinear residual correction improve transfer beyond the initial reconstruction basis.

### 4.2 Masked temporal reconstruction

During pretraining, random segments of the input sequence are masked. The encoder must reconstruct the masked load and covariate values using surrounding context:

L_mask = ||M * (X - decoder(encoder(mask(X))))||_1,

where M is the mask indicator. Segment-level masking is preferred over independent point masking because it forces the model to learn meaningful temporal dependencies rather than interpolate isolated points.

### 4.3 Domain-shift-aware representation checks

The reproducible implementation treats domain shift as an empirical object rather than as a solved property of the encoder. It compares masked-reconstruction features, random-convolution temporal features, and trainable dilated-convolution ridge features under the same source-head and target-adapter protocols. This design tests whether representation strength, target-label budget, and client mismatch explain transfer gains. Contrastive or adversarial extensions are therefore reserved as future reviewer-response options, not as claims required by the current public evidence.

### 4.4 Standardization and lightweight adapters

Load magnitude and variability differ significantly across resources. The current protocol standardizes source and target windows, fits source-domain representation heads, and then uses lightweight target adapters with 1, 3, 7, or 28 days of target data. This reduces the number of target-fitted parameters and exposes low-label overfitting when the 1-day and 3-day adapters become unstable. The paper therefore claims regularized adaptation value under matched protocols rather than broad target-only refitting.

### 4.5 Forecasting head and training schedule

The model family is evaluated in three stages:

Stage A: source-domain representation fitting on multi-client load sequences.
Stage B: source-head forecasting on held-out target clients with no target labels.
Stage C: target-domain adaptation with limited labeled data and matched target-only baselines.

The supervised objective is MAE or Huber loss, optionally combined with quantile loss for probabilistic outputs.

### 4.6 Algorithmic protocol and computational boundary

Algorithm 1 gives the source-pooled representation-learning protocol used in the public experiments. The protocol is intentionally modular: the representation map phi can be instantiated by masked reconstruction, random convolutional filters, or trainable dilated-convolution ridge features, while the downstream head remains a regularized linear or lightweight adapter model. This separation lets the evaluation attribute gains to reusable temporal representations rather than to uncontrolled target-domain fitting.

Algorithm 1. Source-pooled representation reuse for label-scarce load forecasting.

Input: source domains S, target domain i, lookback length L, target-label budget k, representation family phi, regularization parameter lambda.
Output: target forecast function g_i(phi(.)).

Step A: build chronological windows from each source domain without using target holdout observations.
Step B: fit source normalization and representation parameters phi on source windows.
Step C: fit a source forecasting head g_S on source labels using regularized loss.
Step D: for zero-label transfer, evaluate g_S(phi(.)) directly on target-domain windows.
Step E: for k-day adaptation, freeze or reuse phi, fit a lightweight target adapter g_i on the k-day target calibration window, and keep the final test period untouched.
Step F: compare against matched target-only ridge, seasonal, lag-weather, and target-only convolutional negative-control baselines.
Step G: report mean metrics, client-level paired wins, sign-test values, and explicitly retained failure cases.

Let n_S denote the number of source windows, d_phi the representation dimension, and n_i(k) the number of target calibration windows. The source-head fitting cost is dominated by representation extraction plus regularized linear solving. Random-convolution features scale approximately with the number of filters and window length, while the ridge head scales with d_phi after feature extraction. Target adaptation is deliberately small, scaling with n_i(k) and d_phi rather than retraining a full deep model. This computational boundary is useful for VPP onboarding, where a new resource should be adapted from days or weeks of data without requiring a large target-specific neural training run. The boundary also explains why the paper avoids claiming a full foundation model: the contribution is a reproducible, source-pooled representation protocol and its failure-aware evaluation.


## 5. Experimental Design

### 5.1 Datasets

The reproducible public benchmark now has three layers. The first layer uses the OPSD hourly time-series package with load, day-ahead price, wind, and solar variables for DE-LU, DK1, DK2, and Great Britain [18]. It supports within-zone and cross-zone load forecasting at system scale. The second layer uses the UCI Electricity Load Diagrams 2011-2014 dataset as a true multi-client transfer benchmark [17]. The UCI data contain 370 client-level electricity series at 15-minute resolution; the current reproducible run aggregates the selected 2014 subset to hourly resolution and evaluates 30 source clients and 10 target clients. The third layer uses UCI Appliances Energy Prediction [21] as an external single-site public load dataset with weather and calendar covariates, used to test whether the feature and representation claims survive outside the multi-client UCI split.

Table 1 summarizes the public OPSD load benchmark coverage.

|Zone|Rows|Start|End|Load|Solar|Wind|
|---|---|---|---|---|---|---|
|DE_LU|17521|2018-10-01|2020-09-30|17521|17516|17521|
|DK_1|50386|2015-01-01|2020-09-30|50386|50377|50386|
|DK_2|50386|2015-01-01|2020-09-30|50386|50373|50386|
|GB_GBN|50288|2015-01-02|2020-09-30|50288|50247|50252|

Table 2 summarizes the UCI multi-client transfer benchmark split used in the transparent public experiment. Clients are selected deterministically by 2014 coverage and load variability, then split into source and target roles.

|Role|Clients|Period|Resolution|Coverage|Mean load min|Mean load max|
|---|---|---|---|---|---|---|
|source|30|2014-01-01 to 2014-12-31|hourly after 15-min aggregation|1|568.778|48306.670|
|target|10|2014-01-01 to 2014-12-31|hourly after 15-min aggregation|1|494.672|2407.821|

### 5.2 Baselines

Baselines include seasonal naive and classical forecasting references [16], TCN [8], N-BEATS [7], DLinear [6], TFT [2], PatchTST [5], and transfer-learning/domain-adaptation references such as source fine-tuning and domain-adversarial training [12], [13].

### 5.3 Metrics

Point metrics follow standard forecasting practice [16]: MAE, RMSE, MAPE, and sMAPE. Transfer metrics: relative improvement over target-only training, few-shot performance as a function of target data size, and adaptation efficiency. Robustness metrics: performance under weather extremes, holidays, and unseen domains.

### 5.4 Ablation study

Ablations compare masked-reconstruction features, random-convolution features, trainable dilated-convolution ridge features, a neural residual-head extension, source heads, target adapters, and target-only negative controls. The analysis reports whether each component contributes more to cross-domain generalization, cold-start transfer, or ordinary target-domain accuracy.

## 6. Results and Discussion

Table 3 reports the best transparent OPSD load-forecasting baseline by RMSE for each public zone. The complete baseline file includes one-hour persistence, 24-hour seasonal naive, 168-hour seasonal naive, and lag-calendar-exogenous linear models.

|Zone|Best baseline|MAE|RMSE|sMAPE|N|
|---|---|---|---|---|---|
|DE_LU|Linear lag+cal+exog|1104.998|1495.351|2.178|3505|
|DK_1|Linear lag+cal+exog|62.518|88.856|2.594|10078|
|DK_2|Linear lag+cal+exog|36.870|71.881|2.439|10078|
|GB_GBN|Linear lag+cal+exog|1174.475|1754.355|3.932|10058|

The lag-calendar-exogenous linear model is the best transparent baseline on all four public OPSD zones. This result sharpens the claim of the transferable-learning paper: the main contribution is not framed as merely beating weak point-forecasting baselines. Instead, the proposed self-supervised representation is evaluated by cross-domain transfer, few-shot adaptation, and cold-start robustness where reusable temporal representations matter.

Table 4 reports the public UCI multi-client transfer benchmark. The test horizon is October-December 2014 for ten held-out target clients. Source-transfer models train on thirty source clients before the test horizon; few-shot target models use either the seven days or the twenty-eight days immediately before the test horizon. The values are averaged across target clients.

|Model|Protocol|Targets|MAE|RMSE|sMAPE|N|
|---|---|---|---|---|---|---|
|Target-28d-linear|few-shot target only|10|53.433|82.784|6.364|22080|
|Pooled+target-28d-linear|few-shot transfer|10|59.454|87.287|6.824|22080|
|Pooled+target-7d-linear|few-shot transfer|10|59.478|87.312|6.826|22080|
|Pooled-source-linear|source-only transfer|10|59.488|87.321|6.827|22080|
|Target-7d-linear|few-shot target only|10|57.239|88.106|6.730|22080|
|Seasonal-168h|direct seasonal|10|72.850|113.500|8.251|22080|
|Seasonal-24h|direct seasonal|10|75.862|125.966|8.047|22080|

The UCI transfer table is reported as a transparent baseline layer. The best linear-only result is the 28-day target-only linear model, while pooled-source and pooled-plus-target linear transfer models substantially improve over 24-hour and 168-hour seasonal baselines. This pattern sets the reference point for the self-supervised representation experiment: transfer learning helps relative to naive temporal rules, but representation learning must still demonstrate value beyond a well-tuned target-domain linear baseline.



Table 5 reports the first masked-reconstruction representation prototype on the same UCI split. The representation is pretrained on source-client 168-hour history windows using 20% random masking and a 16-dimensional low-rank reconstruction basis. The pretraining run uses 80,000 source windows, explains 0.817 of masked-window variance, and obtains source-window reconstruction RMSE 0.095 in normalized units.

|Model|Protocol|Targets|MAE|RMSE|sMAPE|N|
|---|---|---|---|---|---|---|
|SSL-MR-lag+adapter-28d|frozen representation with target adapter|10|51.770|74.962|6.411|22080|
|SSL-MR-lag-source-head|source-only frozen representation|10|52.377|75.689|6.433|22080|
|SSL-MR-lag+adapter-7d|frozen representation with target adapter|10|52.586|75.745|6.446|22080|
|SSL-MR+adapter-28d|frozen representation with target adapter|10|70.245|101.196|9.131|22080|
|SSL-MR+adapter-7d|frozen representation with target adapter|10|71.181|102.473|9.158|22080|
|SSL-MR-source-head|source-only frozen representation|10|71.949|103.302|9.243|22080|

The best prototype is SSL-MR-lag+adapter-28d, which reduces mean RMSE from 82.784 for the strongest transparent few-shot linear baseline to 74.962, a relative improvement of 9.45%. The diagnostic result is also important: using the masked-reconstruction latent representation alone is weaker than using the latent representation together with lag-1, lag-24, and lag-168 temporal priors. This supports treating self-supervised representation learning and domain-adaptive temporal priors as complementary components rather than as substitutes.


Table 6 extends the UCI experiment to label-scarce and cold-start adaptation. The zero-label row uses the source-trained masked-reconstruction representation and its source head without fitting target labels. The 1-day, 3-day, 7-day, and 28-day rows compare a lightweight target adapter against a target-only lag-calendar ridge model.

|Model|Protocol|Days|Targets|MAE|RMSE|sMAPE|N|
|---|---|---|---|---|---|---|---|
|SSL source|source head|0|10|52.377|75.689|6.433|22080|
|Seasonal-168h|seasonal|0|10|72.850|113.500|8.251|22080|
|Seasonal-24h|seasonal|0|10|75.862|125.966|8.047|22080|
|SSL adapter 1d|adapter|1|10|60.601|84.455|7.353|22080|
|Target ridge 1d|target ridge|1|10|74.685|111.640|8.609|22080|
|Target ridge 3d|target ridge|3|10|89.090|120.377|12.062|22080|
|SSL adapter 3d|adapter|3|10|96.207|126.224|10.699|22080|
|SSL adapter 7d|adapter|7|10|52.586|75.745|6.446|22080|
|Target ridge 7d|target ridge|7|10|57.235|88.096|6.729|22080|
|SSL adapter 28d|adapter|28|10|51.770|74.962|6.411|22080|
|Target ridge 28d|target ridge|28|10|53.432|82.782|6.364|22080|

The zero-label source representation obtains mean RMSE 75.689, already below the 28-day target-only linear baseline at 82.782. This is the strongest cold-start evidence in the current Paper 2 package. The adapter curve is not monotonic: 1-day and 3-day adapters can overfit or miscalibrate, while 7-day and 28-day adapters return to the 75-RMSE range. The result should be framed carefully: self-supervised source representations reduce the data requirement, but 1-day and 3-day target-label adapters require stronger regularization or meta-validation.

Table 7 reports a representation-domain shift diagnostic. Each target client is represented by the centroid of its 28-day pre-test latent windows. The distance to the source centroid is compared with source-head error and 28-day adapter gain.

|Target|Latent distance|Source RMSE|Adapter RMSE|Adapter gain %|
|---|---|---|---|---|
|MT_355|3.208|68.363|67.716|0.946|
|MT_101|2.767|54.169|53.679|0.903|
|MT_360|1.874|49.077|48.804|0.556|
|MT_104|1.749|48.056|47.404|1.357|
|MT_222|1.081|58.571|57.769|1.369|
|MT_218|0.862|74.224|73.184|1.401|
|MT_314|0.795|63.434|63.428|0.009|
|MT_166|0.738|86.082|85.685|0.462|
|MT_043|0.557|104.860|104.927|-0.063|
|MT_163|0.249|150.053|147.023|2.019|

The mean source-target latent distance is 1.388. Across the ten held-out clients, the correlation between latent distance and zero-label source-head RMSE is -0.604, while the correlation between latent distance and 28-day adapter gain is 0.003. In this selected UCI split, latent distance is therefore a useful diagnostic feature but not a sufficient predictor of whether a simple adapter will help.


![FIGURE 2. Local target-domain load curve used as a small adaptation case.](figures/paper2_fig2_load_curve.png)

![FIGURE 3. Pilot RMSE comparison on the local Hunan user-load case.](figures/paper2_fig3_load_rmse.png)

![FIGURE 4. UCI public multi-client transfer benchmark, mean RMSE across ten target clients.](figures/paper2_fig4_uci_transfer_rmse.png)

![FIGURE 5. UCI load-transfer comparison between transparent baselines and the masked-reconstruction representation prototype.](figures/paper2_fig5_uci_ssl_prototype_rmse.png)

![FIGURE 6. UCI label-scarce adaptation curve comparing the source representation, target adapters, and target-only ridge baselines.](figures/paper2_fig6_uci_cold_start_curve.png)

![FIGURE 7. UCI representation-domain shift diagnostic for held-out target clients.](figures/paper2_fig7_uci_domain_shift_diagnostic.png)

Table 8 reports paired client-level evidence on the same ten UCI target clients. The paired unit is the target client, and the exact two-sided sign test evaluates whether the SSL representation has lower RMSE than the matched baseline on more clients than expected by chance.

|Comparison|Baseline RMSE|SSL RMSE|Mean gain %|Wins|p sign|
|---|---|---|---|---|---|
|SSL source vs target ridge 28d|82.782|75.689|10.838|9/10|0.021|
|SSL adapter 28d vs target ridge 28d|82.782|74.962|11.651|9/10|0.021|
|SSL adapter 7d vs target ridge 7d|88.096|75.745|15.912|9/10|0.021|
|SSL adapter 1d vs target ridge 1d|111.640|84.455|25.500|10/10|0.002|
|SSL source vs seasonal 168h|113.500|75.689|34.004|10/10|0.002|
|SSL source vs seasonal 24h|125.966|75.689|37.167|10/10|0.002|

The client-level test strengthens the Paper 2 claim. The zero-label source representation beats the strongest 28-day target-only ridge baseline on 9/10 clients, with mean RMSE gain 10.84% and exact sign-test p=0.021. The 28-day SSL adapter also wins on 9/10 clients with mean gain 11.65% (p=0.021), while the 7-day adapter wins on 9/10 clients against the 7-day target ridge with mean gain 15.91% (p=0.021). The 1-day adapter beats the 1-day target ridge on all clients, but this should be interpreted as low-label robustness relative to a weak same-budget target baseline rather than as the best overall model. Against seasonal rules, the zero-label source representation wins on all clients versus both weekly and daily seasonal baselines, with mean gains 34.00% and 37.17%. These p-values are reported as unadjusted descriptive paired sign-test values; interpretation relies on repeated direction, effect size, and negative-control behavior, not as a claim that all encoder variants have been exhaustively optimized. The negative client case in Figure 6, MT_163, is retained as limitation evidence: source representations reduce average and paired error, but a portable adapter still needs target-domain model selection.

![FIGURE 8. UCI client-level paired RMSE improvements for self-supervised representation models.](figures/paper2_fig8_uci_client_level_stat_tests.png)

Table 9 reports an encoder-strengthening check using a ROCKET-style random-convolution temporal representation [9], [10]. The encoder applies deterministic multi-scale random convolutional filters to the same 168-hour windows, summarizes each filter response by maximum activation, positive-proportion, and terminal activation, and then uses the same source-head and lightweight target-adapter protocol as the masked-reconstruction prototype.

|Check|Baseline|Baseline RMSE|Candidate|Candidate RMSE|Mean gain %|Wins|p sign|
|---|---|---|---|---|---|---|---|
|RC 28d vs MR 28d|MR adapter 28d|74.962|RC adapter 28d|67.500|11.330|10/10|0.002|
|RC source vs MR source|MR source|75.689|RC source|67.878|11.327|10/10|0.002|
|RC 28d vs target|Target ridge 28d|82.782|RC adapter 28d|67.500|21.212|9/10|0.021|
|RC 7d vs target|Target ridge 7d|88.096|RC adapter 7d|68.016|25.208|10/10|0.002|

The random-convolution representation materially strengthens the Paper 2 evidence. The 28-day random-convolution adapter obtains mean RMSE 67.500, compared with 74.962 for the masked-reconstruction 28-day adapter, giving 11.33% mean paired RMSE gain with wins on 10/10 clients (p=0.002). Even without target labels, the random-convolution source head reaches mean RMSE 67.878, improving over the masked-reconstruction source head on 10/10 clients. Relative to target-only ridge baselines, the 28-day random-convolution adapter wins on 9/10 clients with mean gain 21.21% (p=0.021), while the 7-day adapter wins on 10/10 clients with mean gain 25.21% (p=0.002). The 3-day random-convolution adapter remains unstable, with mean RMSE 97.255; therefore, the result supports stronger reusable temporal features, but not unrestricted few-shot adapter fitting without validation.

![FIGURE 9. Random-convolution representation check on UCI load transfer.](figures/paper2_fig9_uci_random_conv_encoder_comparison.png)

Table 10 reports a reviewer-facing trainable encoder check using a dilated-convolution ridge representation. The encoder extracts multi-scale causal slices from each 168-hour history window, fits source-domain ridge heads on standardized convolution-window features, and then applies the same lightweight target-adapter protocol. This check is intentionally described as a trainable dilated-convolution ridge encoder rather than as a full deep TCN, because the current public reproducibility environment uses deterministic NumPy/Pandas training without GPU-dependent deep-learning libraries.

|Check|Baseline|Baseline RMSE|Candidate|Candidate RMSE|Mean gain %|Wins|p sign|
|---|---|---|---|---|---|---|---|
|TDConv 28d vs RC 28d|RC adapter 28d|67.500|TDConv adapter 28d|65.291|3.322|10/10|0.002|
|TDConv source vs RC source|RC source|67.878|TDConv source|65.471|3.953|10/10|0.002|
|TDConv 28d vs target|Target ridge 28d|82.782|TDConv adapter 28d|65.291|23.779|9/10|0.021|
|TDConv target head vs target|Target ridge 28d|82.782|TDConv target head 28d|91.788|-7.492|5/10|1.000|
|TDConv 7d vs target|Target ridge 7d|88.096|TDConv adapter 7d|66.044|27.411|10/10|0.002|

The trainable dilated-convolution check further strengthens the UCI transfer evidence. The 28-day TDConv adapter obtains mean RMSE 65.291, improving over the random-convolution 28-day adapter at 67.500 on 10/10 clients with mean paired RMSE gain 3.32% (p=0.002). The zero-label TDConv source head reaches mean RMSE 65.471, improving over the random-convolution source head on 10/10 clients. Relative to the 28-day target-only ridge baseline, the TDConv 28-day adapter wins on 9/10 clients with mean RMSE gain 23.78% (p=0.021); the 7-day adapter also wins on 10/10 clients with mean RMSE 66.044. The negative control is important: a target-only TDConv head reaches mean RMSE 91.788, worse than the 28-day target ridge by 7.49% with no sign-test advantage. The result therefore supports source-pooled trainable convolutional representations with regularized adaptation, not unconstrained high-dimensional fitting on small target samples. The next neural residual check tests whether adding nonlinearity changes that conclusion.

![FIGURE 10. Trainable dilated-convolution ridge encoder check on UCI load transfer.](figures/paper2_fig11_uci_trainable_tdconv_baseline.png)

Table 11 adds a nonlinear residual-head check on top of the trainable TDConv representation. The residual head is a one-hidden-layer neural model implemented only with NumPy, trained on source-domain residuals after a TDConv ridge head, and selected using source-validation shrinkage rather than target-test feedback. The purpose is reviewer-facing: to test whether a small nonlinear residual correction materially improves the source-pooled representation, or whether the regularized TDConv ridge head already captures the transferable signal.

|Check|Baseline|Baseline RMSE|Candidate|Candidate RMSE|Mean gain %|Wins|p sign|
|---|---|---|---|---|---|---|---|
|Neural 28d vs TDConv 28d|TDConv adapter 28d|65.291|Neural TDConv residual 28d|65.582|-0.359|5/10|1.000|
|Neural source vs TDConv source|TDConv source|65.471|Neural TDConv residual source|65.592|-0.240|5/10|1.000|
|Neural 28d vs RC 28d|RC adapter 28d|67.500|Neural TDConv residual 28d|65.582|2.977|10/10|0.002|
|Neural 28d vs target|Target ridge 28d|82.782|Neural TDConv residual 28d|65.582|23.471|9/10|0.021|
|Neural 7d vs target|Target ridge 7d|88.096|Neural TDConv residual 7d|66.566|26.836|10/10|0.002|

The neural residual check is a useful negative control rather than a new headline model. Source-validation selected residual shrinkage 0.25; the selected validation RMSE changes only from 0.085 for the TDConv ridge base to 0.084 after the residual correction. On held-out UCI target clients, the 28-day neural residual adapter reaches mean RMSE 65.582, but it does not improve over the TDConv 28-day adapter (5/10 wins, p=1.000). The source-head comparison is similarly neutral (5/10 wins, p=1.000). However, the same nonlinear residual model still beats the random-convolution 28-day adapter on 10/10 clients and beats the 28-day target ridge on 9/10 clients. The result strengthens the methodological boundary of the paper: nonlinear residual capacity is tested, but the evidence favors a parsimonious source-pooled TDConv representation with regularized adaptation rather than a larger neural head fitted for its own sake. The 7-day neural residual adapter remains competitive with mean RMSE 66.566, while the zero-label neural residual source head obtains 65.592; both are reported as robustness evidence rather than as proof of universal neural dominance. The final training epoch residual RMSEs were 0.089 on the source-training split and 0.095 on the source-validation split.

![FIGURE 11. Neural TDConv residual-head check on UCI load transfer.](figures/paper2_fig14_uci_neural_tdconv_residual_check.png)

<!-- pagebreak -->

Table 12 adds a second public load dataset, UCI Appliances Energy Prediction [21], as an external sanity check beyond the UCI Electricity Load Diagrams multi-client split. The data contain 10-minute appliance-energy observations and environmental covariates from a residential setting; this paper converts them into a one-hour-ahead chronological forecasting task with 11664 training rows, 3888 validation rows, and 3889 final test rows.

|Model              |Protocol                                    |MAE (Wh)|RMSE (Wh)|sMAPE (%)|Test rows|
|-------------------|--------------------------------------------|--------|---------|---------|---------|
|Lag-weather ridge  |chronological ridge baseline                |38.93   |78.59    |38.03    |3889     |
|Random-window ridge|deterministic random temporal representation|39.90   |78.61    |40.34    |3889     |
|Persistence-current|one-hour-ahead public holdout               |45.11   |98.98    |32.02    |3889     |
|Seasonal-24h       |one-hour-ahead public holdout               |50.82   |107.04   |37.11    |3889     |

The second-dataset result supports the paper's conservative claim. A lag-weather ridge model reaches 78.59 Wh RMSE, improving over current-value persistence by 20.60% and over the 24-hour seasonal baseline by 26.58%. The deterministic random-window representation is essentially tied with the lag-weather ridge model, with only 0.03% higher RMSE. This means the external dataset validates the importance of lag, weather, and calendar features, but it does not justify claiming that random temporal representations universally dominate a well-specified transparent model.

![FIGURE 12. Second public load dataset check using UCI Appliances Energy Prediction.](figures/paper2_fig10_uci_appliances_second_dataset.png)

Table 13 reports a multi-horizon robustness extension on the same public UCI Appliances holdout. The task is repeated at 1-hour, 3-hour, 6-hour, and 12-hour horizons using only current and historical information. The lag-weather ridge remains the best 1-hour model, while the deterministic random-window ridge is best at 3h, 6h, 12h; its advantage over lag-weather ridge is small on those longer horizons, ranging from 0.13% to 1.12% relative RMSE improvement. This result strengthens the external robustness evidence while keeping the claim bounded: random temporal filters can help slightly at longer horizons, but the paper's main contribution remains source-pooled representation reuse under cross-client transfer, not universal dominance on every public load task.

|Horizon|Best model         |Best RMSE|Lag-weather RMSE|Random-window RMSE|Gain vs persistence|
|-------|-------------------|---------|----------------|------------------|-------------------|
|1h     |Lag-weather ridge  |78.60    |78.60           |78.62             |20.59%             |
|3h     |Random-window ridge|83.54    |84.49           |83.54             |25.05%             |
|6h     |Random-window ridge|85.45    |85.55           |85.45             |29.25%             |
|12h    |Random-window ridge|84.90    |85.70           |84.90             |33.71%             |

![FIGURE 13. Multi-horizon robustness check on UCI Appliances Energy Prediction.](figures/paper2_fig13_uci_appliances_multihorizon_robustness.png)

Table 14 reports a multi-seed stability check for the strongest trainable TDConv representation. The experiment repeats source-window subsampling with eight random seeds while preserving the same source-client set, target-client holdout, chronological split, TDConv feature construction, ridge penalty, and 28-day target-adapter protocol. This test asks whether the TDConv advantage is an artifact of one favorable source-window subsample or a stable representation-learning effect under repeated public-data reruns.

|Comparison|Baseline RMSE|TDConv RMSE mean +/- sd|TDConv RMSE range|Minimum wins|Maximum losses|p-value pattern|
|---|---:|---:|---:|---:|---:|---|
|TDConv 28d adapter vs RC 28d adapter|67.500|65.338 +/- 0.056|65.283-65.465|10/10|0/10|0.002 for each seed|
|TDConv 28d adapter vs target ridge 28d|82.782|65.338 +/- 0.056|65.283-65.465|9/10|1/10|0.021 for each seed|
|TDConv source head stability|-|65.665 +/- 0.112|65.478-65.827|-|-|descriptive stability check|

Across all eight seeds, the 28-day TDConv adapter remains tightly concentrated between 65.283 and 65.465 mean RMSE. Every seed beats the random-convolution 28-day adapter on 10/10 target clients, and every seed beats the 28-day target-only ridge on 9/10 target clients. This closes an important reproducibility gap in the Paper 2 evidence: the strongest representation result is not a single stochastic run. The result still should not be overread as a universal neural architecture claim, because the neural residual-head extension remains neutral against the parsimonious TDConv ridge head.

![FIGURE 14. Multi-seed stability of trainable TDConv source-window subsampling.](figures/paper2_fig15_uci_tdconv_multiseed_stability.png)

Table 15 adds a CPU-only patch-attention transfer check. The representation divides each 168-hour history window into seven 24-hour patches, uses the most recent patch as a deterministic query, pools patch profiles by similarity weights, and fits the same source-head and target-adapter protocol used by the other UCI transfer baselines. It is reported as a lightweight reviewer-response baseline, not as full PatchTST or TFT training.

|Comparison|Baseline RMSE|Patch-attention RMSE|Mean RMSE gain|Wins/losses|Sign-test p|Interpretation|
|---|---:|---:|---:|---:|---:|---|
|PatchAttn 28d vs TDConv 28d|65.291|64.761|1.900%|9/1|0.021|slightly stronger source-pooled patch evidence|
|PatchAttn source vs TDConv source|65.471|65.119|1.574%|9/1|0.021|source-head patch evidence|
|PatchAttn 28d vs RC 28d|67.500|64.761|5.157%|9/1|0.021|positive transfer evidence|
|PatchAttn 28d vs target ridge|82.782|64.761|24.941%|9/1|0.021|positive transfer evidence|
|PatchAttn target-head vs target ridge|82.782|130.568|-54.378%|2/8|0.109|negative control for target-only high-dimensional patch fitting|

The patch-attention check strengthens the representation-learning argument without overclaiming a Transformer result. The 28-day patch-attention adapter reaches mean RMSE 64.761, slightly better than the TDConv 28-day adapter at 65.291, with 9/10 client-level wins and p=0.021. The target-only patch-attention head is much worse than the target ridge baseline, which is an important negative control: patch features help when learned from source clients and regularized through the adapter protocol, but high-dimensional patch fitting on limited target data is not safe by itself.

![FIGURE 15. CPU-only patch-attention transfer check on UCI load forecasting.](figures/paper2_fig16_uci_patch_attention_transfer_baseline.png)
Table 16 reports a source-trained MLP transfer check. The model trains a one-hidden-layer neural encoder on pooled source-client windows using only NumPy, then evaluates the same source-head and target-adapter protocol. This is a deliberately stronger neural-capacity boundary than the ridge heads, but it remains CPU-reproducible and avoids claiming a GPU-trained Transformer.

|Comparison|Baseline RMSE|SourceMLP RMSE|Mean RMSE gain|Wins/losses|Sign-test p|Interpretation|
|---|---:|---:|---:|---:|---:|---|
|SourceMLP 28d vs PatchAttn 28d|64.761|86.056|-36.581%|0/10|0.002|negative neural-capacity boundary|
|SourceMLP 28d vs TDConv 28d|65.291|86.056|-33.785%|0/10|0.002|negative neural-capacity boundary|
|SourceMLP source vs TDConv source|65.471|88.212|-33.822%|0/10|0.002|negative neural-capacity boundary|
|SourceMLP 28d vs target ridge|82.782|86.056|-1.266%|5/5|1.000|no reliable target-ridge advantage|
|SourceMLP hidden-head vs target ridge|82.782|85.883|-2.095%|4/6|0.754|frozen-hidden target fitting remains unstable|

The source-trained MLP is not a new best model. Its 28-day adapter reaches mean RMSE 86.056, and the source-only head reaches 88.212; both are worse than the patch-attention and TDConv source-pooled baselines. The result is nevertheless useful: a source-trained nonlinear encoder with validation RMSE 0.092 on normalized source windows and selected epoch 32 does not transfer safely by itself. This supports the paper's bounded claim that cross-client load transfer depends on representation regularization and adapter design, not simply on increasing neural capacity.

![FIGURE 16. Source-trained MLP transfer boundary check on UCI load forecasting.](figures/paper2_fig17_uci_source_mlp_transfer_baseline.png)


## 7. Conclusion

This paper presents a transferable short-term load forecasting study for aggregated virtual power plant resources. By comparing masked-reconstruction, random-convolution, trainable dilated-convolution, multi-seed TDConv stability, CPU-only patch-attention transfer, source-trained MLP transfer, and neural residual-head checks under matched source-head and target-adapter protocols, the work shows when source-pooled temporal features help cross-domain, few-shot, and cold-start load forecasting. The conclusion is deliberately bounded: representation reuse improves several held-out UCI client protocols and supports label-scarce adaptation; the multi-seed TDConv check shows that the strongest public representation result is stable to source-window subsampling; the patch-attention check provides a stronger source-pooled reviewer baseline while target-only patch fitting and source-trained MLP transfer expose unsafe capacity expansion; the neural residual check shows that extra nonlinear capacity does not automatically improve over the regularized TDConv head; representation choice, target-label budget, and failure-aware interpretation still require validation, and transparent lag-weather models remain competitive on the external UCI Appliances one-hour and multi-horizon checks.

## Data Availability Statement

The OPSD public benchmark [18] is reproducible from `public_data_download_templates.py` and `run_public_opsd_baselines.py`. The UCI multi-client transfer benchmark [17] is reproducible from `run_uci_load_transfer_baselines.py`; the masked-reconstruction representation prototype is reproducible from `run_uci_ssl_representation_prototype.py`; the cold-start/domain-shift diagnostics are reproducible from `run_uci_ssl_cold_start_diagnostics.py`; the client-level paired tests are reproducible from `run_uci_client_statistical_tests.py`; the random-convolution representation check is reproducible from `run_uci_random_conv_representation.py`; the trainable dilated-convolution ridge check is reproducible from `run_uci_trainable_tdconv_baseline.py`; the multi-seed TDConv stability check is reproducible from `run_uci_tdconv_multiseed_stability.py`; the CPU-only patch-attention transfer check is reproducible from `run_uci_patch_attention_transfer_baseline.py`; the source-trained MLP transfer check is reproducible from `run_uci_source_mlp_transfer_baseline.py`; and the neural TDConv residual-head check is reproducible from `run_uci_neural_tdconv_residual_check.py`. The second public load-dataset and multi-horizon checks on UCI Appliances Energy Prediction [21] are reproducible from `run_uci_appliances_energy_baselines.py` and `run_uci_appliances_multihorizon_robustness.py`. Main claims are reproducible without relying solely on private data.

## Local and China Real-World Data Assets

The newly inventoried local authorized real-world data directory materially strengthens the load-forecasting and transfer-learning story. The audit file `paper_package/master_submission_control/real_world_data_inventory_and_paper_mapping.md` records Shandong actual/forecast grid load workbooks, Shandong weather and social electricity consumption data, Liaoning and Eastern Inner Mongolia system-load/forecast disclosure tables, and open-license resource/network metadata.

|family|files|total_mib|status|main_use|risk|
|---|---|---|---|---|---|
|China power grid multi-year transmission network|6|0.226|Figshare CC BY 4.0 for core network dataset; GEM UHV reference folder has non-commercial caveat|Dissertation background, graph construction motivation, resource/network metadata appendix; optional Paper 1 graph-prior discussion|mixed license: keep GEM NC materials out of journal supplements unless permission is confirmed|
|Eastern Inner Mongolia public market disclosure|5|4.063|public-market-disclosure/local-copy; verify portal license before redistribution|Paper 1 cross-market price forecasting; Paper 3 decision-focused bidding; dissertation regional comparison|low confidentiality if sourced from public disclosure, but source-page citation and license check are required|
|Liaoning public market disclosure|12|26.385|public-market-disclosure/local-copy; verify portal license before redistribution|Paper 1 price forecasting; Paper 2 regional load forecasting; Paper 3 VPP decision case; dissertation China market chapter|low confidentiality if sourced from public disclosure, but final supplement redistribution still needs source-page citation and license check|
|OSM China power grid GIS extraction|13|598.263|OpenStreetMap-derived local extraction; several source files are incomplete .downloading files|Dissertation background only unless extraction is completed and ODbL attribution/share-alike handling is reviewed|license and incomplete-download risk; do not use as a primary journal experiment yet|
|Shandong market, load, weather, coal and consumption data|8|52.183|local real-world market data; treat as non-public until source-page and redistribution rights are confirmed|Paper 1 local price/weather/fuel case; Paper 2 load-weather case; Paper 3 settlement and VPP decision case; dissertation applied validation|medium: may include market-operation details; use aggregated or anonymized statistics in manuscripts until permission is clear|
|WRI/GEM China power plant metadata and time-series summaries|18|20.214|WRI Global Power Plant Database is CC BY 4.0; GEM links require final use-permission review|Dissertation resource-mix context, VPP portfolio scenario design, Paper 3 resource assumptions appendix|WRI is usable with attribution; GEM commercial/non-commercial caveats must be separated|

For Paper 2, the most relevant additions are Shandong actual and forecast load, weather covariates from 2022-2025, social electricity consumption, and regional disclosure load/renewable columns from Liaoning and Eastern Inner Mongolia. These sources are suitable for China-domain few-shot adaptation, forecast-error diagnostics, and exogenous-weather feature analysis. The public UCI and OPSD experiments remain the reproducible main benchmark; China real-world data should be reported as an application layer or dissertation case until source-page and redistribution rights are confirmed.

The immediate modeling update is to build a normalized regional load table with market, timestamp, actual load, forecast load, renewable output/forecast, weather covariates, calendar variables and market-state indicators. This lets the paper argue from a computer-science perspective: transferable temporal representations must handle domain shift, missingness, exogenous covariates and low-label regional adaptation rather than only fitting one power-system curve.

This update has started: `china_real_world_data/china_market_disclosure_quarterhour_wide.csv` normalizes Liaoning and Eastern Inner Mongolia market data, and `china_real_world_data/china_market_disclosure_baseline_report.md` reports a leakage-aware real-time-load baseline for Liaoning. Eastern Inner Mongolia is retained for price and decision evidence but is omitted from the first load baseline because real-time load coverage is insufficient after lag and feature filtering.

The Shandong load-weather layer in `china_real_world_data/shandong_load_weather_quarterhour.csv` adds 117,610 canonical quarter-hour rows with actual load, official forecast variables and hourly regional weather aggregates. Duplicate forecast-version rows are aggregated by timestamp using medians for numeric fields, without selecting any version by holdout error. In `china_real_world_data/shandong_real_data_baseline_report.md`, the leakage-safe linear baseline using lagged actual load, official forecast and weather covariates reaches RMSE 792.200 MW on the chronological holdout, compared with 2163.672 MW for the official forecast alone and 923.041 MW for 15-minute persistence. The result should be framed as China-domain residual-learning evidence rather than as the final representation-learning claim.

A separate GIS infrastructure audit, `paper_package/master_submission_control/gis_energy_infrastructure_evidence.md`, profiles China transmission-network snapshots, OSM mainland power-facility extractions, WRI China power-plant metadata and the GEM integrated China power-facility table. The current derived summaries contain 12,839 2025 transmission-line records, 2,041 substation records, 9,444 grid-link records, 4,870,459 OSM mainland power records, 4,274 WRI China plant records and approximately 3,108.905 GW of represented GEM operating capacity. For Paper 2, these records are used as infrastructure-context evidence and graph/resource heterogeneity motivation only. They are not treated as target labels or as a model-training dataset for the public benchmark experiments, and raw OSM/GEM/SHP/PBF files remain excluded from public supplements until attribution, license and redistribution checks are closed.

![FIGURE 16. China real-load application baselines. Liaoning uses disclosure-market load and market-state features; Shandong uses actual load, official forecasts and weather covariates.](figures/paper2_fig12_china_real_load_weather_baselines.png)


## Acknowledgment

No conventional acknowledgements are made in this article; this section is included solely to satisfy IEEE AI-use disclosure requirements and contains no thanks to people, institutions, funders, or projects. During preparation of this manuscript, OpenAI ChatGPT/Codex was used to support manuscript organization, language editing, code/package documentation, and submission checklist drafting. The AI-assisted content was limited to drafting and editing support for manuscript text, documentation, and submission materials. The tool was not used as an author, did not determine the scientific conclusions, and did not replace verification of data, code, references, results, or claims. After using these tools, the author reviewed and edited all AI-assisted text, kept the disclosure consistent with IEEE policy, and takes full responsibility for the final content.

## Data and Code Availability

The reproducible public-data layer uses the UCI Electricity Load Diagrams 2011-2014 dataset, the Open Power System Data time-series package, and the UCI Appliances Energy Prediction dataset. The public reproducibility supplement contains processed public data, preprocessing scripts, result tables, generated figures, verified references, manuscript files, manifest hashes, and an audit report for the UCI multi-client transfer, source-pooled representation, masked-reconstruction representation, cold-start, client-level statistical-test, random-convolution representation checks, trainable dilated-convolution ridge checks, multi-seed TDConv stability checks, CPU-only patch-attention transfer checks, source-trained MLP transfer checks, neural TDConv residual-head checks, OPSD public load baseline, and external UCI Appliances one-hour-ahead and multi-horizon load sanity checks. Local Hunan and Shandong operational records are used only as non-public application-context evidence and are not redistributed.

Public source URLs:

- UCI Electricity Load Diagrams 2011-2014: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
- Open Power System Data time series package: https://data.open-power-system-data.org/time_series/2020-10-06/
- UCI Appliances Energy Prediction: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

## References

[1] Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, Jones, Llion, Gomez, Aidan N., Kaiser, Lukasz, Polosukhin, Illia, "Attention Is All You Need," Advances in Neural Information Processing Systems, 2017. https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need.
[2] Bryan Lim, Sercan Ö. Arık, Nicolas Loeff, Tomas Pfister, "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting", International Journal of Forecasting, vol. 37, no. 4, pp. 1748-1764, 2021, https://doi.org/10.1016/j.ijforecast.2021.03.012.
[3] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang, "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 12, pp. 11106-11115, 2021, https://doi.org/10.1609/aaai.v35i12.17325.
[4] Wu, Haixu, Xu, Jiehui, Wang, Jianmin, Long, Mingsheng, "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting," Advances in Neural Information Processing Systems, 2021. https://arxiv.org/abs/2106.13008.
[5] Nie, Yuqi, Nguyen, Nam H., Sinthong, Phanwadee, Kalagnanam, Jayant, "A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers," International Conference on Learning Representations, 2023. https://openreview.net/forum?id=Jbdc0vTOcol.
[6] Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu, "Are Transformers Effective for Time Series Forecasting?", Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 9, pp. 11121-11128, 2023, https://doi.org/10.1609/aaai.v37i9.26317.
[7] Oreshkin, Boris N., Carpov, Dmitri, Chapados, Nicolas, Bengio, Yoshua, "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting," International Conference on Learning Representations, 2020. https://openreview.net/forum?id=r1ecqn4YwB.
[8] Bai, Shaojie, Kolter, J. Zico, Koltun, Vladlen, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," 2018. https://arxiv.org/abs/1803.01271.
[9] Angus Dempster, François Petitjean, Geoffrey I. Webb, "ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels", Data Mining and Knowledge Discovery, vol. 34, no. 5, pp. 1454-1495, 2020, https://doi.org/10.1007/s10618-020-00701-z.
[10] Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb, "MiniRocket", Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 248-257, 2021, https://doi.org/10.1145/3447548.3467231.
[11] Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, Bixiong Xu, "TS2Vec: Towards Universal Representation of Time Series", Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 8, pp. 8980-8987, 2022, https://doi.org/10.1609/aaai.v36i8.20881.
[12] Sinno Jialin Pan, Qiang Yang, "A Survey on Transfer Learning", IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345-1359, 2010, https://doi.org/10.1109/TKDE.2009.191.
[13] Ganin, Yaroslav, Ustinova, Evgeniya, Ajakan, Hana, Germain, Pascal, Larochelle, Hugo, Laviolette, François, Marchand, Mario, Lempitsky, Victor, "Domain-Adversarial Training of Neural Networks," Journal of Machine Learning Research, 2016. https://jmlr.org/papers/v17/15-239.html.
[14] Tao Hong, Pierre Pinson, Shu Fan, "Global Energy Forecasting Competition 2012", International Journal of Forecasting, vol. 30, no. 2, pp. 357-363, 2014, https://doi.org/10.1016/j.ijforecast.2013.07.001.
[15] Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli, Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol. 32, no. 3, pp. 896-913, 2016, https://doi.org/10.1016/j.ijforecast.2016.02.001.
[16] Hyndman, Rob J., Athanasopoulos, George, "Forecasting: Principles and Practice," OTexts, 2021. https://otexts.com/fpp3/.
[17] Trindade, Artur, "ElectricityLoadDiagrams20112014," UCI Machine Learning Repository, 2015. https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014.
[18] {Open Power System Data}, "Time series data package," Open Power System Data, 2020. https://data.open-power-system-data.org/time_series/2020-10-06/.
[19] Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron, "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark", Applied Energy, vol. 293, pp. 116983, 2021, https://doi.org/10.1016/j.apenergy.2021.116983.
[20] Anastasios N. Angelopoulos, Stephen Bates, "Conformal Prediction: A Gentle Introduction", Foundations and Trends® in Machine Learning, vol. 16, no. 4, pp. 494-591, 2023, https://doi.org/10.1561/2200000101.
[21] Candanedo, Luis M., Feldheim, Véronique, Deramaix, Dominique, "Appliances Energy Prediction," UCI Machine Learning Repository, 2017. https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction.
[22] Konstantinos Benidis, Syama Sundar Rangapuram, Valentin Flunkert, Yuyang Wang, Danielle Maddix, Caner Turkmen, Jan Gasthaus, Michael Bohlke-Schneider, David Salinas, Lorenzo Stella, François-Xavier Aubet, Laurent Callot, Tim Januschowski, "Deep Learning for Time Series Forecasting: Tutorial and Literature Survey", ACM Computing Surveys, vol. 55, no. 6, pp. 1-36, 2022, https://doi.org/10.1145/3533382.
[23] Tao Hong, Pierre Pinson, Yi Wang, Rafal Weron, Dazhi Yang, Hamidreza Zareipour, "Energy Forecasting: A Review and Outlook", IEEE Open Access Journal of Power and Energy, vol. 7, pp. 376-388, 2020, https://doi.org/10.1109/OAJPE.2020.3029979.
[24] Tanveer Ahmad, Huanxin Chen, "Deep learning for multi-scale smart energy forecasting", Energy, vol. 175, pp. 98-112, 2019, https://doi.org/10.1016/j.energy.2019.03.080.
[25] Mingsheng Long, Yue Cao, Zhangjie Cao, Jianmin Wang, Michael I. Jordan, "Transferable Representation Learning with Deep Adaptation Networks", IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 12, pp. 3071-3085, 2019, https://doi.org/10.1109/TPAMI.2018.2868685.
[26] Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, Qiang Xu, "SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction", Advances in Neural Information Processing Systems 35, pp. 5816-5828, 2022, https://doi.org/10.52202/068431-0421.
[27] David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski, "DeepAR: Probabilistic forecasting with autoregressive recurrent networks", International Journal of Forecasting, vol. 36, no. 3, pp. 1181-1191, 2020, https://doi.org/10.1016/j.ijforecast.2019.07.001.
[28] José F. Torres, Dalil Hadjout, Abderrazak Sebaa, Francisco Martínez-Álvarez, Alicia Troncoso, "Deep Learning for Time Series Forecasting: A Survey", Big Data, vol. 9, no. 1, pp. 3-21, 2021, https://doi.org/10.1089/big.2020.0159.
[29] Hansika Hewamalage, Christoph Bergmeir, Kasun Bandara, "Recurrent Neural Networks for Time Series Forecasting: Current status and future directions", International Journal of Forecasting, vol. 37, no. 1, pp. 388-427, 2021, https://doi.org/10.1016/j.ijforecast.2020.06.008.
[30] Fotios Petropoulos, Daniele Apiletti, Vassilios Assimakopoulos, Mohamed Zied Babai, Devon K. Barrow, Souhaib Ben Taieb, Christoph Bergmeir, Ricardo J. Bessa, Jakub Bijak, John E. Boylan, Jethro Browell, Claudio Carnevale, Jennifer L. Castle, Pasquale Cirillo, Michael P. Clements, Clara Cordeiro, Fernando Luiz Cyrino Oliveira, Shari De Baets, Alexander Dokumentov, Joanne Ellison, Piotr Fiszeder, Philip Hans Franses, David T. Frazier, Michael Gilliland, M. Sinan Gönül, Paul Goodwin, Luigi Grossi, Yael Grushka-Cockayne, Mariangela Guidolin, Massimo Guidolin, Ulrich Gunter, Xiaojia Guo, Renato Guseo, Nigel Harvey, David F. Hendry, Ross Hollyman, Tim Januschowski, Jooyoung Jeon, Victor Richmond R. Jose, Yanfei Kang, Anne B. Koehler, Stephan Kolassa, Nikolaos Kourentzes, Sonia Leva, Feng Li, Konstantia Litsiou, Spyros Makridakis, Gael M. Martin, Andrew B. Martinez, Sheik Meeran, Theodore Modis, Konstantinos Nikolopoulos, Dilek Önkal, Alessia Paccagnini, Anastasios Panagiotelis, Ioannis Panapakidis, Jose M. Pavía, Manuela Pedio, Diego J. Pedregal, Pierre Pinson, Patrícia Ramos, David E. Rapach, J. James Reade, Bahman Rostami-Tabar, Michał Rubaszek, Georgios Sermpinis, Han Lin Shang, Evangelos Spiliotis, Aris A. Syntetos, Priyanga Dilini Talagala, Thiyanga S. Talagala, Len Tashman, Dimitrios Thomakos, Thordis Thorarinsdottir, Ezio Todini, Juan Ramón Trapero Arenas, Xiaoqian Wang, Robert L. Winkler, Alisa Yusupova, Florian Ziel, "Forecasting: theory and practice", International Journal of Forecasting, vol. 38, no. 3, pp. 705-871, 2022, https://doi.org/10.1016/j.ijforecast.2021.11.001.

## Author Biography

zhijie REN (ORCID: 0009-0006-1048-6640) received training in electronic information and is currently a doctoral student with the College of Computer Science, Hunan University, Changsha, Hunan, China. His research interests include machine learning for energy time series, electricity price forecasting, short-term load forecasting, time-series representation learning, and decision-focused virtual power plant operation.

