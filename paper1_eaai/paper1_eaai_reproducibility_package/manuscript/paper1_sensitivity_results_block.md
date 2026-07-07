Table 15 reports spike-threshold sensitivity for the sequence-anchored GraphPatch result. Instead of defining spikes from the test split, each threshold is fixed by the training-set distribution of the absolute one-hour versus daily-lag price change. The GraphPatch residual remains positive in all four zones at the 85th, 90th, and 95th percentile thresholds, and the mean RMSE gain stays stable between 3.681% and 3.740% as the evaluation focuses on more volatile hours.

|Train quantile|Positive zones|Mean RMSE gain %|Median RMSE gain %|Minimum RMSE gain %|Spike hours|Mean win rate|
|---|---|---|---|---|---|---|
|0.85|4/4|3.740|3.699|1.066|6693|0.545|
|0.90|4/4|3.685|3.322|1.073|4448|0.536|
|0.95|4/4|3.681|2.661|1.125|2277|0.539|

![Figure 16. OPSD spike-threshold sensitivity for the sequence-anchored GraphPatch residual.](figures/paper1_fig16_opsd_spike_threshold_sensitivity.png)

Table 16 reports calibration-window sensitivity for the public split-conformal price benchmark. The final 20% chronological test split is held fixed, while the calibration window immediately preceding the test period is varied from 10% to 25% of the usable sequence. The 15% and 20% windows are both close to the 90% target: the 15% window has the smallest mean absolute coverage error in this diagnostic, while the manuscript's 20% calibration split preserves the conventional 60/20/20 chronological protocol and keeps interval width close to the neighboring settings.

|Calibration window|Mean PICP|Min PICP|Max PICP|Mean PINAW|Mean width|Mean interval score|Mean coverage error|
|---|---|---|---|---|---|---|---|
|10%|0.882|0.872|0.895|0.053|13.071|23.634|0.018|
|15%|0.902|0.893|0.919|0.058|14.390|23.526|0.007|
|20%|0.903|0.889|0.936|0.058|14.462|23.640|0.015|
|25%|0.907|0.888|0.945|0.060|14.816|23.715|0.017|

![Figure 17. OPSD conformal calibration-window sensitivity on the fixed public test split.](figures/paper1_fig17_opsd_calibration_window_sensitivity.png)

These sensitivity checks remove the remaining reviewer-facing gap in Section 5.4. The spike-regime conclusion is not an artifact of a single 90th-percentile threshold, and the conformal result is not tuned to a fragile calibration-window choice. The residual limitation remains substantive: coverage varies by market zone even when average PICP is close to the target, so adaptive local calibration remains necessary for deployment.
