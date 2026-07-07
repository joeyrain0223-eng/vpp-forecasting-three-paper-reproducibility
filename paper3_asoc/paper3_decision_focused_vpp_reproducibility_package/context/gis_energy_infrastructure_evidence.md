# GIS Energy-Infrastructure Evidence Audit

Generated: 2026-07-06
Status: PASS_WITH_LICENSE_AND_SOURCE_PAGE_GATES

GIS energy-infrastructure evidence layer for dissertation context and bounded journal external-validity discussion.

## Summary

- 2025 transmission-line records: 12839
- 2025 substation records: 2041
- 2025 grid-link records: 9444
- OSM mainland power records profiled: 4870459
- WRI China power plant records: 4274
- GEM operating facility capacity represented: 3108.905 GW
- 2015-2025 transmission-line record growth: 610.122%
- OSM generator renewable/storage source-tag share: 98.832%
- WRI renewable capacity share: 25.407%

## China Grid Time-Series Snapshot

|year|feature|rows|columns|voltage_non_null|max_voltage_kv|median_voltage_kv|share_500kv_plus|top_status|
|---|---|---|---|---|---|---|---|---|
|2015|01_power_transmission_lines|1808|30|1808|1000.0|220.0|0.2129|operating:1146; missing:172; cancelled:144; retired:143|
|2015|02_substations|1824|219|1824|1100.0|220.0|0.2171|operating:1192; missing:188; retired:156; pre-construction:106|
|2015|04_grid_links|4401|9|0|||||
|2020|01_power_transmission_lines|4921|30|4921|1000.0|220.0|0.154|operating:2734; missing:945; pre-construction:338; cancelled:289|
|2020|02_substations|4281|219|4281|1100.0|220.0|0.1495|operating:2310; missing:981; pre-construction:300; retired:221|
|2020|04_grid_links|7165|9|0|||||
|2025|01_power_transmission_lines|12839|30|12839|1100.0|220.0|0.165|operating:4757; missing:4633; pre-construction:1106; construction:696|
|2025|02_substations|2041|221|2041|1100.0|220.0|0.1411|missing:816; operating:758; pre-construction:181; construction:100|
|2025|04_grid_links|9444|9|0|||||

![Figure. Open GIS evidence for network-scale VPP decision context.](./outputs/[run-id]/figures/paper3_fig14_china_grid_gis_externality.png)

## OSM Mainland Power Extraction

|dataset|rows|columns|key_distribution|voltage_non_null_share|
|---|---|---|---|---|
|power_lines|216700|11||0.6836|
|power_points_generator|194613|13|wind:162400; solar:29322; unknown:1187; coal:920; hydro:548; gas:88; battery:70; nuclear:35||
|power_points_other|27398|13||0.0136|
|power_points_plant|43|13||0.0|
|power_points_pole|157896|13||0.0003|
|power_points_substation|247|13||0.3765|
|power_points_tower|3645674|13||0.0002|
|power_points_transformer|15216|13||0.0024|
|power_polygons|612672|11||0.0274|

## WRI China Power Plants

|fuel|plant_count|capacity_gw|
|---|---|---|
|Coal|953|969.908|
|Hydro|956|262.851|
|Gas|179|71.448|
|Solar|1321|54.916|
|Wind|842|51.185|
|Nuclear|15|38.616|
|Oil|6|3.329|
|Geothermal|2|0.026|
|__metadata__|4274||

## GEM Integrated Power Tracker

|type|status|rows|capacity_gw|missing_capacity|exact_or_approx_location_share|
|---|---|---|---|---|---|
|coal|operating|3373|1258.131|0|1.0|
|utility-scale solar|operating|14037|619.74|0|1.0|
|coal|cancelled|1010|585.237|0|1.0|
|wind|operating|6063|547.29|0|1.0|
|wind|pre-construction|2394|377.11|0|1.0|
|hydropower|operating|1027|376.969|0|1.0|
|utility-scale solar|pre-construction|1934|328.538|0|1.0|
|hydropower|announced|211|264.987|0|1.0|
|utility-scale solar|construction|1102|249.822|0|1.0|
|hydropower|construction|151|247.432|0|1.0|
|utility-scale solar|announced|697|238.343|0|1.0|
|wind|construction|1023|214.522|0|1.0|
|oil/gas|operating|818|209.803|0|1.0|
|coal|construction|335|206.361|0|1.0|
|nuclear|cancelled - inferred 4 y|185|201.41|0|1.0|
|wind|announced|676|174.878|0|1.0|
|coal|announced|221|165.528|0|1.0|
|coal|retired|1227|142.782|0|1.0|

![Figure. China resource-mix GIS metadata for thesis context.](./outputs/[run-id]/figures/dissertation_fig_gis_resource_mix_context.png)

## Derived Spatiotemporal and Resource-Heterogeneity Metrics

|metric_group|metric|value|unit|interpretation|
|---|---|---|---|---|
|grid_temporal_drift|transmission_line_records_2015_to_2025_growth_pct|610.122|percent|Infrastructure-context drift across the available China grid snapshots; motivates rolling validation and time-aware graph construction.|
|grid_temporal_drift|transmission_line_records_2025_share_500kv_plus|16.5|percent|High-voltage coverage proxy; supports the claim that VPP decisions may face network-scale coupling rather than isolated device control.|
|grid_temporal_drift|substation_records_2015_to_2025_growth_pct|11.897|percent|Infrastructure-context drift across the available China grid snapshots; motivates rolling validation and time-aware graph construction.|
|grid_temporal_drift|substation_records_2025_share_500kv_plus|14.11|percent|High-voltage coverage proxy; supports the claim that VPP decisions may face network-scale coupling rather than isolated device control.|
|grid_temporal_drift|grid_link_records_2015_to_2025_growth_pct|114.588|percent|Infrastructure-context drift across the available China grid snapshots; motivates rolling validation and time-aware graph construction.|
|osm_generator_mix|osm_generator_points_count|194613|records|Generator-point context extracted from OSM-derived mainland power-grid records; used only as coverage/context evidence.|
|osm_generator_mix|osm_generator_renewable_or_storage_share|98.832|percent|Resource-mix heterogeneity proxy; motivates load-transfer and VPP resource-state interfaces.|
|osm_generator_mix|osm_generator_source_entropy|0.517|nats|Diversity of generator source tags; not a reconciled capacity statistic and not used for model training.|
|wri_capacity_mix|wri_capacity_total_gw|1452.279|GW|Open plant-capacity context from WRI; provides resource-scale background only.|
|wri_capacity_mix|wri_renewable_capacity_share|25.407|percent|Capacity-mix heterogeneity proxy across fuel classes.|
|wri_capacity_mix|wri_capacity_mix_entropy|1.0795|nats|Capacity-weighted resource diversity; supports scenario-design discussion without becoming a training label.|
|gem_operating_mix|gem_operating_capacity_gw|3108.905|GW|Operating-capacity context from the local China GEM integrated-power table.|
|gem_operating_mix|gem_variable_or_dispatchable_context_share|56.412|percent|Broad resource-context share used to motivate VPP state heterogeneity; not a dispatchable-capacity estimate.|
|gem_operating_mix|gem_operating_type_entropy|1.5599|nats|Operating resource-type diversity proxy for scenario realism and source-domain heterogeneity.|

![Figure. GIS-derived spatiotemporal and resource heterogeneity metrics.](./outputs/[run-id]/figures/dissertation_fig_gis_spatiotemporal_resource_heterogeneity.png)

## Paper Integration Plan

|paper|use|
|---|---|
|Paper 1 price forecasting|Graph-prior and regional heterogeneity motivation only; do not turn GIS metadata into a price target.|
|Paper 2 load forecasting|Resource and network heterogeneity motivation for cross-domain load transfer; keep quantitative claims on UCI/OPSD/Shandong validated load data.|
|Paper 3 VPP decision-focused bidding|External-validity evidence that VPP decision policies operate in a large resource-network context with renewables, substations, transmission lines, and storage-adjacent assets.|
|Doctoral dissertation|Chapter 2 infrastructure background, Chapter 5 VPP scenario design, and Chapter 6 data-governance appendix.|

## Hard Boundaries

- The GIS layer is not used to train the public OPSD/UCI journal baselines in the current package.
- OSM-derived records require ODbL attribution/share-alike review before redistribution.
- GEM materials may carry non-commercial or tracker-specific use caveats; keep raw GEM files out of public supplements until confirmed.
- Figshare/WRI-style open-license sources can be cited with attribution, but exact source pages and versions still need final manuscript reference checks.

## Recommended Manuscript Wording

The GIS layer should be described as infrastructure-context evidence: it documents large-scale network, generation, and resource heterogeneity that motivates graph-aware forecasting and decision-focused VPP evaluation. It should not be described as a new model-training dataset for the current public OPSD/UCI experiments.
