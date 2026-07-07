# Public Data Download Templates

## opsd_time_series_60min_singleindex

URL template: `https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv`

Verified direct public download in this workspace on 2026-06-30. Contains hourly ENTSO-E load, day-ahead price, wind, and solar series used by the current reproducible OPSD baseline experiment.

## uci_electricity_load_diagrams

URL template: `https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip`

Official UCI archive for 370 electricity-consumption clients at 15-minute resolution. HEAD request returned HTTP 200 in this workspace; not downloaded in the current run because OPSD already covers the first public experiment.

## open_power_system_data_time_series_landing

URL template: `https://data.open-power-system-data.org/time_series/`

Official OPSD time-series landing directory. Use this for provenance and package-version verification.

## nyiso_realtime_zone_example

URL template: `http://mis.nyiso.com/public/csv/realtime/{yyyymmdd}realtime_zone.csv`

Real-time zone LBMP CSV archive. The local shell DNS failed for mis.nyiso.com during this session.

## nyiso_pal_example

URL template: `http://mis.nyiso.com/public/csv/pal/{yyyymmdd}pal.csv`

NYISO actual load CSV archive.

## pjm_data_miner

URL template: `https://api.pjm.com/api/v1/{endpoint}`

PJM Data Miner 2 API. Usually requires endpoint-specific parameters and sometimes a subscription key.

## caiso_oasis

URL template: `http://oasis.caiso.com/oasisapi/SingleZip?queryname={queryname}&startdatetime={start}&enddatetime={end}&version=1`

CAISO OASIS zipped CSV API; query names and parameters must be selected per dataset.

## aemo_nemweb

URL template: `https://nemweb.com.au/Reports/Current/`

AEMO NEMWeb public report folders. Use exact report subdirectories for dispatch price/demand data.
