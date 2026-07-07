"""
Public data download templates for the doctoral paper package.

These are intentionally lightweight templates. Some public market sources may
require API keys, archive-specific filenames, or network access unavailable in
the current shell. Do not treat this script as a finished downloader until each
source is verified in the target execution environment.
"""

from pathlib import Path
import argparse
import urllib.request
import shutil


OUT = Path(__file__).resolve().parents[1] / "public_data"
OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
DATA_RAW = OUT.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


SOURCES = {
    "opsd_time_series_60min_singleindex": {
        "url_template": "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv",
        "note": "Verified direct public download in this workspace on 2026-06-30. Contains hourly ENTSO-E load, day-ahead price, wind, and solar series used by the current reproducible OPSD baseline experiment.",
        "local_filename": "opsd_time_series_60min_singleindex.csv",
    },
    "uci_electricity_load_diagrams": {
        "url_template": "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        "note": "Official UCI archive for 370 electricity-consumption clients at 15-minute resolution. HEAD request returned HTTP 200 in this workspace; not downloaded in the current run because OPSD already covers the first public experiment.",
        "local_filename": "electricityloaddiagrams20112014.zip",
    },
    "open_power_system_data_time_series_landing": {
        "url_template": "https://data.open-power-system-data.org/time_series/",
        "note": "Official OPSD time-series landing directory. Use this for provenance and package-version verification.",
    },
    "nyiso_realtime_zone_example": {
        "url_template": "http://mis.nyiso.com/public/csv/realtime/{yyyymmdd}realtime_zone.csv",
        "note": "Real-time zone LBMP CSV archive. The local shell DNS failed for mis.nyiso.com during this session.",
    },
    "nyiso_pal_example": {
        "url_template": "http://mis.nyiso.com/public/csv/pal/{yyyymmdd}pal.csv",
        "note": "NYISO actual load CSV archive.",
    },
    "pjm_data_miner": {
        "url_template": "https://api.pjm.com/api/v1/{endpoint}",
        "note": "PJM Data Miner 2 API. Usually requires endpoint-specific parameters and sometimes a subscription key.",
    },
    "caiso_oasis": {
        "url_template": "http://oasis.caiso.com/oasisapi/SingleZip?queryname={queryname}&startdatetime={start}&enddatetime={end}&version=1",
        "note": "CAISO OASIS zipped CSV API; query names and parameters must be selected per dataset.",
    },
    "aemo_nemweb": {
        "url_template": "https://nemweb.com.au/Reports/Current/",
        "note": "AEMO NEMWeb public report folders. Use exact report subdirectories for dispatch price/demand data.",
    },
}


def download_url(url: str, out_path: Path):
    with urllib.request.urlopen(url, timeout=30) as response:
        out_path.write_bytes(response.read())


def write_source_notes():
    lines = ["# Public Data Download Templates", ""]
    for name, meta in SOURCES.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"URL template: `{meta['url_template']}`")
        lines.append("")
        lines.append(meta["note"])
        lines.append("")
    (OUT / "PUBLIC_DATA_SOURCE_NOTES.md").write_text("\n".join(lines), encoding="utf-8")


def download_named_source(name: str):
    meta = SOURCES[name]
    filename = meta.get("local_filename")
    if not filename:
        raise SystemExit(f"Source {name} is a template and has no direct local filename.")
    out_path = RAW / filename
    download_url(meta["url_template"], out_path)
    mirror_path = DATA_RAW / filename
    shutil.copy2(out_path, mirror_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", choices=sorted(SOURCES), help="Download a directly downloadable registered source.")
    args = parser.parse_args()
    write_source_notes()
    if args.download:
        print(download_named_source(args.download))
    print(OUT / "PUBLIC_DATA_SOURCE_NOTES.md")
