"""External inputs that are fetched, not committed, each pinned by SHA-256.

`path_for(name)` downloads a source into data/cache/ on first use and refuses
to continue if the bytes differ from the recorded hash, so a changed upstream
file fails loudly instead of silently shifting the results.

Run `python fetch.py` to download everything up front and print the hashes.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
POPEST = "https://www2.census.gov/programs-surveys/popest/datasets"

SOURCES = {
    # Census TIGER/Line 2023 county boundaries (public domain)
    "tiger_2023_county": (
        "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip",
        "tl_2023_us_county.zip",
        "692e12c30c83adcaabdbac0d3954fafa55e1c89a24b36d95e72e02dff938652e",
    ),
    # Census county population by five-year age group (public domain)
    "agesex_2000_2010": (f"{POPEST}/2000-2010/intercensal/county/co-est00int-agesex-5yr.csv", "co-est00int-agesex-5yr.csv", "09077952bd388f1df05b139aa594e599e40095c85266550254a93d107501120d"),
    "agesex_2010_2020": (f"{POPEST}/2010-2020/intercensal/county/asrh/cc-est2020int-agesex-all.csv", "cc-est2020int-agesex-all.csv", "c32ec8c76125c795946066da3b371d9a67c7afe83765cb56bf4557ca49797a88"),
    "agesex_2020_2024": (f"{POPEST}/2020-2024/counties/asrh/cc-est2024-agesex-all.csv", "cc-est2024-agesex-all.csv", "fecfb886971c4a3725ed518bfc7e31a0dacbb01c6d815fa5226f1c7c8849120e"),
    # Census county population by race and Hispanic origin (public domain)
    "alldata_2020_2024": (f"{POPEST}/2020-2024/counties/asrh/cc-est2024-alldata.csv", "cc-est2024-alldata.csv", "9f772747fe3c48cce8ea72b3a61f515448fd276bc5ba6abce2c75c98fa6b10b4"),
    # Washington Post ARCOS county-year pill shipments, 2006-2014 (ODbL: fetched, never committed)
    "arcos_county_annual": (
        "https://raw.githubusercontent.com/wpinvestigative/arcos-api/3a8bc220c3821b8a8ec123d14b96f1223b61ad79/data/county_annual14.csv",
        "arcos_county_annual14.csv",
        "9344072793edb7dbe7bf5bb90b43ab178eb0566e0d0aa9094584d57489929045",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path_for(name: str) -> Path:
    url, filename, expected = SOURCES[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / filename
    if not path.exists():
        print(f"  fetching {name}: {url}", file=sys.stderr)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    got = sha256(path)
    if expected and got != expected:
        raise RuntimeError(f"{name}: {path.name} has SHA-256 {got}, expected {expected}. "
                           "The upstream file changed; see revisit/fetch.py.")
    return path


if __name__ == "__main__":
    for name in SOURCES:
        p = path_for(name)
        print(f"{name:<22} {sha256(p)}  {p.stat().st_size:>11,}  {p.name}")
