"""Shared data layer for the 2026 revisit.

Every revisit script loads WONDER exports and county geometry through this
module, so one sample rule and one contiguity graph apply everywhere.

Two facts about the original exports drive the design:

* WONDER omits counties with fewer than 10 deaths in the window (suppressed).
* The 1999–2020 database blanks the *published* crude rate for counties with
  10–19 deaths ("Unreliable"), while the 2018–2024 export printed it. The
  original scripts read the published rate, so the historical windows silently
  dropped every 10–19-death county and 2018–2024 kept them. Recomputing the
  rate from Deaths / Population restores one rule for every window.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"
FIGURES = ROOT / "figures" / "revisit"
OUT = ROOT / "out" / "revisit"

TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}

# The four windows of the April analysis. 2013–2019 and 2018–2024 overlap by
# two years (different WONDER databases); kept only to reproduce and compare.
APRIL_WINDOWS = ["1999_2005", "2006_2012", "2013_2019", "2018_2024"]
CAUSES = {"suicide": "suicide", "overdose": "overdose", "k70": "alcohol_liver"}


def county_shapes() -> gpd.GeoDataFrame:
    """Contiguous-US counties (TIGER 2023), EPSG:5070. Downloads once, then caches."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "tl_2023_us_county.zip"
    if not path.exists():
        print(f"  downloading {TIGER_URL} (83 MB, once) ...")
        urllib.request.urlretrieve(TIGER_URL, path)
    g = gpd.read_file(path).rename(columns={"GEOID": "fips"})
    g = g[~g["STATEFP"].isin(NON_CONUS)].to_crs("EPSG:5070")
    return g[["fips", "STATEFP", "NAME", "geometry"]].sort_values("fips").reset_index(drop=True)


def read_wonder(path: Path | str) -> pd.DataFrame:
    """One WONDER county export -> fips, deaths, population, rate (recomputed), published_rate."""
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.strip('"')
    df = df[df["County Code"].str.match(r"^\d{5}$", na=False)].copy()
    out = pd.DataFrame({
        "fips": df["County Code"].to_numpy(),
        "deaths": pd.to_numeric(df["Deaths"], errors="coerce").to_numpy(),
        "population": pd.to_numeric(df["Population"], errors="coerce").to_numpy(),
        "published_rate": pd.to_numeric(df["Crude Rate"], errors="coerce").to_numpy(),
    })
    out["rate"] = 1e5 * out["deaths"] / out["population"]
    return out.dropna(subset=["deaths", "population"])


def load_window(window: str) -> dict[str, pd.DataFrame]:
    """All three causes for one April window, keyed suicide / overdose / k70."""
    return {k: read_wonder(DATA / f"{stem}_county_{window}.csv") for k, stem in CAUSES.items()}


def queen(gdf: gpd.GeoDataFrame) -> libpysal.weights.W:
    """Row-standardised Queen contiguity on exactly these rows (ids = row order)."""
    w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    w.transform = "r"
    return w
