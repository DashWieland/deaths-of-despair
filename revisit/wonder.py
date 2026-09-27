"""Readers for the 2026 CDC WONDER exports in data/wonder/ (see QUERIES.md).

Every export is WONDER's tab-separated file with its footnotes attached. The
readers strip the footnotes, keep counts and populations as numbers, and flag
suppressed rows (1-9 deaths) instead of dropping them, because the difference
between "suppressed" and "absent" (zero deaths) matters downstream.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

WONDER = Path(__file__).resolve().parents[1] / "data" / "wonder"

# Non-overlapping three-year windows (1999-2000 is two years); the database
# switches from bridged race (D76) to single race (D158) at 2019.
WINDOWS = ["1999-2000", "2001-2003", "2004-2006", "2007-2009", "2010-2012",
           "2013-2015", "2016-2018", "2019-2021", "2022-2024"]
COARSE = ["1999-2005", "2006-2012", "2013-2018", "2019-2024"]
PRIMARY = ["suicide", "overdose", "k70"]


def read_export(path: Path | str) -> pd.DataFrame:
    """A WONDER export as a DataFrame: numeric columns parsed, 'suppressed' flagged."""
    text = Path(path).read_text(encoding="utf-8")
    body = text.split('\n"---"', 1)[0]
    df = pd.read_csv(io.StringIO(body), sep="\t", dtype=str).drop(columns=["Notes"])
    df = df.dropna(how="all")
    df["suppressed"] = df["Deaths"].eq("Suppressed")
    for col in df.columns:
        if col in ("Deaths", "Population") or "Rate" in col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def county(cause: str, window: str, variant: str = "county") -> pd.DataFrame:
    """One county export -> fips, deaths (NaN if suppressed), suppressed, population, rate, aar."""
    df = read_export(WONDER / variant / f"{cause}_{window}.tsv")
    out = pd.DataFrame({
        "fips": df["County Code"].str.zfill(5),
        "deaths": df["Deaths"],
        "suppressed": df["suppressed"],
        "population": df["Population"],
    })
    out["rate"] = 1e5 * out["deaths"] / out["population"]
    out["aar_wonder"] = df["Age Adjusted Rate"] if "Age Adjusted Rate" in df else np.nan
    return out.reset_index(drop=True)


def years(window: str) -> list[int]:
    a, b = (int(x) for x in window.split("-"))
    return list(range(a, b + 1))
