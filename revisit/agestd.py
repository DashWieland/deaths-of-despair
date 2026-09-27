"""Indirect age standardisation for county rates, identical in every window.

WONDER gives direct age-adjusted county rates only in the 1999-2020 database;
the 2018-2024 single-race database has none at county level. So every window
here uses indirect standardisation instead:

    expected_c = sum over age groups g of  county person-years_{c,g} x national rate_g
    SMR_c      = observed deaths_c / expected_c
    ISR_c      = SMR_c x national crude rate        (per 100,000)

National age-specific rates come from WONDER (data/wonder/national/age_*.tsv),
pooled over the same years as the window; county person-years by age come from
the Census intercensal / postcensal estimates (revisit/fetch.py). Where WONDER
checks are possible (D76 windows) the ISR is compared against WONDER's own
direct age-adjusted rate in r04.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from fetch import path_for
from wonder import FIPS_CROSSWALK, MERGE_INTO, WONDER, read_export, years

AGE10 = ["0-4", "5-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"]
WONDER_AGE = {"1": "0-4", "1-4": "0-4", "5-14": "5-14", "15-24": "15-24", "25-34": "25-34",
              "35-44": "35-44", "45-54": "45-54", "55-64": "55-64", "65-74": "65-74",
              "75-84": "75-84", "85+": "85+"}  # "NS" (age not stated) is dropped
FIVE_TO_TEN = {"AGE04": "0-4", "AGE59": "5-14", "AGE1014": "5-14", "AGE1519": "15-24", "AGE2024": "15-24",
               "AGE2529": "25-34", "AGE3034": "25-34", "AGE3539": "35-44", "AGE4044": "35-44",
               "AGE4549": "45-54", "AGE5054": "45-54", "AGE5559": "55-64", "AGE6064": "55-64",
               "AGE6569": "65-74", "AGE7074": "65-74", "AGE7579": "75-84", "AGE8084": "75-84",
               "AGE85PLUS": "85+"}
AGEGRP_TO_TEN = {1: "0-4", 2: "5-14", 3: "5-14", 4: "15-24", 5: "15-24", 6: "25-34", 7: "25-34",
                 8: "35-44", 9: "35-44", 10: "45-54", 11: "45-54", 12: "55-64", 13: "55-64",
                 14: "65-74", 15: "65-74", 16: "75-84", 17: "75-84", 18: "85+"}

_X = lambda *codes: {c for c in codes}
CAUSES = {
    "suicide": lambda c: "X60" <= c[:3] <= "X84",
    "overdose": lambda c: c[:3] in _X("X40", "X41", "X42", "X43", "X44", "Y10", "Y11", "Y12", "Y13", "Y14"),
    "k70": lambda c: c[:3] == "K70",
    "cd_poisoning": lambda c: c[:3] in _X("X40", "X41", "X42", "X43", "X44", "X45", "Y10", "Y11", "Y12",
                                          "Y13", "Y14", "Y15", "Y45", "Y47", "Y49"),
    "cd_liver": lambda c: c[:3] in _X("K70", "K73", "K74"),
}


def _database_file(window: str) -> str:
    return "age_d76.tsv" if years(window)[-1] <= 2018 else "age_d158.tsv"


@lru_cache(maxsize=None)
def _national_table(fname: str) -> pd.DataFrame:
    df = read_export(WONDER / "national" / fname)
    df = df[df["Ten-Year Age Groups Code"].isin(WONDER_AGE)].copy()
    df["age10"] = df["Ten-Year Age Groups Code"].map(WONDER_AGE)
    df["year"] = df["Year Code"].astype(int)
    return df


def national_rates(cause: str, window: str) -> pd.DataFrame:
    """Per ten-year age group: national deaths, person-years, rate, and suppressed cells."""
    df = _national_table(_database_file(window))
    df = df[df.year.isin(years(window))]
    pop = (df.drop_duplicates(["year", "Ten-Year Age Groups Code"])
             .groupby("age10")["Population"].sum())
    mine = df[df["Cause of death Code"].map(CAUSES[cause])]
    deaths = mine.groupby("age10")["Deaths"].sum(min_count=0)
    supp = mine.groupby("age10")["suppressed"].sum()
    out = pd.DataFrame({"person_years": pop}).join(deaths.rename("deaths")).join(supp.rename("suppressed_cells"))
    out = out.reindex(AGE10).fillna({"deaths": 0, "suppressed_cells": 0})
    out["rate"] = out.deaths / out.person_years
    return out


@lru_cache(maxsize=None)
def _census_age() -> pd.DataFrame:
    """County person-counts by ten-year age group and calendar year, 2000-2024 (long)."""
    frames = []
    # 2000-2009, five-year AGEGRP rows, both sexes
    a = pd.read_csv(path_for("agesex_2000_2010"), dtype={"STATE": str, "COUNTY": str}, encoding="latin-1")
    a = a[(a.SEX == 0) & (a.AGEGRP >= 1) & (a.SUMLEV == 50)]
    a["fips"] = a.STATE.str.zfill(2) + a.COUNTY.str.zfill(3)
    a["age10"] = a.AGEGRP.map(AGEGRP_TO_TEN)
    for y in range(2000, 2010):
        frames.append(a.groupby(["fips", "age10"])[f"POPESTIMATE{y}"].sum().rename("pop").reset_index().assign(year=y))
    # 2010-2020 intercensal (YEAR 2..12 = July 2010..2020) and 2020-2024 postcensal
    # (YEAR 2..6 = July 2020..2024). The postcensal file wins for 2020; the
    # intercensal 2020 row survives only for counties it lacks (Connecticut's
    # old counties, replaced by planning regions in the postcensal file).
    for name, first, last, offset in [("agesex_2010_2020", 2010, 2020, 2008), ("agesex_2020_2024", 2020, 2024, 2018)]:
        b = pd.read_csv(path_for(name), dtype={"STATE": str, "COUNTY": str}, encoding="latin-1")
        b["fips"] = b.STATE.str.zfill(2) + b.COUNTY.str.zfill(3)
        b["year"] = b.YEAR + offset
        b = b[b.year.between(first, last) & (b.YEAR >= 2)]
        long = b.melt(id_vars=["fips", "year"], value_vars=[f"{k}_TOT" for k in FIVE_TO_TEN], var_name="band", value_name="pop")
        long["age10"] = long.band.str.replace("_TOT", "", regex=False).map(FIVE_TO_TEN)
        frames.append(long.groupby(["fips", "year", "age10"])["pop"].sum().reset_index().assign(src=name))
    df = pd.concat(frames, ignore_index=True)
    df["fips"] = df.fips.replace(FIPS_CROSSWALK).replace(MERGE_INTO)
    post2020 = set(df.loc[df.src == "agesex_2020_2024", "fips"])
    df = df[~((df.src == "agesex_2010_2020") & (df.year == 2020) & df.fips.isin(post2020))]
    df = df.groupby(["fips", "year", "age10"])["pop"].sum()
    # Carry each county's latest age mix forward over years its vintage lacks
    # (Connecticut after 2020); 1999 borrows 2000.
    wide = df.unstack(["year"]).reindex(columns=range(2000, 2025))
    return wide.T.ffill().T.stack().rename("pop").reset_index()


def county_age_structure(window: str) -> pd.DataFrame:
    """Person-years by county x ten-year age group for the window (1999 uses the 2000 estimates)."""
    df = _census_age()
    ys = [max(y, 2000) for y in years(window)]
    parts = [df[df.year == y] for y in ys]
    return (pd.concat(parts).groupby(["fips", "age10"])["pop"].sum()
              .unstack("age10").reindex(columns=AGE10).fillna(0))


def standardise(counties: pd.DataFrame, cause: str, window: str) -> pd.DataFrame:
    """Add expected, smr and isr columns to a wonder.county() frame."""
    nat = national_rates(cause, window)
    struct = county_age_structure(window)
    share = struct.div(struct.sum(axis=1), axis=0)            # each county's age mix
    expected_per_py = share.mul(nat.rate, axis=1).sum(axis=1)  # expected deaths per person-year
    crude = nat.deaths.sum() / nat.person_years.sum()
    out = counties.copy()
    out["expected"] = out.fips.map(expected_per_py) * out.population
    out["smr"] = out.deaths / out.expected
    out["isr"] = 1e5 * out.smr * crude
    return out
