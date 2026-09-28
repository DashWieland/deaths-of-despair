"""r07 — Does the life cycle hold under Case & Deaton's definitions and population?

Four coarse windows (1999-2005, 2006-2012, 2013-2018, 2019-2024), Spearman
correlations across counties with >= 10 deaths of both causes:

* primary:  suicide X60-X84, overdose X40-X44 + Y10-Y14, alcoholic liver K70,
            all ages (1999-2005 and 2006-2012 from the April seven-year
            exports; 2013-2018 and 2019-2024 pooled from the three-year
            windows, counties known in both halves)
* cd:       Case & Deaton's poisonings (adds alcohol poisoning and undetermined
            intent X45, Y15, Y45, Y47, Y49) and liver (adds K73-K74), with
            the primary suicide series
* age25_64: all three primary definitions, deaths and population ages 25-64

Writes out/revisit/r07_definitions.csv
"""
import pandas as pd
from scipy import stats

from agestd import _census_age
from common import DATA, OUT, read_wonder
from wonder import FIPS_CROSSWALK, WONDER, county, read_export, years

COARSE = ["1999-2005", "2006-2012", "2013-2018", "2019-2024"]
HALVES = {"2013-2018": ["2013-2015", "2016-2018"], "2019-2024": ["2019-2021", "2022-2024"]}
APRIL_STEM = {"suicide": "suicide", "overdose": "overdose", "k70": "alcohol_liver"}


def primary(cause: str, window: str) -> pd.Series:
    """Deaths and population for a coarse window -> rate, counties with >= 10 deaths."""
    if window in HALVES:
        a, b = (county(cause, w).set_index("fips") for w in HALVES[window])
        both = a[["deaths", "population"]].join(b[["deaths", "population"]], rsuffix="_b", how="inner").dropna()
        deaths, pop = both.deaths + both.deaths_b, both.population + both.population_b
    else:
        d = read_wonder(DATA / f"{APRIL_STEM[cause]}_county_{window.replace('-', '_')}.csv")
        d["fips"] = d.fips.replace(FIPS_CROSSWALK)
        d = d.set_index("fips")
        deaths, pop = d.deaths, d.population
    keep = deaths >= 10
    return (1e5 * deaths[keep] / pop[keep]).rename(cause)


def census_person_years(window: str, bands: list[str]) -> pd.Series:
    df = _census_age()
    ys = [max(y, 2000) for y in years(window)]
    return df[df.year.isin(ys) & df.age10.isin(bands)].groupby("fips")["pop"].sum()


def variant(cause: str, window: str, folder: str) -> pd.Series:
    path = WONDER / folder / f"{cause}_{window}.tsv"
    raw = read_export(path)
    if "Population" not in raw:
        # The 2018-2024 single-race database gives no county population for
        # ten-year age filters, only deaths. Denominators: Census person-years
        # for ages 25-64 over the same years (the source WONDER's come from).
        d = pd.DataFrame({"fips": raw["County Code"].str.zfill(5).replace(FIPS_CROSSWALK), "deaths": raw["Deaths"]})
        d["population"] = d.fips.map(census_person_years(window, ["25-34", "35-44", "45-54", "55-64"]))
        d = d.dropna().set_index("fips")
        d["rate"] = 1e5 * d.deaths / d.population
    else:
        d = county(cause, window, variant=folder).set_index("fips")
    d = d[d.deaths >= 10]
    return d.rate.rename(cause)


def rho(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    m = pd.concat([x, y], axis=1, join="inner").dropna()
    return len(m), stats.spearmanr(m.iloc[:, 0], m.iloc[:, 1])[0]


def main():
    rows = []
    for w in COARSE:
        s, o, k = primary("suicide", w), primary("overdose", w), primary("k70", w)
        cp, cl = variant("cd_poisoning", w, "county_cd"), variant("cd_liver", w, "county_cd")
        s2, o2, k2 = (variant(c, w, "county_25_64") for c in ["suicide", "overdose", "k70"])
        for label, x, y in [
            ("primary: suicide x overdose", s, o), ("primary: suicide x liver", s, k), ("primary: overdose x liver", o, k),
            ("C&D: suicide x poisonings", s, cp), ("C&D: suicide x liver", s, cl), ("C&D: poisonings x liver", cp, cl),
            ("ages 25-64: suicide x overdose", s2, o2), ("ages 25-64: suicide x liver", s2, k2), ("ages 25-64: overdose x liver", o2, k2),
        ]:
            n, r = rho(x, y)
            rows.append({"series": label, "window": w, "n": n, "spearman": r})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r07_definitions.csv", index=False)
    wide = df.pivot(index="series", columns="window", values="spearman")[COARSE]
    ns = df.pivot(index="series", columns="window", values="n")[COARSE]
    print(wide.round(3).to_string())
    print("\ncounties per cell:\n" + ns.to_string())


if __name__ == "__main__":
    main()
