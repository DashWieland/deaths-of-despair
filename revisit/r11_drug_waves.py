"""r11 — The drug waves, 1999-2024, and which of them matched the suicide map.

State x year exports from WONDER's multiple-cause databases (D77 for
1999-2017, D157 for 2018-2024; they overlap in 2018-20 and agree): deaths
from drug overdose (CDC definition) that involved each drug class, plus
state suicide, alcoholic liver disease and all-drug overdose.

1. National and regional rates by drug class and year: prescription-type
   opioids (T40.2), heroin (T40.1), synthetic opioids (T40.4, mostly
   fentanyl), methadone (T40.3), cocaine (T40.5), psychostimulants (T43.6).
   A death involving two drugs counts under both.
2. Year by year, the Spearman correlation across states between the
   age-adjusted suicide rate and each drug's age-adjusted death rate: an
   annual, coarse-geography companion to the county life cycle (r02).
   States with a suppressed or unreliable rate that year are left out.

Writes out/revisit/r11_waves.csv and out/revisit/r11_state_corr.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import OUT
from r09_recent_waves import FIPS_STATE, STATE_REGION
from wonder import WONDER, read_export

DRUGS = ["rx_opioids", "heroin", "synthetic_opioids", "methadone", "cocaine", "psychostimulants"]
CAUSES = ["suicide_ucd", "k70_ucd", "overdose_ucd"]


def state_series(name: str) -> pd.DataFrame:
    """One measure for 1999-2024: D77 through 2017, D157 from 2018."""
    parts = []
    for db, keep in [("d77", lambda y: y <= 2017), ("d157", lambda y: y >= 2018)]:
        path = WONDER / "state" / f"{name}_{db}.tsv"
        if not path.exists():
            continue
        d = read_export(path)
        d["year"] = d["Year Code"].astype(int)
        d = d[d.year.map(keep)]
        d["state"] = d["State Code"].astype(str).str.zfill(2).map(FIPS_STATE)
        parts.append(d[["state", "year", "Deaths", "Population", "Age Adjusted Rate", "suppressed"]])
    out = pd.concat(parts, ignore_index=True).dropna(subset=["state"])
    return out.rename(columns={"Deaths": "deaths", "Population": "population", "Age Adjusted Rate": "aar"})


def main():
    series = {n: state_series(n) for n in DRUGS + CAUSES
              if any((WONDER / "state" / f"{n}_{db}.tsv").exists() for db in ("d77", "d157"))}
    rows = []
    for name, d in series.items():
        d = d.assign(region=d.state.map(STATE_REGION))
        for (year, region), g in d.groupby(["year", "region"]):
            rows.append({"measure": name, "year": year, "region": region,
                         "rate": 1e5 * g.deaths.sum() / g.population.sum(), "suppressed_states": int(g.suppressed.sum())})
        for year, g in d.groupby("year"):
            rows.append({"measure": name, "year": year, "region": "US",
                         "rate": 1e5 * g.deaths.sum() / g.population.sum(), "suppressed_states": int(g.suppressed.sum())})
    waves = pd.DataFrame(rows)

    corr = []
    if "suicide_ucd" in series:
        s = series["suicide_ucd"].set_index(["state", "year"]).aar.rename("suicide")
        for name in [n for n in series if n != "suicide_ucd"]:
            x = series[name].set_index(["state", "year"]).aar.rename("x")
            m = pd.concat([s, x], axis=1, join="inner").dropna()
            for year, g in m.groupby(level="year"):
                if len(g) >= 20:
                    corr.append({"measure": name, "year": year, "states": len(g), "spearman": stats.spearmanr(g.suicide, g.x)[0]})
    corr = pd.DataFrame(corr)

    OUT.mkdir(parents=True, exist_ok=True)
    waves.to_csv(OUT / "r11_waves.csv", index=False)
    corr.to_csv(OUT / "r11_state_corr.csv", index=False)
    pd.set_option("display.width", 220)
    us = waves[waves.region == "US"].pivot(index="year", columns="measure", values="rate")
    print("US deaths per 100,000 by drug involved (crude, summed over states):")
    print(us[[c for c in DRUGS if c in us]].round(2).to_string())
    if len(corr):
        print("\nSpearman across states, suicide vs each measure (age-adjusted):")
        print(corr.pivot(index="year", columns="measure", values="spearman").round(2).to_string())


if __name__ == "__main__":
    main()
