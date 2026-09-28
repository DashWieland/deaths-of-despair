"""r05 — The prescription-pill supply and the geography of the three causes.

Pills = oxycodone + hydrocodone dosage units shipped to pharmacies and
practitioners in each county (DEA ARCOS via The Washington Post, 2006-2014),
per resident per year over 2006-2012. Everything below is an association
across counties; none of it identifies cause.

1. Pills vs each cause, window by window: does the county geography of each
   cause track where the pills went, and does that change after the pill era?
2. Suicide x overdose with pills held constant: in each window, the Spearman
   correlation of suicide and overdose rates, raw and partialling out pills.
   If the shared geography ran through the pill supply, the partial should
   fall well below the raw correlation in the pill era.
3. Did pills go where despair already was? Case & Deaton (2022) read the
   supply story as pharma targeting places already in pain. Correlate
   2006-2008 pills with each cause in the windows before 2006.

Credit: Source: The Washington Post, DEA ARCOS (ODbL; fetched, never committed).
Writes out/revisit/r05_supply.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from agestd import _census_age
from common import OUT
from fetch import path_for
from r03_aian import partial_spearman
from wonder import FIPS_CROSSWALK, MERGE_INTO, PRIMARY, WINDOWS, county


def pills_per_capita(first: int, last: int) -> pd.Series:
    a = pd.read_csv(path_for("arcos_county_annual"), dtype={"countyfips": str})
    a = a[a.year.between(first, last) & a.countyfips.notna()]
    a["fips"] = a.countyfips.str.zfill(5).replace(FIPS_CROSSWALK).replace(MERGE_INTO)
    pills = a.groupby("fips").DOSAGE_UNIT.sum()
    pop = _census_age()
    person_years = pop[pop.year.between(first, last)].groupby("fips")["pop"].sum()
    return (pills / person_years).dropna().rename("pills")


def main():
    pills = pills_per_capita(2006, 2012)
    early = pills_per_capita(2006, 2008).rename("pills_early")
    print(f"pills per resident per year, 2006-12: median {pills.median():.1f}, "
          f"90th pct {pills.quantile(.9):.1f}, max {pills.max():.1f} across {len(pills)} counties")
    rows = []
    for w in WINDOWS:
        rates = {c: county(c, w)[lambda d: d.deaths >= 10].set_index("fips")["rate"] for c in PRIMARY}
        for c in PRIMARY:
            m = pd.concat([rates[c], pills, early], axis=1, join="inner").dropna()
            rows.append({"analysis": "pills_vs_cause", "window": w, "cause": c, "n": len(m),
                         "rho_pills_2006_12": stats.spearmanr(m.iloc[:, 0], m.pills)[0],
                         "rho_pills_2006_08": stats.spearmanr(m.iloc[:, 0], m.pills_early)[0]})
        m = pd.concat([rates["suicide"].rename("s"), rates["overdose"].rename("o"), pills], axis=1, join="inner").dropna()
        raw = stats.spearmanr(m.s, m.o)[0]
        part = partial_spearman(m.s.to_numpy(), m.o.to_numpy(), m.pills.to_numpy())
        rows.append({"analysis": "suicide_x_overdose_given_pills", "window": w, "n": len(m),
                     "rho_raw": raw, "rho_partial_pills": part})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r05_supply.csv", index=False)

    p = df[df.analysis == "pills_vs_cause"].pivot(index="window", columns="cause", values="rho_pills_2006_12")
    print("\nSpearman: 2006-12 pills per resident vs each cause's rate (counties with >= 10 deaths)")
    print(p[PRIMARY].round(3).to_string())
    q = df[df.analysis == "suicide_x_overdose_given_pills"].set_index("window")[["n", "rho_raw", "rho_partial_pills"]]
    print("\nSuicide x overdose, raw and with pills held constant")
    print(q.round(3).to_string())


if __name__ == "__main__":
    main()
