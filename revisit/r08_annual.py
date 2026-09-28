"""r08 — When did it turn? Annual timing from NCHS's model-based overdose rates.

NCHS publishes smoothed (hierarchical Bayesian) drug-poisoning death rates for
every county and year, 2003-2021, with no suppression (snapshot in data/cdc/).
Pairing each year's overdose estimate with the county's suicide rate from the
three-year window containing that year gives a near-annual curve of
co-location, to see when the decline began.

Caveats: the NCHS measure is all-intent drug poisoning (it includes drug
suicides, X60-X64), and it is modelled, so it is smoother than raw rates.

RESULT (negative): this cannot resolve annual timing. Within each three-year
block the model moves every county by a common factor (e.g. 2004 -> 2005 is
x1.078-1.079 for counties in AL, WV and NM alike), so county rankings, and
every rank correlation here, only change every three years. The output is
kept as a record of the check; the three-year WONDER windows remain the
finest reliable resolution.

Writes out/revisit/r08_annual.csv
"""
import pandas as pd
from scipy import stats

from common import DATA, OUT
from wonder import FIPS_CROSSWALK, WINDOWS, county, years


def main():
    path = sorted((DATA / "cdc").glob("nchs_county_drug_poisoning_model_*.csv"))[-1]
    nchs = pd.read_csv(path, dtype={"fips": str})
    nchs["fips"] = nchs.fips.str.zfill(5).replace(FIPS_CROSSWALK)
    nchs["year"] = nchs.year.astype(int)
    rate_col = "model_based_death_rate"
    nchs[rate_col] = pd.to_numeric(nchs[rate_col], errors="coerce")
    window_of = {y: w for w in WINDOWS for y in years(w)}
    suicide = {w: county("suicide", w)[lambda d: d.deaths >= 10].set_index("fips").rate for w in WINDOWS}
    k70 = {w: county("k70", w)[lambda d: d.deaths >= 10].set_index("fips").rate for w in WINDOWS}
    rows = []
    for y in sorted(nchs.year.unique()):
        od = nchs[nchs.year == y].set_index("fips")[rate_col]
        w = window_of[y]
        for label, other in [("suicide", suicide[w]), ("k70", k70[w])]:
            m = pd.concat([od.rename("od"), other.rename("x")], axis=1, join="inner").dropna()
            rows.append({"year": y, "window": w, "pair": f"{label} x overdose(model)", "n": len(m),
                         "spearman": stats.spearmanr(m.od, m.x)[0]})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r08_annual.csv", index=False)
    print(df.pivot(index="year", columns="pair", values="spearman").round(3).to_string())


if __name__ == "__main__":
    main()
