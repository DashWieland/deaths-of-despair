"""r09 — The fentanyl era by region, 2015-2025 (CDC provisional counts).

VSRR provisional drug overdose deaths (snapshot in data/cdc/): the December
"12 month-ending" value is the calendar-year count; the predicted value
(adjusted for records still pending investigation) is used where present.
New York City (YC) is folded into New York. Rates use Census July population
estimates summed from the county files (2025 borrows 2024).

Per Census region and year: all-drug overdose rate, and rates for synthetic
opioids (T40.4, mostly fentanyl), heroin (T40.1), prescription-type opioids
(T40.2), cocaine (T40.5) and psychostimulants (T43.6); plus the West's share
of all US overdose deaths. Descriptive only.

Writes out/revisit/r09_recent_waves.csv
"""
import pandas as pd

from agestd import _census_age
from common import DATA, OUT

REGION = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_REGION = {s: r for r, ss in REGION.items() for s in ss.split()}
FIPS_STATE = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
              "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
              "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
              "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
              "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
              "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}
INDICATORS = {
    "Number of Drug Overdose Deaths": "all_drugs",
    "Synthetic opioids, excl. methadone (T40.4)": "synthetic_opioids",
    "Heroin (T40.1)": "heroin",
    "Natural & semi-synthetic opioids (T40.2)": "rx_opioids",
    "Cocaine (T40.5)": "cocaine",
    "Psychostimulants with abuse potential (T43.6)": "psychostimulants",
}


def main():
    path = sorted((DATA / "cdc").glob("vsrr_provisional_overdose_*.csv"))[-1]
    v = pd.read_csv(path, dtype=str)
    v = v[(v.month == "December") & v.indicator.isin(INDICATORS)].copy()
    v["deaths"] = pd.to_numeric(v.predicted_value, errors="coerce").fillna(pd.to_numeric(v.data_value, errors="coerce"))
    v["state"] = v.state.replace({"YC": "NY"})
    v = v[v.state.isin(STATE_REGION)]
    v["year"] = v.year.astype(int)
    v["measure"] = v.indicator.map(INDICATORS)
    # States whose drug-specific counts are withheld for low data quality are
    # NaN; a region's drug-specific rate uses only states reporting it.
    counts = v.groupby(["state", "year", "measure"]).deaths.sum(min_count=1).unstack("measure")

    pop = _census_age().assign(state=lambda d: d.fips.str[:2].map(FIPS_STATE)).groupby(["state", "year"])["pop"].sum()
    pop = pop.unstack("year")
    pop[2025] = pop[2024]
    counts = counts.join(pop.stack().rename("population"), how="left")
    counts["region"] = counts.index.get_level_values("state").map(STATE_REGION)

    rows = []
    for (region, year), g in counts.groupby(["region", counts.index.get_level_values("year")]):
        row = {"region": region, "year": year, "deaths": g.all_drugs.sum()}
        for m in INDICATORS.values():
            ok = g[m].notna()
            row[f"{m}_rate"] = 1e5 * g.loc[ok, m].sum() / g.loc[ok, "population"].sum() if ok.any() else float("nan")
            row[f"{m}_states_reporting"] = int(ok.sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    total = df.groupby("year").deaths.sum()
    df["share_of_us_deaths"] = df.apply(lambda r: r.deaths / total[r.year], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r09_recent_waves.csv", index=False)
    pd.set_option("display.width", 200)
    print("All-drug overdose deaths per 100,000:")
    print(df.pivot(index="year", columns="region", values="all_drugs_rate").round(1).to_string())
    print("\nSynthetic opioids (T40.4) per 100,000:")
    print(df.pivot(index="year", columns="region", values="synthetic_opioids_rate").round(1).to_string())
    print("\nShare of US overdose deaths:")
    print(df.pivot(index="year", columns="region", values="share_of_us_deaths").round(3).to_string())


if __name__ == "__main__":
    main()
