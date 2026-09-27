"""r10 — Who the deaths reached: national age-adjusted rates by race and ethnicity.

Case & Deaton's population was white non-Hispanic adults in midlife. This asks
whether the three causes stayed concentrated there. From WONDER's national
tables grouped by year, race, Hispanic origin and ICD-10 code: a cause's
age-adjusted rate is the sum of its codes' age-adjusted rates (direct
standardisation is additive across causes within one population). Suppressed
code cells (1-9 deaths) drop out, a slight undercount that matters only for
the smallest groups.

Non-Hispanic single-race groups only: an age-adjusted rate for Hispanic people
of all races would need its own WONDER query. Overdose uses CDC's standard
definition here (X40-X44, X60-X64, X85, Y10-Y14) so it can be compared with
CDC's published figures.

Writes out/revisit/r10_race.csv
"""
import pandas as pd

from common import OUT
from wonder import WONDER, read_export

OVERDOSE_CDC = {"X40", "X41", "X42", "X43", "X44", "X60", "X61", "X62", "X63", "X64", "X85",
                "Y10", "Y11", "Y12", "Y13", "Y14"}
CAUSES = {
    "suicide": lambda c: "X60" <= c[:3] <= "X84",
    "overdose_cdc": lambda c: c[:3] in OVERDOSE_CDC,
    "k70": lambda c: c[:3] == "K70",
}
RACE_COLS = {"race_d76.tsv": "Race", "race_d158.tsv": "Single Race 6"}


def main():
    frames = []
    for fname, race_col in RACE_COLS.items():
        path = WONDER / "national" / fname
        if not path.exists():
            print(f"(skipping {fname}: not downloaded yet)")
            continue
        d = read_export(path)
        d = d[d["Hispanic Origin"] == "Not Hispanic or Latino"]
        d["year"] = d["Year Code"].astype(int)
        d["group"] = "NH " + d[race_col]
        for cause, pick in CAUSES.items():
            s = d[d["Cause of death Code"].map(pick)]
            g = s.groupby(["year", "group"]).agg(aar=("Age Adjusted Rate", "sum"), deaths=("Deaths", "sum"),
                                                 suppressed_cells=("suppressed", "sum"))
            frames.append(g.assign(cause=cause, source=fname).reset_index())
    df = pd.concat(frames, ignore_index=True)
    # Where the two databases overlap (2018-2020), keep the single-race table.
    df = df.sort_values("source").drop_duplicates(["year", "group", "cause"], keep="last")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r10_race.csv", index=False)
    pd.set_option("display.width", 220)
    for cause in CAUSES:
        w = df[df.cause == cause].pivot(index="year", columns="group", values="aar")
        keep = [c for c in w.columns if c.split(" ", 1)[1] in
                ("White", "Black or African American", "American Indian or Alaska Native", "Asian")]
        print(f"\n{cause}: age-adjusted deaths per 100,000")
        print(w[keep].round(1).to_string())


if __name__ == "__main__":
    main()
