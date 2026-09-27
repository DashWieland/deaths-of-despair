"""r04 — Does the life cycle survive age adjustment?

Crude rates carry each county's age mix: older counties have more suicide and
alcoholic liver deaths per head simply because they are older, and overdose
skews younger. Two adjustments, on counties with >= 10 deaths of both causes:

* isr:   indirect standardisation (agestd.py), the same method in every window
         whose national age table is present
* waar:  WONDER's own direct age-adjusted rate (2000 standard), published for
         the 1999-2020 database only, and only for counties with >= 20 deaths

For each, the Spearman correlation is reported beside the crude one on the
identical county set, so any difference is the adjustment, not the sample.

Writes out/revisit/r04_age.csv
"""
import pandas as pd
from scipy import stats

from agestd import WONDER as _W, _database_file, standardise
from common import OUT
from wonder import PRIMARY, WINDOWS, county

PAIRS = [("suicide", "overdose"), ("suicide", "k70"), ("overdose", "k70")]


def main():
    rows = []
    for w in WINDOWS:
        have_isr = all((_W / "national" / _database_file(w).format(cause=c)).exists() for c in PRIMARY)
        data = {}
        for c in PRIMARY:
            d = county(c, w)
            data[c] = standardise(d, c, w) if have_isr else d.assign(isr=float("nan"))
        for a, b in PAIRS:
            m = data[a][data[a].deaths >= 10].merge(data[b][data[b].deaths >= 10], on="fips", suffixes=("_a", "_b"))
            row = {"pair": f"{a} x {b}", "window": w}
            if have_isr:
                s = m.dropna(subset=["isr_a", "isr_b"])
                row.update(n_isr=len(s), crude_on_isr_set=stats.spearmanr(s.rate_a, s.rate_b)[0],
                           isr=stats.spearmanr(s.isr_a, s.isr_b)[0])
            s = m.dropna(subset=["aar_wonder_a", "aar_wonder_b"])
            if len(s) > 30:
                row.update(n_waar=len(s), crude_on_waar_set=stats.spearmanr(s.rate_a, s.rate_b)[0],
                           waar=stats.spearmanr(s.aar_wonder_a, s.aar_wonder_b)[0])
            rows.append(row)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r04_age.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
