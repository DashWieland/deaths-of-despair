"""r03 — Is the suicide x alcoholic-liver-disease pairing an American Indian county effect?

American Indian and Alaska Native (AIAN) people die of chronic liver disease
at about four times the white rate and of suicide at about 1.7 times, and
many high-suicide counties are reservation counties. If the pairing were
mostly those counties, it would say little about "despair" in general.

For each window: Spearman correlations across counties with >= 10 deaths of
both causes, (a) all counties, (b) without counties >= 20% AIAN, (c) without
counties >= 5% AIAN, and (d) all counties with the AIAN share partialled out.
AIAN share = alone or in combination, Census 2020-2024 estimates (a stable
county attribute; one vintage serves every window).

Writes out/revisit/r03_aian.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import OUT
from fetch import path_for
from wonder import PRIMARY, WINDOWS, county


def aian_share() -> pd.Series:
    cols = ["STATE", "COUNTY", "YEAR", "AGEGRP", "TOT_POP", "IAC_MALE", "IAC_FEMALE"]
    df = pd.read_csv(path_for("alldata_2020_2024"), usecols=cols, dtype={"STATE": str, "COUNTY": str}, encoding="latin-1")
    df = df[(df.AGEGRP == 0) & (df.YEAR >= 2)]  # July estimates 2020-2024
    df["fips"] = df.STATE + df.COUNTY
    g = df.groupby("fips")[["TOT_POP", "IAC_MALE", "IAC_FEMALE"]].sum()
    return (g.IAC_MALE + g.IAC_FEMALE) / g.TOT_POP


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    rxy, rxz, ryz = np.corrcoef(rx, ry)[0, 1], np.corrcoef(rx, rz)[0, 1], np.corrcoef(ry, rz)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def main():
    share = aian_share().rename("aian")
    rows = []
    for a, b in [("suicide", "k70"), ("suicide", "overdose"), ("overdose", "k70")]:
        for w in WINDOWS:
            A = county(a, w)[lambda d: d.deaths >= 10]
            B = county(b, w)[lambda d: d.deaths >= 10]
            m = A.merge(B, on="fips", suffixes=(f"_{a}", f"_{b}")).merge(share, left_on="fips", right_index=True)
            x, y, z = m[f"rate_{a}"].to_numpy(), m[f"rate_{b}"].to_numpy(), m["aian"].to_numpy()
            row = {"pair": f"{a} x {b}", "window": w, "n": len(m),
                   "n_aian20": int((z >= 0.20).sum()), "rho_all": stats.spearmanr(x, y)[0]}
            for label, cut in [("rho_no_aian20", 0.20), ("rho_no_aian5", 0.05)]:
                keep = z < cut
                row[label] = stats.spearmanr(x[keep], y[keep])[0]
            row["rho_partial_aian"] = partial_spearman(x, y, z)
            rows.append(row)
            print(f"{row['pair']:<20} {w}  n={row['n']:>4} (AIAN>=20%: {row['n_aian20']:>2})  all={row['rho_all']:+.3f}  "
                  f"no20={row['rho_no_aian20']:+.3f}  no5={row['rho_no_aian5']:+.3f}  partial={row['rho_partial_aian']:+.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "r03_aian.csv", index=False)


if __name__ == "__main__":
    main()
