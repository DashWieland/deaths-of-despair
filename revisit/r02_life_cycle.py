"""r02 — The life cycle of co-location at three-year resolution, 1999-2024.

For each window and each pair of causes: plain Spearman and population-
weighted Pearson correlation across counties, bivariate Moran's I and Lee's L
on Queen contiguity. Two samples per window:

* window: every county with >= 10 deaths from both causes in that window
  (WONDER suppresses fewer, so this is the most any window can use)
* panel:  the same counties in every window (>= 10 deaths of both causes in
  all nine), so changes over time are not changes in who is counted

Writes out/revisit/r02_life_cycle.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from esda import Moran_BV

from common import OUT, county_shapes, queen
from r01_consistency import lees_l
from wonder import PRIMARY, WINDOWS, county

SEED = 42
PAIRS = [("suicide", "overdose"), ("suicide", "k70"), ("overdose", "k70")]


def wpearson(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return np.sum(w * (x - mx) * (y - my)) / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2))


def main():
    shapes = county_shapes()
    data = {w: {c: county(c, w) for c in PRIMARY} for w in WINDOWS}

    def pair_frame(w, a, b):
        A = data[w][a][lambda d: d.deaths >= 10]
        B = data[w][b][lambda d: d.deaths >= 10]
        m = A.merge(B, on="fips", suffixes=(f"_{a}", f"_{b}"))
        return shapes.merge(m, on="fips").reset_index(drop=True)

    rows = []
    for a, b in PAIRS:
        panel = set.intersection(*[set(pair_frame(w, a, b)["fips"]) for w in WINDOWS])
        for w in WINDOWS:
            for sample in ["window", "panel"]:
                f = pair_frame(w, a, b)
                if sample == "panel":
                    f = f[f["fips"].isin(panel)].reset_index(drop=True)
                x, y = f[f"rate_{a}"].to_numpy(), f[f"rate_{b}"].to_numpy()
                wts = f[f"population_{a}"].to_numpy(float)
                W = queen(f)
                np.random.seed(SEED)
                bv = Moran_BV(x, y, W, permutations=999)
                rows.append({
                    "pair": f"{a} x {b}", "window": w, "sample": sample, "n": len(f),
                    "spearman": stats.spearmanr(x, y)[0], "pearson_popw": wpearson(x, y, wts),
                    "bv_I": bv.I, "bv_p": bv.p_sim, "lees_L": lees_l(x, y, W),
                })
                r = rows[-1]
                print(f"{r['pair']:<20} {w}  {sample:<6} n={r['n']:>4}  rho={r['spearman']:+.3f}  "
                      f"r_w={r['pearson_popw']:+.3f}  BV_I={r['bv_I']:+.3f} (p={r['bv_p']:.3f})  L={r['lees_L']:+.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "r02_life_cycle.csv", index=False)


if __name__ == "__main__":
    main()
