"""r01 — Reproduce the April trajectory, then re-run it under one sample rule.

Question: does the rise-and-collapse of suicide x overdose co-clustering
survive (a) recomputing rates so every window keeps counties with >= 10
deaths, (b) a stricter >= 20-death rule, (c) a fixed panel of counties present
in all four windows, and (d) Assuncao-Reis empirical-Bayes standardisation?

For each variant and pair it reports plain Pearson and Spearman correlations,
bivariate Moran's I in both directions, and Lee's L (which separates the
in-place correlation from the spatial spillover that BV Moran's I mixes in).

Writes out/revisit/r01_consistency.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import sparse, stats
from esda import Moran_BV, Spatial_Pearson

from common import APRIL_WINDOWS, OUT, county_shapes, load_window, queen

SEED = 42
PAIRS = [("suicide", "overdose"), ("suicide", "k70"), ("overdose", "k70")]
APRIL = {  # published trajectory.json values, for the reproduction check
    ("suicide", "overdose"): [0.1775, 0.2770, 0.0761, 0.0155],
    ("suicide", "k70"):      [0.3883, 0.4564, 0.4541, 0.3405],
}


def ar_standardise(deaths: np.ndarray, pop: np.ndarray) -> np.ndarray:
    """Assuncao & Reis (1999) empirical-Bayes standardised rates."""
    b = deaths.sum() / pop.sum()
    r = deaths / pop
    s2 = (pop * (r - b) ** 2).sum() / pop.sum()
    a = s2 - b / pop.mean()
    v = a + b / pop
    v = np.where(v > 0, v, b / pop)
    return (r - b) / np.sqrt(v)


def lees_l(x: np.ndarray, y: np.ndarray, w) -> float:
    """Lee's L with self-inclusive, row-standardised contiguity (Lee 2001)."""
    w.transform = "b"
    v = w.sparse + sparse.identity(w.n, format="csr")
    w.transform = "r"
    v = sparse.diags(1.0 / np.asarray(v.sum(axis=1)).ravel()) @ v
    sp = Spatial_Pearson(connectivity=v, permutations=0).fit(x.reshape(-1, 1), y.reshape(-1, 1))
    return float(sp.association_[0, 1])


def bv(x, y, w, perms):
    np.random.seed(SEED)
    m = Moran_BV(x, y, w, permutations=perms)
    return m.I, m.p_sim


def describe(frame, a, b, w, perms, xa=None, xb=None):
    x = frame[f"rate_{a}"].to_numpy() if xa is None else xa
    y = frame[f"rate_{b}"].to_numpy() if xb is None else xb
    i_ab, p_ab = bv(x, y, w, perms)
    i_ba, _ = bv(y, x, w, perms)
    return {
        "n": len(frame),
        "pearson": stats.pearsonr(x, y)[0],
        "spearman": stats.spearmanr(x, y)[0],
        "bv_I": i_ab, "bv_p": p_ab, "bv_I_reverse": i_ba,
        "lees_L": lees_l(x, y, w),
    }


def main():
    shapes = county_shapes(2023)  # the April boundaries, so the reproduction stays exact
    data = {win: load_window(win) for win in APRIL_WINDOWS}
    rows = []

    def frame_for(win, a, b, rule):
        d = data[win]
        m = d[a].merge(d[b], on="fips", suffixes=(f"_{a}", f"_{b}"))
        if rule == "april":
            # The April pipeline read WONDER's published rate (blank for 10-19 deaths)
            # and, for any pair with K70, also required the third cause (its gdf_all).
            keep = m[f"published_rate_{a}"].notna() & m[f"published_rate_{b}"].notna()
            if "k70" in (a, b):
                third = next(c for c in ("suicide", "overdose") if c not in (a, b))
                keep &= m["fips"].isin(d[third].loc[d[third]["published_rate"].notna(), "fips"])
            m = m[keep].copy()
            m[f"rate_{a}"], m[f"rate_{b}"] = m[f"published_rate_{a}"], m[f"published_rate_{b}"]
        elif rule == "min20":
            m = m[(m[f"deaths_{a}"] >= 20) & (m[f"deaths_{b}"] >= 20)]
        return shapes.merge(m, on="fips").reset_index(drop=True)

    for a, b in PAIRS:
        # fixed panel: counties with >= 20 deaths of both causes in every window
        panel = set.intersection(*[set(frame_for(win, a, b, "min20")["fips"]) for win in APRIL_WINDOWS])
        for wi, win in enumerate(APRIL_WINDOWS):
            for rule in ["april", "min10", "min20", "panel", "eb"]:
                f = frame_for(win, a, b, "min20" if rule == "panel" else ("min10" if rule == "eb" else rule))
                if rule == "panel":
                    f = f[f["fips"].isin(panel)].reset_index(drop=True)
                w = queen(f)
                perms = 9999 if rule == "april" else 999
                if rule == "eb":
                    za = ar_standardise(f[f"deaths_{a}"].to_numpy(float), f[f"population_{a}"].to_numpy(float))
                    zb = ar_standardise(f[f"deaths_{b}"].to_numpy(float), f[f"population_{b}"].to_numpy(float))
                    res = describe(f, a, b, w, perms, za, zb)
                else:
                    res = describe(f, a, b, w, perms)
                rows.append({"pair": f"{a} x {b}", "window": win, "rule": rule, **res})
                print(f"{a:>8} x {b:<8} {win}  {rule:<6} n={res['n']:>4}  r={res['pearson']:+.3f}  "
                      f"rho={res['spearman']:+.3f}  BV_I={res['bv_I']:+.4f} (p={res['bv_p']:.4f})  "
                      f"rev={res['bv_I_reverse']:+.4f}  L={res['lees_L']:+.3f}")
                if rule == "april" and (a, b) in APRIL:
                    print(f"{'':>30}April published {APRIL[(a, b)][wi]:+.4f} -> "
                          f"{'match' if abs(APRIL[(a, b)][wi] - res['bv_I']) < 0.002 else 'MISMATCH'}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "r01_consistency.csv", index=False)
    print(f"\nwrote {OUT / 'r01_consistency.csv'}")


if __name__ == "__main__":
    main()
