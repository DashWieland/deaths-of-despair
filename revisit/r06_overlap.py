"""r06 — Plain-language overlap and burden measures, window by window.

1. Joint-high counties: counties in the top third of BOTH rates (among
   counties with >= 10 deaths of both causes). If the two rates were unrelated,
   one county in nine would land in both top thirds; the ratio observed /
   expected is a co-location index anyone can read ("2x chance"). Also the
   states that hold the most joint-high counties. These are the counties the
   original thesis project went looking for.
2. Burden versus rate: what share of a cause's deaths happen in the fifth of
   counties with its highest rates, against their share of the population.
   Rate maps show where risk is high; this says how many of the deaths are
   there.

Writes out/revisit/r06_overlap.csv and out/revisit/r06_burden.csv
"""
import pandas as pd

from common import OUT
from wonder import PRIMARY, WINDOWS, county

STATE = {"01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
         "13": "GA", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
         "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV",
         "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR",
         "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA",
         "53": "WA", "54": "WV", "55": "WI", "56": "WY"}


def main():
    overlap, burden = [], []
    for w in WINDOWS:
        data = {c: county(c, w) for c in PRIMARY}
        for a, b in [("suicide", "overdose"), ("suicide", "k70"), ("overdose", "k70")]:
            m = (data[a][lambda d: d.deaths >= 10].merge(data[b][lambda d: d.deaths >= 10], on="fips", suffixes=("_a", "_b")))
            m = m[m.fips.str[:2].isin(STATE)]
            hi_a = m.rate_a >= m.rate_a.quantile(2 / 3)
            hi_b = m.rate_b >= m.rate_b.quantile(2 / 3)
            joint = m[hi_a & hi_b]
            expected = len(m) * hi_a.mean() * hi_b.mean()
            states = joint.fips.str[:2].map(STATE).value_counts().head(5)
            overlap.append({"pair": f"{a} x {b}", "window": w, "n": len(m), "joint_high": len(joint),
                            "expected_if_unrelated": round(expected, 1), "ratio": len(joint) / expected,
                            "top_states": ", ".join(f"{s} {k}" for s, k in states.items())})
        for c in PRIMARY:
            d = data[c][lambda x: (x.deaths >= 10) & x.fips.str[:2].isin(STATE)]
            top = d.rate >= d.rate.quantile(0.8)
            burden.append({"cause": c, "window": w, "counties": len(d),
                           "deaths_share_in_top_fifth": d.deaths[top].sum() / d.deaths.sum(),
                           "population_share_in_top_fifth": d.population[top].sum() / d.population.sum()})
    ov, bu = pd.DataFrame(overlap), pd.DataFrame(burden)
    OUT.mkdir(parents=True, exist_ok=True)
    ov.to_csv(OUT / "r06_overlap.csv", index=False)
    bu.to_csv(OUT / "r06_burden.csv", index=False)
    pd.set_option("display.width", 200)
    print(ov.round(2).to_string(index=False))
    print(bu.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
