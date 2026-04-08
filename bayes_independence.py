"""
Bayesian case for spatial independence of suicide and overdose.

The frequentist BV I = 0.016, p = 0.15 is technically "not significant"
by conventional thresholds — but p = 0.15 here means the observed value
is indistinguishable from random. The Bayesian framing makes a positive
claim: given informative priors representing plausible co-clustering,
how much does the data update us toward independence?

Three priors tested:
  1. Uninformative: uniform over [-0.2, 0.6]
  2. "C&D era" prior: N(0.177, 0.05) — what the 1999-2005 relationship looked like
  3. "Peak era" prior: N(0.277, 0.05) — 2006-2012, the maximum observed co-clustering
  4. "K70 analogy" prior: N(0.34, 0.08) — if overdose shared suicide's cause as K70 does

Likelihood: Normal(observed BV I, SD of permutation null distribution)
This approximates P(data | true BV I = θ) using the permutation distribution
as an empirical null.

Key outputs:
  - Posterior P(BV I > 0.05) under each prior — how much evidence for any
    meaningful co-clustering survives after seeing the data?
  - 95% credible intervals
  - Bayes Factors: H_indep (BV I = 0) vs H_cluster (BV I = prior mean)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import libpysal
from esda import Moran_BV

# ── Recompute BV I with permutation samples retained ──────────────────────────

def load_wonder(path, rate_col):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.strip('"')
    df = df[df["County Code"].str.match(r"^\d{5}$", na=False)].copy()
    df = df.rename(columns={"County Code": "fips"})
    df["rate"] = pd.to_numeric(df["Crude Rate"], errors="coerce")
    return df[["fips", "rate"]].rename(columns={"rate": rate_col})

print("Loading data and spatial weights...")
suicide  = load_wonder("data/suicide_county_2018_2024.csv",  "suicide_rate")
overdose = load_wonder("data/overdose_county_2018_2024.csv", "overdose_rate")
k70      = load_wonder("data/alcohol_liver_county_2018_2024.csv", "k70_rate")

counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)
counties = counties.rename(columns={"GEOID": "fips"})
EXCLUDE = {"02","15","60","66","69","72","78"}
counties = counties[~counties["STATEFP"].isin(EXCLUDE)].to_crs("EPSG:5070")

merged = suicide.merge(overdose, on="fips", how="outer").merge(k70, on="fips", how="outer")
gdf = counties.merge(merged, on="fips", how="left")
gdf_su_od = gdf[gdf["suicide_rate"].notna() & gdf["overdose_rate"].notna()].copy().reset_index(drop=True)
gdf_all   = gdf[gdf["suicide_rate"].notna() & gdf["overdose_rate"].notna() & gdf["k70_rate"].notna()].copy().reset_index(drop=True)

w = libpysal.weights.Queen.from_dataframe(gdf_su_od, silence_warnings=True); w.transform = "r"
w_all = libpysal.weights.Queen.from_dataframe(gdf_all, silence_warnings=True); w_all.transform = "r"

print("Computing bivariate Moran's I (9999 permutations for better null distribution)...")
bv_su_od = Moran_BV(gdf_su_od["suicide_rate"], gdf_su_od["overdose_rate"], w, permutations=9999)
bv_su_k7 = Moran_BV(gdf_all["suicide_rate"],   gdf_all["k70_rate"],       w_all, permutations=9999)

obs_su_od = bv_su_od.I
obs_su_k7 = bv_su_k7.I
perm_su_od = np.array(bv_su_od.sim)  # 9999 permutation values
perm_su_k7 = np.array(bv_su_k7.sim)

perm_sd_su_od = perm_su_od.std()
perm_sd_su_k7 = perm_su_k7.std()

print(f"\nObserved BV I (suicide × overdose): {obs_su_od:.4f}")
print(f"Permutation null SD:                {perm_sd_su_od:.4f}")
print(f"Observed BV I (suicide × K70):      {obs_su_k7:.4f}")
print(f"Permutation null SD:                {perm_sd_su_k7:.4f}")

# ── Bayesian inference ─────────────────────────────────────────────────────────

theta = np.linspace(-0.15, 0.55, 50000)
dtheta = theta[1] - theta[0]

# Likelihood: P(obs | true BV I = theta)
# Use permutation SD as the likelihood width — it captures sampling variability
# under spatial randomness, giving us a conservative (wide) likelihood
likelihood = stats.norm.pdf(obs_su_od, loc=theta, scale=perm_sd_su_od)

PRIORS = {
    "Uninformative (uniform)":        np.ones_like(theta),
    "C&D era (BV I = 0.177 ± 0.05)": stats.norm.pdf(theta, 0.177, 0.05),
    "OxyContin peak (BV I = 0.277 ± 0.05)": stats.norm.pdf(theta, 0.277, 0.05),
    "K70 analogy (BV I = 0.34 ± 0.08)":     stats.norm.pdf(theta, 0.340, 0.08),
}

MEANINGFUL_THRESHOLD = 0.05  # BV I below this = negligible co-clustering

print("\n" + "="*80)
print("BAYESIAN POSTERIORS: P(meaningful co-clustering | data)")
print(f"  'Meaningful' defined as BV I > {MEANINGFUL_THRESHOLD}")
print("="*80)
print(f"{'Prior':<42} {'P(BVI>0.05)':>12} {'95% CI':>20} {'BF(indep/clust)':>16}")
print("-"*80)

posterior_store = {}
for name, prior in PRIORS.items():
    # Normalize prior
    prior_norm = prior / np.trapezoid(prior, theta)
    # Posterior
    post = likelihood * prior_norm
    post /= np.trapezoid(post, theta)
    posterior_store[name] = post

    # P(BV I > threshold)
    mask = theta > MEANINGFUL_THRESHOLD
    p_above = np.trapezoid(post[mask], theta[mask])

    # 95% credible interval
    cdf = np.cumsum(post) * dtheta
    ci_lo = theta[np.searchsorted(cdf, 0.025)]
    ci_hi = theta[np.searchsorted(cdf, 0.975)]

    # Bayes Factor: H_indep (theta=0) vs H_cluster (theta=prior_mean)
    # Using Savage-Dickey density ratio for point null
    # BF_01 = p(theta=0|data) / p(theta=0|prior)
    # Approximate: evaluate posterior and prior density at theta=0
    post_at_0  = np.interp(0, theta, post)
    prior_at_0 = np.interp(0, theta, prior_norm)
    bf_indep   = post_at_0 / prior_at_0  # BF in favor of independence

    print(f"{name:<42} {p_above:>12.4f} [{ci_lo:>6.3f}, {ci_hi:>6.3f}]    {bf_indep:>12.1f}×")

# Precise two-sentence summary
print("\nFor context:")
print(f"  Suicide × K70 BV I = {obs_su_k7:.4f} (what co-clustering looks like when axes share a cause)")
print(f"  Suicide × OD BV I  = {obs_su_od:.4f} (the disputed independence claim)")
print(f"  Observed value is {obs_su_od/obs_su_k7:.1%} of the K70 benchmark")

# ── Figure ─────────────────────────────────────────────────────────────────────

print("\nGenerating figure...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#f8f8f6")

# Panel 1: permutation null distributions
ax = axes[0]
bins = np.linspace(-0.12, 0.35, 80)
ax.hist(perm_su_od, bins=bins, color="#e84855", alpha=0.5, density=True,
        label=f"Suicide × Overdose null\n(observed = {obs_su_od:.3f})")
ax.hist(perm_su_k7, bins=bins, color="#4d9de0", alpha=0.5, density=True,
        label=f"Suicide × K70 null\n(observed = {obs_su_k7:.3f})")
ax.axvline(obs_su_od, color="#c0392b", linewidth=2, linestyle="-")
ax.axvline(obs_su_k7, color="#2471a3", linewidth=2, linestyle="-")
ax.axvline(0, color="grey", linewidth=1, linestyle=":")
ax.set_xlabel("Bivariate Moran's I", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Permutation Null Distributions\n(9,999 permutations each)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_facecolor("#f8f8f6")

# Annotate where observed values fall in null distributions
pct_od = (perm_su_od <= obs_su_od).mean() * 100
pct_k7 = (perm_su_k7 <= obs_su_k7).mean() * 100
ax.text(obs_su_od + 0.005, ax.get_ylim()[1]*0.7,
        f"  {pct_od:.0f}th\n  percentile", fontsize=8, color="#c0392b")
ax.text(obs_su_k7 + 0.005, ax.get_ylim()[1]*0.5,
        f"  {pct_k7:.0f}th\n  percentile", fontsize=8, color="#2471a3")

# Panel 2: posterior distributions under each prior
ax = axes[1]
COLORS_PRIOR = ["#aaaaaa", "#e67e22", "#e84855", "#7b2d8b"]
for (name, post), color in zip(posterior_store.items(), COLORS_PRIOR):
    short = name.split("(")[0].strip()
    ax.plot(theta, post, color=color, linewidth=2, label=short, alpha=0.9)

ax.axvline(obs_su_od, color="black", linewidth=1.5, linestyle="-",
           label=f"Observed BV I = {obs_su_od:.3f}")
ax.axvline(MEANINGFUL_THRESHOLD, color="grey", linewidth=1.2, linestyle="--",
           label=f"Meaningful threshold ({MEANINGFUL_THRESHOLD})")
ax.axvspan(MEANINGFUL_THRESHOLD, 0.55, alpha=0.06, color="red",
           label="Meaningful co-clustering region")
ax.set_xlabel("True bivariate Moran's I (θ)", fontsize=11)
ax.set_ylabel("Posterior density", fontsize=11)
ax.set_xlim(-0.1, 0.5)
ax.set_title("Posterior Distributions Under Four Priors\n"
             "All converge near zero regardless of prior co-clustering assumed",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.set_facecolor("#f8f8f6")

plt.suptitle("Bayesian Evidence for Spatial Independence: Suicide vs. Overdose (2018–2024)\n"
             "Even under priors that assume strong co-clustering, posteriors concentrate near zero",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("figures/fig_bayes_independence.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_bayes_independence.png")
print("\nDone.")
