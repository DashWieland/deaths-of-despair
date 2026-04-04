"""
Spatial regression: what predicts each death regime?

Two separate models:
  A) Suicide regime: rate ~ elevation + firearm_fraction + pct_aian + log_density + income
  B) Overdose regime: rate ~ early_overdose (2006-12 rate) + log_density + pct_poverty + median_age

The early overdose rate serves as the opioid-exposure proxy in lieu of
DEA ARCOS pill count data (ARCOS API currently unreachable). 2006-2012
overdose deaths measure which counties were being flooded with OxyContin
at the epidemic's peak — the actual human exposure, not just supply.

For each model:
  1. OLS baseline
  2. Lagrange Multiplier tests (which spatial model is warranted?)
  3. Spatial lag model (SLM) or spatial error model (SEM) as indicated
  4. Compare R² and spatial residual autocorrelation

If overdose geography in 2018-2024 is well-predicted by 2006-2012 exposure
but NOT by demographics, that supports the manufactured-epidemic argument.
If suicide geography is well-predicted by elevation + guns + isolation but
NOT by early overdose exposure, the two-regime structure is confirmed from
both sides simultaneously.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import libpysal
from esda import Moran
import spreg

# ── Load all predictors ────────────────────────────────────────────────────────

def load_wonder(path, rate_col):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.strip('"')
    df = df[df["County Code"].str.match(r"^\d{5}$", na=False)].copy()
    df = df.rename(columns={"County Code": "fips"})
    df["rate"] = pd.to_numeric(df["Crude Rate"], errors="coerce")
    return df[["fips", "rate"]].rename(columns={"rate": rate_col})

print("Loading death rates...")
su_now = load_wonder("suicide_county_2018_2024.csv",      "suicide_rate")
od_now = load_wonder("overdose_county_2018_2024.csv",     "overdose_rate")
od_old = load_wonder("overdose_county_2006_2012.csv",     "od_2006_2012")
su_old = load_wonder("suicide_county_2006_2012.csv",      "su_2006_2012")
k70    = load_wonder("alcohol_liver_county_2018_2024.csv","k70_rate")
fw     = load_wonder("firearm_suicide_county_2018_2024.csv","firearm_rate")

deaths = (su_now
          .merge(od_now, on="fips", how="outer")
          .merge(od_old, on="fips", how="left")
          .merge(su_old, on="fips", how="left")
          .merge(k70,    on="fips", how="outer")
          .merge(fw,     on="fips", how="left"))
deaths["firearm_fraction"] = (deaths["firearm_rate"] / deaths["suicide_rate"]).clip(0, 1)

with open("elevation_cache.json") as f:
    elevations = json.load(f)
deaths["elevation_m"] = deaths["fips"].map(elevations)

print("Fetching ACS demographics...")
VARS = {
    "B19013_001E": "median_income",
    "B02001_001E": "race_total",
    "B02001_004E": "race_aian",
    "B17001_001E": "pov_total",
    "B17001_002E": "pov_below",
    "B01002_001E": "median_age",
    "B15003_001E": "edu_total",
    "B15003_022E": "edu_ba",
    "B15003_023E": "edu_ma",
    "B15003_024E": "edu_prof",
    "B15003_025E": "edu_phd",
}
resp = requests.get(
    f"https://api.census.gov/data/2023/acs/acs5?get=NAME,{','.join(VARS)}&for=county:*&in=state:*",
    timeout=60
)
raw = resp.json()
acs = pd.DataFrame(raw[1:], columns=raw[0])
acs["fips"] = acs["state"] + acs["county"]
acs = acs.rename(columns=VARS)
for col in VARS.values():
    acs[col] = pd.to_numeric(acs[col], errors="coerce").replace(-666666666, np.nan)
acs["pct_aian"]   = acs["race_aian"] / acs["race_total"] * 100
acs["pct_poverty"] = acs["pov_below"] / acs["pov_total"] * 100
acs["pct_ba_plus"] = (acs["edu_ba"]+acs["edu_ma"]+acs["edu_prof"]+acs["edu_phd"]) / acs["edu_total"] * 100

df = deaths.merge(acs[["fips","median_income","pct_aian","pct_poverty","median_age","pct_ba_plus"]], on="fips", how="left")
df["log_income"] = np.log(df["median_income"].clip(1))

# ── Spatial setup ──────────────────────────────────────────────────────────────

print("Loading shapefile...")
counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)
counties = counties.rename(columns={"GEOID": "fips"})
EXCLUDE = {"02","15","60","66","69","72","78"}
counties = counties[~counties["STATEFP"].isin(EXCLUDE)].to_crs("EPSG:5070")
counties["log_area"] = np.log(counties["ALAND"].astype(float))

gdf = counties.merge(df, on="fips", how="left")

# ── Model A: Suicide rate ──────────────────────────────────────────────────────

SUICIDE_VARS = ["elevation_m", "firearm_fraction", "pct_aian", "pct_ba_plus", "log_income"]
OVERDOSE_VARS = ["od_2006_2012", "pct_poverty", "median_age", "pct_ba_plus", "log_income"]

def run_models(gdf, outcome, predictor_cols, label):
    sub = gdf[[outcome] + predictor_cols + ["geometry"]].dropna().copy().reset_index(drop=True)
    print(f"\n{'='*65}")
    print(f"MODEL: {label}  (n={len(sub)})")
    print(f"{'='*65}")

    w = libpysal.weights.Queen.from_dataframe(sub, silence_warnings=True)
    w.transform = "r"

    y = sub[outcome].values.reshape(-1, 1)
    X = sub[predictor_cols].values
    X_names = predictor_cols

    # ── OLS ──
    ols = spreg.OLS(y, X, w=w, spat_diag=True, name_y=outcome, name_x=X_names)
    print(f"\nOLS:  R²={ols.r2:.4f}  AIC={ols.aic:.1f}")
    print(f"{'Variable':<25} {'Coef':>10} {'t':>8} {'p':>8}")
    print("-"*55)
    for name, coef, (t, p) in zip(["const"]+X_names, ols.betas.flatten(), ols.t_stat):
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        print(f"  {name:<23} {coef:>10.4f} {t:>8.3f} {p:>8.4f} {sig}")

    # Moran's I on OLS residuals
    mi_resid = Moran(ols.u.flatten(), w, permutations=999)
    print(f"\nOLS residuals Moran's I = {mi_resid.I:.4f}  p={mi_resid.p_sim:.4f}")

    # LM tests to choose between lag and error
    lm_lag   = ols.lm_lag
    lm_error = ols.lm_error
    lm_rllag = ols.rlm_lag
    lm_rlerr = ols.rlm_error
    print(f"\nLagrange Multiplier tests:")
    print(f"  LM-lag:   stat={lm_lag[0]:.3f}   p={lm_lag[1]:.4f}")
    print(f"  LM-error: stat={lm_error[0]:.3f}   p={lm_error[1]:.4f}")
    print(f"  RLM-lag:  stat={lm_rllag[0]:.3f}   p={lm_rllag[1]:.4f}")
    print(f"  RLM-err:  stat={lm_rlerr[0]:.3f}   p={lm_rlerr[1]:.4f}")

    # Choose model based on robust LM tests
    use_lag = lm_rllag[1] < lm_rlerr[1]
    model_type = "Spatial Lag" if use_lag else "Spatial Error"
    print(f"\n→ Robust LM favours: {model_type}")

    if use_lag:
        spatial_model = spreg.ML_Lag(y, X, w=w, name_y=outcome, name_x=X_names)
        rho = spatial_model.rho
        print(f"\nSpatial Lag: Pseudo-R²={spatial_model.pr2:.4f}  ρ={rho:.4f}")
    else:
        spatial_model = spreg.ML_Error(y, X, w=w, name_y=outcome, name_x=X_names)
        lam = spatial_model.lam
        print(f"\nSpatial Error: Pseudo-R²={spatial_model.pr2:.4f}  λ={lam:.4f}")

    print(f"{'Variable':<25} {'Coef':>10} {'z':>8} {'p':>8}")
    print("-"*55)
    for name, coef, (z, p) in zip(["const"]+X_names,
                                  spatial_model.betas.flatten(),
                                  spatial_model.z_stat):
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        print(f"  {name:<23} {coef:>10.4f} {z:>8.3f} {p:>8.4f} {sig}")

    # Moran's I on spatial model residuals
    mi_sp = Moran(spatial_model.u.flatten(), w, permutations=999)
    print(f"\n{model_type} residuals Moran's I = {mi_sp.I:.4f}  p={mi_sp.p_sim:.4f}")

    return ols, spatial_model, sub, w

ols_su, sp_su, sub_su, w_su = run_models(gdf, "suicide_rate",  SUICIDE_VARS,  "Suicide (Regime A predictors)")
ols_od, sp_od, sub_od, w_od = run_models(gdf, "overdose_rate", OVERDOSE_VARS, "Overdose (Regime B predictors)")

# Cross-model sanity check: does early overdose predict suicide? (should be weak)
print("\n" + "="*65)
print("CROSS-CHECK: early overdose (2006-12) predicting suicide rate")
print("(should be weak if regimes are truly independent)")
print("="*65)
sub_x = gdf[["suicide_rate","od_2006_2012","log_income","pct_ba_plus"]].dropna().copy().reset_index(drop=True)
w_x = libpysal.weights.Queen.from_dataframe(
    gdf[gdf["suicide_rate"].notna() & gdf["od_2006_2012"].notna() &
        gdf["log_income"].notna() & gdf["pct_ba_plus"].notna()].copy().reset_index(drop=True),
    silence_warnings=True)
w_x.transform = "r"
ols_x = spreg.OLS(
    sub_x["suicide_rate"].values.reshape(-1,1),
    sub_x[["od_2006_2012","log_income","pct_ba_plus"]].values,
    w=w_x, name_y="suicide_rate", name_x=["od_2006_2012","log_income","pct_ba_plus"]
)
print(f"OLS R² = {ols_x.r2:.4f}  (compare to suicide model R² above)")

# ── Figure: coefficient comparison ────────────────────────────────────────────

print("\nGenerating coefficient plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#f8f8f6")

def coef_plot(ax, model, names, title, color):
    coefs = model.betas.flatten()[1:]   # drop intercept
    ses   = np.sqrt(np.diag(model.vm))[1:]
    ci_lo = coefs - 1.96 * ses
    ci_hi = coefs + 1.96 * ses
    y_pos = np.arange(len(names))

    ax.barh(y_pos, coefs, xerr=1.96*ses, color=color, alpha=0.7,
            height=0.5, capsize=4, error_kw={"linewidth":1.5})
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Coefficient (standardised units)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_facecolor("#f8f8f6")
    ax.grid(axis="x", alpha=0.3)

# Standardise coefficients manually: beta_std = beta * SD(X) / SD(y)
def std_coefs_and_ses(model, sub, xcols, outcome):
    sd_y  = sub[outcome].std()
    betas = model.betas.flatten()[1:]   # drop intercept
    ses   = np.sqrt(np.diag(model.vm))[1:]
    std_b = np.array([b * sub[c].std() / sd_y for b, c in zip(betas, xcols)])
    std_s = np.array([s * sub[c].std() / sd_y for s, c in zip(ses,   xcols)])
    return std_b, std_s

sub_su2 = sub_su[["suicide_rate"]  + SUICIDE_VARS].dropna()
sub_od2 = sub_od[["overdose_rate"] + OVERDOSE_VARS].dropna()

bsu, ssu = std_coefs_and_ses(sp_su, sub_su2, SUICIDE_VARS,  "suicide_rate")
bod, sod = std_coefs_and_ses(sp_od, sub_od2, OVERDOSE_VARS, "overdose_rate")

def coef_plot_std(ax, coefs, ses, names, title, color):
    y_pos = np.arange(len(names))
    ax.barh(y_pos, coefs, xerr=1.96*ses, color=color, alpha=0.7,
            height=0.5, capsize=4, error_kw={"linewidth":1.5})
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Standardised coefficient", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_facecolor("#f8f8f6")
    ax.grid(axis="x", alpha=0.3)

coef_plot_std(axes[0], bsu, ssu, SUICIDE_VARS,  "Suicide rate predictors\n(standardised, spatial error model)", "#4d9de0")
coef_plot_std(axes[1], bod, sod, OVERDOSE_VARS, "Overdose rate predictors\n(standardised, spatial error model)", "#e84855")

plt.suptitle("Spatial OLS: Standardised Coefficients for Each Regime\n"
             "Different predictors, different regime — two crises, not one",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("fig_spatial_regression.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_spatial_regression.png")
print("\nDone.")
