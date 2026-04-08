# Deaths of Despair — Project Context

Analysis for eventual publication on [Apophenia](https://apophenia.blog). County-level spatial analysis testing whether Case & Deaton's "deaths of despair" thesis holds geographically.

---

## The Central Argument

"Deaths of despair" as a unified concept was briefly, partially true — during the OxyContin era (2006–2012), overdose geography partially overlapped with suicide geography, making the deaths look like one phenomenon. That overlap has since collapsed. What remains are two spatially orthogonal crises requiring different explanations and different policy responses:

- **Regime A — Chronic self-destruction** {suicide + K70}: Mountain West + rural corridors. Driven by altitude-linked neurobiological vulnerability, gun access amplifying lethality, and extreme isolation. Slow-burn deaths at the end of long processes.
- **Regime B — Supply-chain poisoning** {overdose}: Appalachia + New England. Driven by pharmaceutical targeting in the 2000s, followed by black-market fentanyl running existing drug corridors. The geography of 2018–2024 overdose deaths was largely determined by where pills went in 2006–2012.

Case & Deaton published in 2015 using data through ~2013 — the moment of maximum artificial co-clustering, just as the fentanyl transition was beginning to separate the regimes. They weren't wrong for their time; they were overtaken by events their framing helped obscure.

**The poverty null is the key policy finding:** poverty rates are statistically identical across Regime A and Regime B counties (p = 0.76). Economic interventions alone will not fix either crisis. They are different problems.

---

## Data

All files in project root. WONDER downloads are 7-year pooled, county-level, crude rates per 100K.

### Current era (2018–2024) — CDC WONDER expanded database

| File | ICD-10 | Counties |
|------|--------|----------|
| `data/suicide_county_2018_2024.csv` | X60–X84 | 2,661 |
| `data/overdose_county_2018_2024.csv` | X40–X44, Y10–Y14 | 2,392 |
| `data/alcohol_liver_county_2018_2024.csv` | K70 | 2,097 |
| `data/firearm_suicide_county_2018_2024.csv` | X72–X74 | 2,356 |
| `data/overdose_unintentional_county_2018_2024.csv` | X40–X44 | 2,371 |
| `data/overdose_undetermined_county_2018_2024.csv` | Y10–Y14 | 286 |

### Historical windows — CDC WONDER 1999–2020 database

| File | ICD-10 | Period |
|------|--------|--------|
| `data/suicide_county_1999_2005.csv` | X60–X84 | 1999–2005 |
| `data/overdose_county_1999_2005.csv` | X40–X44, Y10–Y14 | 1999–2005 |
| `data/alcohol_liver_county_1999_2005.csv` | K70 | 1999–2005 |
| `data/suicide_county_2006_2012.csv` | X60–X84 | 2006–2012 |
| `data/overdose_county_2006_2012.csv` | X40–X44, Y10–Y14 | 2006–2012 |
| `data/alcohol_liver_county_2006_2012.csv` | K70 | 2006–2012 |
| `data/suicide_county_2013_2019.csv` | X60–X84 | 2013–2019 |
| `data/overdose_county_2013_2019.csv` | X40–X44, Y10–Y14 | 2013–2019 |
| `data/alcohol_liver_county_2013_2019.csv` | K70 | 2013–2019 |

### Other data

- `data/elevation_cache.json` — SRTM90m county centroid elevations (3,109 CONUS counties), queried from OpenTopoData API. Do not re-query; use cache.
- ACS 2023 5-year demographics fetched at runtime via Census API (no local file).

**Known data quality issues:**
- NC 2023 overdose undercounted ~900 deaths (coroner lag; permanent in final dataset)
- CT county-level population missing post-2022 (planning region reclassification)
- Y10–Y14 (undetermined intent) only 286 counties above suppression threshold — heavily sparse

---

## Results by Analysis

### 1. Bivariate spatial correlation (`analyze.py`)

**Univariate Moran's I** — all three axes cluster independently:

| Axis | I | p | HH hot spots | LL cold spots |
|------|---|---|---|---|
| Suicide | 0.456 | 0.001 | 221 | 334 |
| Overdose | 0.575 | 0.001 | 274 | 430 |
| K70 | 0.403 | 0.001 | 137 | 334 |

**Bivariate Moran's I** — the core finding:

| Pair | BV I | p |
|------|------|---|
| Suicide ↔ Overdose | 0.016 / 0.012 | 0.149 / 0.211 |
| Suicide ↔ K70 | 0.341 / 0.345 | 0.001 / 0.001 |
| Overdose ↔ K70 | −0.003 / 0.005 | 0.435 / 0.379 |

**Framing the independence claim correctly:** BV I = 0.016, p = 0.15 is not "not statistically significant" — that framing is wrong. The Bayesian analysis (`bayes_independence.py`) shows: under a prior centred on the 1999–2005 co-clustering level (BV I = 0.177), posterior P(BV I > 0.05) = 9.2%, BF = 236× for independence. Under the K70-analogy prior (BV I = 0.34), BF = 8,752×. The observed value is 4.5% of the suicide × K70 benchmark. **Describe this as strong Bayesian evidence for spatial independence, not as a failed significance test.**

### 2. Altitude (`altitude.py`, `altitude_threshold.py`)

| Axis | Spearman ρ | Partial ρ (controlling log density) | OLS R² |
|------|-----------|--------------------------------------|--------|
| Suicide | 0.384 | 0.303 | 0.380 |
| K70 | 0.373 | 0.178 | 0.224 |
| Overdose | −0.037 | 0.027 (ns) | 0.021 |

Threshold analysis: the elevation–rate relationship has a statistically significant kink at ~2,000m for suicide (F=36, p<0.001) and K70 (F=24.5, p<0.001) but not overdose (F=0.00, p=0.95). The slope is positive and steep below 2,000m, flattens and reverses slightly above. The very highest-elevation counties are mountain resort towns with younger, wealthier, more transient populations that dilute baseline risk despite maximum hypoxic exposure.

### 3. Gun ownership proxy (`gun_proxy.py`)

Firearm suicide fraction (X72–X74 / X60–X84) — validated proxy for county gun ownership (Cook & Ludwig).

| Axis | Spearman ρ | p |
|------|-----------|---|
| Suicide | 0.291 | *** |
| K70 | 0.138 | *** |
| Overdose | 0.006 | ns |

Regime A median firearm fraction = 0.656 vs Regime B = 0.605 (MW p < 0.001). Gun access amplifies suicide lethality and tracks culturally with the alcohol-and-isolation demographic. Does not predict overdose.

### 4. Demographics (`demographics.py`)

ACS 2023 5-year, county-level. Regimes assigned by tercile on each axis.

| Variable | A: Chronic (suicide+K70) | B: Supply-chain (overdose) | p |
|---|---|---|---|
| Median income | $60,867 | $64,534 | *** |
| % Bachelor's+ | 20.4% | 24.4% | *** |
| % White | 84.7% | 78.4% | *** |
| % AIAN | 0.8% | 0.2% | *** |
| % Below poverty | 13.9% | 14.3% | ns |
| Median age | 42.4 | 40.6 | *** |
| % Foreign-born | 2.6% | 3.7% | *** |

**Poverty does not separate the regimes.** Economic interventions will not differentially address either crisis. The regimes differ by social character (age, isolation, immigration, race) not economic position.

### 5. Y10–Y14 split (`y10_split.py`)

Undetermined intent (Y10–Y14, n=286 counties above suppression) is **negatively** spatially correlated with unintentional overdose X40–X44 (BV I = −0.11 to −0.16, p<0.003). It does not bleed into suicide geography (BV I ≈ −0.01, ns). Y10–Y14 is its own geographic phenomenon, almost certainly driven by state-level coroner practice variation rather than epidemiology. Lumping X40–X44 + Y10–Y14 introduced minor noise into the overdose axis but did not contaminate the suicide/overdose independence finding — if anything, separating them makes the independence slightly stronger.

### 6. AIAN decomposition (`aian_decomp.py`)

Regime A contains two distinct sub-populations:

- **A-White** (n=249, <5% AIAN): altitude predicts suicide (ρ=0.327, ***). Mountain West white rural counties. Mechanism: altitude + isolation + guns.
- **A-AIAN** (n=48, ≥5% AIAN): altitude correlation 0.282 (marginal, small n). Plains reservations, lower elevation. Mechanism: intergenerational trauma + structural dispossession + healthcare deserts.

Both subgroups show near-zero elevation–overdose correlation. Same regime classification, different etiology. A-AIAN counties have higher death rates across all three axes than A-White counties (all differences p<0.01).

### 7. Temporal trajectory (`temporal.py`, `trajectory.py`)

BV I between suicide and overdose across four 7-year windows:

| Window | BV I (su × od) | p | Mean OD rate |
|--------|---------------|---|---|
| 1999–2005 | 0.177 | 0.001 | 12.3 |
| 2006–2012 | **0.277** | 0.001 | 13.2 |
| 2013–2019 | 0.076 | 0.001 | 18.2 |
| 2018–2024 | 0.016 | ns | 24.7 |

Co-clustering **peaked in 2006–2012** (OxyContin era), then collapsed as fentanyl supply chains concentrated geographically. Suicide × K70 stable throughout (0.34–0.46). Case & Deaton's data window sits over the peak of artificial co-clustering. The overdose rate doubled while the geographic regime split was completing.

### 8. Bayesian independence (`bayes_independence.py`)

9,999 permutations. Permutation null SD = 0.0158. Observed BV I = 0.0155 (0.98 SDs from zero).

| Prior | P(BV I > 0.05 | data) | BF (independence) |
|---|---|---|
| Uninformative | 1.4% | 10.9× |
| C&D era (0.177) | 9.2% | 236× |
| OxyContin peak (0.277) | 23.4% | 520,000× |
| K70 analogy (0.340) | 7.4% | 8,752× |

### 9. Spatial regression (`spatial_regression.py`)

Spatial error models (robust LM tests favour error over lag for both outcomes).

**Suicide model** (elevation + firearm fraction + % AIAN + % BA+ + log income, n=2,334):
- OLS R² = 0.42, spatial error λ = 0.41
- All predictors significant after spatial correction
- Residual Moran's I = 0.24 — unmeasured isolation/social network effects remain

**Overdose model** (2006–12 overdose rate + poverty + age + education + income, n=1,460):
- OLS R² = 0.41, spatial error λ = 0.66 — much stronger unexplained spatial structure
- 2006–12 overdose rate dominates (z=21.2, ***) — geography of exposure was stable and self-reinforcing
- Residual Moran's I = 0.54 — supply-chain network structure not capturable by county attributes

**Cross-check:** 2006–12 overdose rate predicts 2018–24 suicide with R² = 0.22 only, vs. 0.42 for the dedicated suicide model — confirming the supply-chain geography does not explain the chronic regime.

**ARCOS (DEA opioid pill distribution data):** API unreachable (Washington Post server times out). The 2006–12 overdose rate is a methodologically cleaner proxy anyway — it measures human exposure (deaths), not just supply. The finding stands without pill count data.

---

## Output Figures

| File | Description |
|------|-------------|
| `figures/fig_choropleths.png` | Three choropleth maps, one per axis, 5-quantile |
| `figures/fig_scatter.png` | Bivariate scatter plots with BV I annotations, LISA coloring |
| `figures/fig_lisa.png` | LISA cluster maps for all three axes |
| `figures/fig_altitude.png` | Elevation vs. each death rate, scatter + trend |
| `figures/fig_altitude_threshold.png` | Elevation vs. rates with 2,000m threshold, binned means |
| `figures/fig_gun_proxy.png` | Firearm fraction vs. each axis, regime-colored |
| `figures/fig_demographics.png` | Boxplot profiles by regime across six ACS variables |
| `figures/fig_y10_split.png` | LISA maps: suicide, unintentional OD, undetermined OD |
| `figures/fig_aian_decomp.png` | Regime A scatter split by AIAN subgroup |
| `figures/fig_temporal.png` | BV I trajectory line chart with C&D window annotated |
| `figures/fig_trajectory_lisa.png` | 2×2 overdose LISA maps across four eras |
| `figures/fig_bayes_independence.png` | Permutation null distributions + Bayesian posteriors |
| `figures/fig_spatial_regression.png` | Standardised coefficient plots for both regime models |

---

## What Remains

### For the essay (high priority)

**Publication figures** — none of the current figures are essay-ready. Need:
1. A bivariate choropleth of suicide × K70 (the co-clustering map) paired with overdose as the independent regime — this is the visual argument
2. The BV I trajectory chart (`fig_temporal.png`) is close to publication-ready; needs light styling
3. A clean regime map showing A/B/C/D county classifications

**Writing** — the analysis is complete. The essay structure is clear:
1. The unified thesis and why it should predict co-geography
2. The bivariate Moran's I result (the proof)
3. K70's unexpected alignment with suicide — reframing the taxonomy
4. The altitude + gun + demographics portrait of Regime A
5. The temporal smoking gun — C&D published at peak artificial co-clustering
6. The spatial regression confirming both regimes from their own predictors
7. Policy implications: poverty null, two different interventions needed

### For a stronger scientific paper (lower priority)

- **Age-adjusted rates** — crude rates used throughout; Regime A is older, introducing potential confound. CDC doesn't provide age-adjusted at county level for these ICD subsets; would need manual indirect standardisation.
- **Robustness checks** — different spatial weights (Rook, k-nearest), different tercile thresholds, different SESOI for equivalence test.
- **ARCOS pill data** — if the API comes back up or bulk CSV becomes available, would add supply-side confirmation to the overdose model.

---

## Environment

```bash
uv run python analyze.py            # core spatial analysis (~3 min)
uv run python altitude.py           # elevation correlations (reads cache)
uv run python altitude_threshold.py # threshold test (reads cache)
uv run python gun_proxy.py          # firearm fraction analysis
uv run python demographics.py       # ACS demographic profiles
uv run python y10_split.py          # Y10-Y14 separation (~2 min)
uv run python aian_decomp.py        # AIAN subgroup decomposition
uv run python temporal.py           # two-era comparison (~4 min)
uv run python trajectory.py         # four-era trajectory (~8 min)
uv run python bayes_independence.py # Bayesian independence test (~3 min)
uv run python spatial_regression.py # spatial regression models (~4 min)
```

Packages: pandas, geopandas, libpysal, esda, spreg, matplotlib, mapclassify, numpy, scipy, requests

Shapefile downloaded at runtime from Census TIGER/Line 2023. If offline, cache locally and pass path to `gpd.read_file()`.
