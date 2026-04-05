# Deaths of Despair — Spatial Analysis

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

County-level spatial analysis of US "deaths of despair" (2018–2024), examining whether suicide, drug overdose, and alcoholic liver disease co-cluster geographically as the Case & Deaton unified-epidemic thesis implies.

Analysis and write-up for [Apophenia](https://apophenia.blog).

&copy; 2026 Dash Wieland. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — you are free to share and adapt this work for any purpose provided you give appropriate credit.

---

## Data

All data files are included in this repository.

### CDC WONDER — current era (2018–2024)

Downloaded from the CDC WONDER Underlying Cause of Death expanded database (2018–2024). All files are county-level, 7-year pooled, crude rates per 100,000 population. Counties with fewer than 10 deaths in the period are suppressed and absent from the files.

| File | ICD-10 codes | Counties |
|---|---|---|
| `suicide_county_2018_2024.csv` | X60–X84 | 2,661 |
| `overdose_county_2018_2024.csv` | X40–X44, Y10–Y14 | 2,392 |
| `alcohol_liver_county_2018_2024.csv` | K70 | 2,097 |
| `firearm_suicide_county_2018_2024.csv` | X72–X74 | 2,356 |
| `overdose_unintentional_county_2018_2024.csv` | X40–X44 | 2,371 |
| `overdose_undetermined_county_2018_2024.csv` | Y10–Y14 | 286 |

### CDC WONDER — historical windows (1999–2020 database)

Used for temporal trajectory analysis across four non-overlapping 7-year windows.

| File | ICD-10 codes | Period |
|---|---|---|
| `suicide_county_1999_2005.csv` | X60–X84 | 1999–2005 |
| `overdose_county_1999_2005.csv` | X40–X44, Y10–Y14 | 1999–2005 |
| `alcohol_liver_county_1999_2005.csv` | K70 | 1999–2005 |
| `suicide_county_2006_2012.csv` | X60–X84 | 2006–2012 |
| `overdose_county_2006_2012.csv` | X40–X44, Y10–Y14 | 2006–2012 |
| `alcohol_liver_county_2006_2012.csv` | K70 | 2006–2012 |
| `suicide_county_2013_2019.csv` | X60–X84 | 2013–2019 |
| `overdose_county_2013_2019.csv` | X40–X44, Y10–Y14 | 2013–2019 |
| `alcohol_liver_county_2013_2019.csv` | K70 | 2013–2019 |

### Elevation cache

`elevation_cache.json` — SRTM 90m elevation (meters) at the centroid of all 3,109 contiguous US counties, queried from the [OpenTopoData API](https://www.opentopodata.org/). Do not re-query; use the cache.

### ACS demographics

ACS 2023 5-year county-level estimates are fetched at runtime from the [Census API](https://api.census.gov/). No local file; requires an internet connection.

### County shapefile

The 2023 TIGER/Line county shapefile is downloaded at runtime from the Census Bureau in all scripts that produce maps:

```
https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip
```

Maps are projected to EPSG:5070 (NAD83 / Conus Albers).

**Known data quality issues:**
- NC 2023 overdose deaths undercounted by ~900 (coroner reporting lag; permanent in the final dataset)
- CT county-level population data missing post-2022 due to planning region reclassification
- Y10–Y14 (undetermined intent) is sparse — only 286 counties above the suppression threshold

---

## Analysis scripts

Each script is self-contained: it loads the data it needs, runs the analysis, prints results to stdout, and saves one or more figures.

| Script | What it does | Output figure(s) |
|---|---|---|
| `analyze.py` | Univariate Moran's I, bivariate Moran's I across all three axes, LISA cluster maps | `fig_choropleths.png`, `fig_scatter.png`, `fig_lisa.png` |
| `altitude.py` | Spearman and partial correlations between county elevation and each death rate | `fig_altitude.png` |
| `altitude_threshold.py` | Piecewise linear test for a kink in the elevation–rate relationship at 2,000m | `fig_altitude_threshold.png` |
| `gun_proxy.py` | Firearm suicide fraction as a county gun ownership proxy; correlation with each axis | `fig_gun_proxy.png` |
| `demographics.py` | ACS demographic profiles by regime; Mann-Whitney tests across regime pairs | `fig_demographics.png` |
| `y10_split.py` | Separates undetermined intent (Y10–Y14) from unintentional overdose (X40–X44); bivariate Moran's I for each against suicide | `fig_y10_split.png` |
| `aian_decomp.py` | Splits Regime A counties by AIAN population share; tests whether altitude–suicide correlation holds in predominantly white counties | `fig_aian_decomp.png` |
| `temporal.py` | Bivariate Moran's I for two eras (1999–2005 and 2006–2012) with LISA maps | `fig_temporal.png` |
| `trajectory.py` | Bivariate Moran's I across all four 7-year windows; overdose LISA maps per era | `fig_trajectory.png`, `fig_trajectory_lisa.png` |
| `bayes_independence.py` | Bayesian inference on the suicide–overdose BV I using four priors; Savage-Dickey Bayes factors | `fig_bayes_independence.png` |
| `spatial_regression.py` | OLS and spatial error models for each regime; Lagrange Multiplier tests; standardised coefficient plots | `fig_spatial_regression.png` |
| `pub_figures.py` | Publication-quality versions of the three core figures | `fig_pub_bivariate.png`, `fig_pub_trajectory.png`, `fig_pub_regimes.png` |

---

## Reproducing the analysis

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Setup

```bash
git clone https://github.com/DashWieland/deaths-of-despair.git
cd deaths-of-despair
uv sync
```

### Run

Scripts can be run in any order except that `pub_figures.py` recomputes the four-era trajectory internally and is slower than the others. Recommended order matches the analytical sequence:

```bash
uv run python analyze.py            # ~3 min
uv run python altitude.py
uv run python altitude_threshold.py
uv run python gun_proxy.py
uv run python demographics.py
uv run python y10_split.py          # ~2 min
uv run python aian_decomp.py
uv run python temporal.py           # ~4 min
uv run python trajectory.py         # ~8 min
uv run python bayes_independence.py # ~3 min
uv run python spatial_regression.py # ~4 min
uv run python pub_figures.py        # ~10 min
```

All figures are written to the project root. Runtime estimates assume a typical broadband connection for the shapefile download; each script downloads the shapefile independently.

### Dependencies

Managed via `pyproject.toml` and `uv.lock`. Key packages: `pandas`, `geopandas`, `libpysal`, `esda`, `spreg`, `matplotlib`, `mapclassify`, `numpy`, `scipy`, `requests`.

---

## Data citations

CDC WONDER data is in the public domain and may be freely used, copied, distributed, or published. Please cite as:

> Centers for Disease Control and Prevention, National Center for Health Statistics. National Vital Statistics System, Mortality 2018-2024 on CDC WONDER Online Database. Accessed 2025. https://wonder.cdc.gov/

> Centers for Disease Control and Prevention, National Center for Health Statistics. National Vital Statistics System, Mortality 1999-2020 on CDC WONDER Online Database. Accessed 2025. https://wonder.cdc.gov/

County elevation data from the SRTM 90m dataset via the OpenTopoData API (https://www.opentopodata.org/). County shapefile from the US Census Bureau TIGER/Line 2023 (public domain). Demographic data from the American Community Survey 5-Year Estimates, 2023 (US Census Bureau, public domain).
