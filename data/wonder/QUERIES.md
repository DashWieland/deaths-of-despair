# CDC WONDER queries for the 2026 revisit

CDC WONDER does not allow county-level queries through its API, so these
exports come from the web interface. This file is the record of exactly what
was requested. Every export also carries WONDER's own "Query Criteria" block
in its footnotes; keep the footnotes on.

## Settings common to every query

| Setting | Value |
|---|---|
| Section 1, Group Results By | as listed per query |
| Section 1, Measures | Deaths, Population, Crude Rate, plus the 95% confidence interval. Add **Age Adjusted Rate** wherever WONDER offers it (it does not at county level in the 2018–2024 single-race database) |
| Section 2, Location | All states |
| Section 3–5, Demographics | All, unless the query says otherwise |
| Section 6, Cause of death | **ICD-10 Codes**, the codes listed per query |
| Section 8, Other options | Export Results ✓, Show Totals ✗, Show Zero Values ✗ (a county absent from an export had zero deaths), **Show Suppressed Values ✓**, Precision 2 decimal places |

**Data use restriction that binds every output:** NCHS forbids presenting or
publishing death counts of 9 or fewer, or rates based on them, in figures,
maps or tables. Counties below 10 deaths in a window may enter the statistics
(correlations, spatial tests), but no published county-level figure, table or
web export may show a value for them. `revisit/` enforces this with a check that
fails the build if any published output contains such a county.

Databases:

* **D76**: Underlying Cause of Death, 1999–2020 (bridged race)
* **D158**: Underlying Cause of Death, 2018–2024 (single race)
* **D77**: Multiple Cause of Death, 1999–2020 (bridged race)
* **D157**: Multiple Cause of Death, 2018–2024 (single race)

Cause definitions:

| Name | ICD-10 underlying cause | Used for |
|---|---|---|
| `suicide` | X60–X84 | primary (as in the April analysis) |
| `overdose` | X40–X44, Y10–Y14 | primary (drug poisoning, accidental + undetermined; intentional self-poisoning stays in suicide) |
| `k70` | K70 (all subcodes) | primary (alcoholic liver disease) |
| `cd_poisoning` | X40–X45, Y10–Y15, Y45, Y47, Y49 | Case & Deaton's definition (adds alcohol poisoning) |
| `cd_liver` | K70, K73, K74 | Case & Deaton's definition (adds unspecified chronic liver disease and cirrhosis) |
| `overdose_cdc` | X40–X44, X60–X64, X85, Y10–Y14 | CDC's standard drug-overdose definition (national and state queries) |

## A. County, primary definitions, all ages — 27 exports

Group by **County**. Non-overlapping windows; the database switches at 2019.

| Window | DB | Files |
|---|---|---|
| 1999–2000 | D76 | `county/{suicide,overdose,k70}_1999-2000.tsv` |
| 2001–2003 | D76 | `county/{…}_2001-2003.tsv` |
| 2004–2006 | D76 | `county/{…}_2004-2006.tsv` |
| 2007–2009 | D76 | `county/{…}_2007-2009.tsv` |
| 2010–2012 | D76 | `county/{…}_2010-2012.tsv` |
| 2013–2015 | D76 | `county/{…}_2013-2015.tsv` |
| 2016–2018 | D76 | `county/{…}_2016-2018.tsv` |
| 2019–2021 | D158 | `county/{…}_2019-2021.tsv` |
| 2022–2024 | D158 | `county/{…}_2022-2024.tsv` |

## B. County, Case & Deaton definitions, all ages — 8 exports

Group by **County**. Causes `cd_poisoning` and `cd_liver`; windows 1999–2005,
2006–2012, 2013–2018 (D76) and 2019–2024 (D158).
Files: `county_cd/{cd_poisoning,cd_liver}_{window}.tsv`.

## C. County, primary definitions, ages 25–64 — 12 exports

Case & Deaton's working-age population. Group by **County**; Section 3,
Ten-Year Age Groups = 25–34, 35–44, 45–54, 55–64. Causes `suicide`,
`overdose`, `k70`; the same four windows as B.
Files: `county_25_64/{cause}_{window}.tsv`.

## D. Database calibration — 2 exports

`suicide`, group by **County**, 2018–2020, once from D76 and once from D158,
to measure whether the database switch moves county rates.
Files: `calibration/suicide_2018-2020_{d76,d158}.tsv`.

## E. National age-specific deaths, for age standardisation — 4 exports

* 2018–2024: D158, group by **Year**, **Ten-Year Age Groups**, **ICD-10 Codes**,
  codes the union of every definition above (X40–X45, X60–X85, Y10–Y15, Y45,
  Y47, Y49, K70, K73, K74). File: `national/age_d158.tsv`.
* 1999–2020: D77, one export per primary cause (`suicide`, `overdose`, `k70`),
  group by **Year**, **Ten-Year Age Groups**. Files:
  `national/age_{suicide,overdose,k70}_d77.tsv`.

The 1999–2020 tables come from the multiple-cause database: its underlying-cause
counts are the same death records as D76, and the D76 session had lapsed. A
single D77 file grouped by ICD code timed out (HTTP 504), hence one file per
cause. Checked against D158 on 2018–20: the only differences are D158's
suppressed per-code cells (47–55 cells a year, covering the 0.4% gap).

## F. National rates by race and Hispanic origin — 2 exports

Who the deaths reached. Group by **Year**, **Race** (D77: bridged race; D158:
Single Race 6), **Hispanic Origin**, **ICD-10 Codes**, with Age Adjusted Rate.
Codes as in E. Files: `national/race_{d77,d158}.tsv`.

## G. State drug waves, multiple cause — 12 exports

D77 (1999–2020) and D157 (2018–2024). Underlying cause `overdose_cdc`; one
export per drug, selected by Multiple Cause of Death: **T40.1** `heroin`,
**T40.2** `rx_opioids` (natural and semi-synthetic opioids — prescription
pills), **T40.3** `methadone`, **T40.4** `synthetic_opioids` (fentanyl),
**T40.5** `cocaine`, **T43.6** `psychostimulants` (methamphetamine). Group by
**State**, **Year**, with Age Adjusted Rate. (Grouping by Multiple Cause of
Death instead lists every cause on those certificates, 211,171 rows, past
WONDER's 75,000-row limit.) Files: `state/{drug}_{d77,d157}.tsv`.

## H. State series, underlying cause — 6 exports

D77 (1999–2020) and D157 (2018–2024), group by **State**, **Year**, with Age
Adjusted Rate: `suicide`, `k70` and `overdose_cdc` (all drug overdoses), for
comparing each drug wave with the suicide map year by year.
Files: `state/{suicide,k70,overdose}_ucd_{d77,d157}.tsv`.

**Total: 73 exports.** The machine-readable list is `queries.json`; `tools/wonder_runner.js` runs it.
WONDER limits request rates: about 60 queries in 15 minutes from four tabs at
once brought HTTP 429 on every request for over half an hour. Run one tab at a
time, at least 20 seconds apart.

## Fetched by script (not WONDER)

`revisit/common.py` downloads these once into `data/cache/` and verifies each
against its recorded SHA-256:

| File | Source | Size | Use |
|---|---|---|---|
| `tl_2023_us_county.zip` | Census TIGER/Line 2023 | 83 MB | county geometry, contiguity |
| `co-est00int-agesex-5yr.csv` | Census intercensal estimates 2000–2010 | 18 MB | county age structure |
| `cc-est2020int-agesex-all.csv` | Census intercensal estimates 2010–2020 | 17 MB | county age structure |
| `cc-est2024-agesex-all.csv` | Census estimates 2020–2024 | 9 MB | county age structure |
| `cc-est2024-alldata.csv` | Census estimates 2020–2024 | 89 MB | county American Indian / Alaska Native share |
| NCHS model-based county drug poisoning rates 2003–2021 | data.cdc.gov `rpvx-m2md` | ~6 MB | smoothed annual overdose check |
| VSRR provisional drug overdose deaths | data.cdc.gov `xkb8-kh2a` | ~15 MB | state data through 2025 |
| `county_annual14.csv` | Washington Post ARCOS, [wpinvestigative/arcos-api @ 3a8bc22](https://raw.githubusercontent.com/wpinvestigative/arcos-api/3a8bc220c3821b8a8ec123d14b96f1223b61ad79/data/county_annual14.csv), SHA-256 `9344072793edb7dbe7bf5bb90b43ab178eb0566e0d0aa9094584d57489929045` | 1 MB | oxycodone + hydrocodone pills shipped per county, 2006–2014 |

The ARCOS table is licensed ODbL, which is share-alike for derived databases
and so cannot be relicensed under this repo's CC BY 4.0. It is therefore
fetched at build time and never committed; figures and statistics computed from
it are "produced works" under ODbL and carry the credit "Source: The
Washington Post, DEA ARCOS." A byte-identical mirror with a DOI exists at
Mendeley Data `dwfgxrh7tn` v9 (`ARCOS Data 11-29-20.csv`).
