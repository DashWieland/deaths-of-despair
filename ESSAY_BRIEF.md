# Essay Brief: Deaths of Despair
## Handoff document for the Apophenia writing agent

---

## What you are doing

Writing a data-journalism essay for [Apophenia](https://apophenia.blog) based on a completed spatial analysis. The analysis is finished and the findings are settled — your job is to write the essay, not to interpret or re-run the data. All statistics cited below are confirmed output from the analysis scripts.

The essay should read like serious data journalism: precise about methods and numbers, willing to take a position, not padded with qualifications it doesn't need.

---

## The publication

**Apophenia** is a personal blog focused on finding patterns in data. The house style is analytical and direct — it trusts readers to follow a statistical argument without over-explanation. It uses visualizations as evidence, not decoration. Tone is confident, not academic. Write in first person where natural.

---

## The argument

Case & Deaton (2015) coined "deaths of despair" to describe a unified syndrome: the simultaneous rise of suicide, drug overdose, and alcoholic liver disease in white working-class Americans. The framing implies a common cause — economic dislocation, loss of meaning — and therefore a common solution.

The spatial data doesn't support the unified thesis, at least not anymore. Suicide and drug overdose now occupy **geographically distinct** parts of the country with near-zero spatial correlation. They are two different crises with different drivers, different geographies, and different policy implications. The unified "deaths of despair" frame obscures more than it reveals.

Case & Deaton weren't wrong when they wrote — they were working with data from the window of maximum artificial co-clustering. The OxyContin epidemic initially spread broadly enough that high-suicide and high-overdose areas partially overlapped. Then fentanyl arrived, supply chains concentrated geographically, and the regimes split. They published in 2015 using data through ~2013, exactly at the peak of that overlap, just as the split was beginning.

**The key finding the essay hangs on:** Bivariate Moran's I between suicide and overdose = 0.016, p = 0.15. This is not a failed significance test — it is strong positive evidence for spatial independence. Under a Bayesian prior centered on the 1999–2005 co-clustering level, the Bayes factor for independence is 236×. Under the OxyContin-peak prior, it is 520,000×.

The unexpected finding that reframes the taxonomy: **alcoholic liver disease (K70) clusters with suicide, not overdose** (BV I = 0.341, p = 0.001). What Case & Deaton called "deaths of despair" is actually two distinct phenomena:
- **Regime A — Chronic self-destruction** {suicide + K70}: Mountain West, driven by altitude, gun access, isolation
- **Regime B — Supply-chain poisoning** {overdose}: Appalachia + New England, driven by pharmaceutical targeting and fentanyl geography

**The policy punchline:** Poverty rates are statistically identical across both regimes (13.9% vs 14.3%, p = 0.76). Economic interventions will not differentially address either crisis.

---

## Essay structure

Work through these sections in order. The argument builds — don't reorder.

### 1. The thesis and its prediction
Introduce Case & Deaton's unified framing. If the thesis is correct — a common cause of economic and social despair — then the three death types should cluster together geographically. High-suicide counties should also be high-overdose counties. Set up the test.

### 2. The result: they don't co-cluster
Present the bivariate Moran's I finding. Use the Bayesian framing, not the p-value framing. BF = 236× under the most conservative informative prior. The observed BV I is 4.7% of the suicide × K70 benchmark (which is what co-clustering looks like when two things actually share a cause).

**Figure here:** `fig_pub_bivariate.png` — the bivariate choropleth showing suicide (purple, Mountain West) vs. overdose (teal, Appalachia/New England). This is the visual centerpiece of the essay. Use it large.

### 3. K70 breaks the taxonomy
The surprise: alcoholic liver disease doesn't go with overdose — it goes with suicide. BV I(suicide × K70) = 0.341, p = 0.001 across all four time periods tested. BV I(overdose × K70) ≈ 0.00, ns. This means the "three deaths of despair" split 2-1, not 1-2, and the taxonomy needs to be redrawn.

**Figure here:** `fig_pub_regimes.png` — the regime classification map showing Regime A (purple, Mountain West) and Regime B (teal, Appalachia/New England).

### 4. Portrait of Regime A (suicide + K70)
What predicts the chronic regime? Elevation is the headline predictor (Spearman ρ = 0.384 with suicide, 0.373 with K70; near-zero with overdose). The relationship has a kink at ~2,000m — the slope is steep below that, flattens above (resort towns dilute the signal). Gun access amplifies lethality (firearm fraction correlates with suicide ρ = 0.291***, with overdose ρ = 0.006 ns). Demographics: older (median age 42.4), more white (84.7%), more AIAN (0.8%), lower foreign-born (2.6%).

The AIAN finding deserves a sentence: Regime A contains two sub-populations — white rural Mountain West counties where altitude + isolation + guns drive the pattern, and tribal-lands counties where structural dispossession and healthcare deserts are the more plausible mechanisms. Same regime classification, different etiology.

Spatial regression confirms: elevation + firearm fraction + % AIAN + education + income together explain 42% of suicide variance (R² = 0.42). Residual spatial autocorrelation (Moran's I = 0.24 on residuals) suggests unmeasured isolation and social network effects.

### 5. Portrait of Regime B (overdose)
What predicts the supply-chain regime? The 2006–2012 overdose rate — which era counties were being flooded with OxyContin — explains more of 2018–2024 overdose geography than any demographic variable. In the spatial regression, its z-statistic is 21.2 (***). The overdose geography of the fentanyl era was largely determined 15 years ago by where pills went. Demographics: slightly lower income ($64,534 vs $60,867), higher education, less white (78.4%), more foreign-born (3.7%).

Residual spatial autocorrelation on the overdose model is much higher (Moran's I = 0.54 on residuals) — supply-chain network structure doesn't reduce to county attributes.

### 6. The temporal smoking gun
This is the mechanism that explains how Case & Deaton got a real finding that is now obsolete. Show the trajectory:

| Window | BV I (suicide × overdose) | p |
|---|---|---|
| 1999–2005 | 0.177 | 0.001 |
| 2006–2012 | 0.277 | 0.001 |
| 2013–2019 | 0.076 | 0.001 |
| 2018–2024 | 0.016 | ns |

OxyContin was distributed broadly enough to create partial geographic overlap with the chronic suicide belt. As prescription crackdowns pushed users to heroin and then fentanyl, the supply chain concentrated in Appalachia and New England — regions with existing drug infrastructure — and the overlap collapsed. Suicide × K70 stayed stable throughout (0.34–0.46). Case & Deaton's data window covered 1999–2013, the period of peak artificial co-clustering.

**Figure here:** `fig_pub_trajectory.png` — the BV I trajectory chart showing the collapse of suicide–overdose co-clustering alongside the stable suicide–K70 line.

### 7. Policy implications
The poverty null is the most important policy finding: poverty rates do not differ between the regimes (p = 0.76). This means economic interventions are not the lever that separates high-suicide counties from high-overdose counties. If poverty drove despair uniformly, you'd expect the most economically distressed counties to be highest on both axes. They're not.

What Regime A needs: mental health infrastructure, lethal means counseling, treatment for altitude-linked mood disorders, rural isolation interventions. What Regime B needs: fentanyl harm reduction, naloxone distribution, treatment capacity in the specific counties where the supply chain runs. These are different programs targeting different places.

Calling both crises "deaths of despair" and reaching for economic policy as the solution misses both of them.

---

## Figures: what's available and when to use each

### Publication-ready (use these in the essay)

| File | What it shows | Where in essay |
|---|---|---|
| `fig_pub_bivariate.png` | Bivariate choropleth: suicide (purple) vs. overdose (teal) geography, full CONUS | Section 2 — visual centerpiece |
| `fig_pub_trajectory.png` | BV I trajectory 1999–2024, with C&D data window annotated | Section 6 — the temporal argument |
| `fig_pub_regimes.png` | County regime classification map (A/B/C/D) | Section 3 or conclusion |

### Analytical figures (available for inline use or appendix)

| File | What it shows |
|---|---|
| `fig_choropleths.png` | Three side-by-side choropleth maps, one per axis, 5-quantile |
| `fig_scatter.png` | Bivariate scatter plots with BV I annotations, LISA coloring |
| `fig_lisa.png` | LISA cluster maps (HH/LL/HL/LH) for all three axes |
| `fig_altitude.png` | Elevation vs. each death rate, scatter + trend |
| `fig_altitude_threshold.png` | Elevation vs. rates with 2,000m threshold and binned means |
| `fig_gun_proxy.png` | Firearm fraction vs. each axis, regime-colored |
| `fig_demographics.png` | Boxplot demographic profiles by regime |
| `fig_y10_split.png` | LISA maps: suicide, unintentional OD, undetermined OD |
| `fig_aian_decomp.png` | Regime A scatter split by AIAN subgroup |
| `fig_trajectory_lisa.png` | 2×2 overdose LISA maps across four eras |
| `fig_bayes_independence.png` | Permutation null distributions + Bayesian posteriors |
| `fig_spatial_regression.png` | Standardised coefficient plots for both regime models |

---

## Key statistics to cite

Cite these exactly as written. They are confirmed analysis output.

**Spatial independence:**
- BV I (suicide × overdose) = 0.016, p = 0.15
- BV I (suicide × K70) = 0.341, p = 0.001
- BV I (overdose × K70) = −0.003, p = 0.435
- Bayes factor for independence (C&D-era prior): 236×
- Bayes factor for independence (OxyContin-peak prior): 520,000×
- Observed BV I is 4.7% of the suicide × K70 benchmark

**Univariate clustering (each axis clusters strongly on its own):**
- Suicide Moran's I = 0.456, p = 0.001
- Overdose Moran's I = 0.575, p = 0.001
- K70 Moran's I = 0.403, p = 0.001

**Altitude:**
- Suicide: Spearman ρ = 0.384, partial ρ = 0.303 (controlling log density), OLS R² = 0.38
- K70: ρ = 0.373, partial ρ = 0.178, R² = 0.22
- Overdose: ρ = −0.037, partial ρ = 0.027 (ns), R² = 0.02
- Elevation kink at ~2,000m: significant for suicide (F=36, p<0.001) and K70 (F=24.5, p<0.001), absent for overdose (F=0.00, p=0.95)

**Gun proxy (firearm suicide fraction):**
- Suicide ρ = 0.291, p < 0.001
- K70 ρ = 0.138, p < 0.001
- Overdose ρ = 0.006, ns
- Regime A median firearm fraction = 0.656 vs. Regime B = 0.605 (p < 0.001)

**Demographics:**
- Poverty: Regime A 13.9%, Regime B 14.3%, p = 0.76 (not significant)
- Median income: $60,867 vs. $64,534, p < 0.001
- % White: 84.7% vs. 78.4%, p < 0.001
- Median age: 42.4 vs. 40.6, p < 0.001

**Temporal trajectory (BV I, suicide × overdose):**
- 1999–2005: 0.177 (p = 0.001)
- 2006–2012: 0.277 (p = 0.001) ← peak co-clustering
- 2013–2019: 0.076 (p = 0.001)
- 2018–2024: 0.016 (p = 0.15, ns)

**Spatial regression:**
- Suicide model: OLS R² = 0.42, spatial error λ = 0.41, all predictors significant
- Overdose model: OLS R² = 0.41, spatial error λ = 0.66
- 2006–12 overdose rate predicting 2018–24 overdose: z = 21.2 (p < 0.001)
- 2006–12 overdose rate predicting 2018–24 suicide: R² = 0.22 (vs. 0.42 for dedicated model)

---

## What to avoid

- Don't say "p = 0.15 means not statistically significant" — frame BV I = 0.016 as Bayesian evidence for independence, not a failed test
- Don't editorialize about Case & Deaton — they were working with real data; the world changed after their publication
- Don't oversell the AIAN finding — n = 48 counties, results are directionally consistent but underpowered
- Don't claim the poverty null means poverty doesn't matter to despair generally — it means poverty doesn't *differentiate* the two regimes
- These are crude rates, not age-adjusted — Regime A is older, which is a potential confound worth acknowledging briefly
