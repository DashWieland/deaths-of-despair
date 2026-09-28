"""Figures for the 2026 findings write-up (REPORT.md), from out/revisit/*.csv.

Plain matplotlib, one idea per figure. County maps colour only counties with
>= 10 deaths in the window: NCHS forbids showing a count or rate based on
fewer, so smaller counties are drawn as "too few deaths to show".

Writes figures/revisit/fig*.png
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, OUT, county_shapes
from wonder import WINDOWS, county

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})
PAIR_STYLE = {"suicide x overdose": ("#b2182b", "-"), "suicide x k70": ("#2166ac", "-"), "overdose x k70": ("#7f7f7f", "--")}
LABEL = {"suicide x overdose": "suicide × overdose", "suicide x k70": "suicide × alcoholic liver disease",
         "overdose x k70": "overdose × alcoholic liver disease"}
MID = {w: np.mean([int(x) for x in w.split("-")]) for w in WINDOWS}


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIGURES / name)


def fig1_life_cycle():
    d = pd.read_csv(OUT / "r02_life_cycle.csv")
    ov = pd.read_csv(OUT / "r06_overlap.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for pair, (color, ls) in PAIR_STYLE.items():
        for sample, alpha, mk in [("panel", 1.0, "o"), ("window", 0.45, "")]:
            s = d[(d.pair == pair) & (d["sample"] == sample)]
            x = s.window.map(MID)
            axes[0].plot(x, s.spearman, color=color, ls=ls, alpha=alpha, marker=mk, ms=3,
                         label=f"{LABEL[pair]}" if sample == "panel" else None)
            axes[1].plot(x, s.bv_I, color=color, ls=ls, alpha=alpha, marker=mk, ms=3)
        o = ov[ov.pair == pair]
        axes[2].plot(o.window.map(MID), o.ratio, color=color, ls=ls, marker="o", ms=3)
    axes[0].set_title("Rank correlation across counties (ρ)")
    axes[1].set_title("Bivariate Moran's I (spatial)")
    axes[2].set_title("Both-high counties vs. chance")
    axes[2].axhline(1, color="black", lw=0.6)
    axes[2].text(1999.5, 1.02, "chance", fontsize=7)
    for ax in axes:
        ax.axvspan(1999, 2013, color="#e6d8a8", alpha=0.35, lw=0)
        ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2024])
    axes[0].text(2000, axes[0].get_ylim()[1] * 0.97, "Case & Deaton's data\n1999–2013", fontsize=7, va="top")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.text(0.01, -0.04, "Solid = the same 394–402 counties in every window; faint = every county with ≥10 deaths of both causes "
             "in that window. Three-year windows (1999–2000 is two). CDC WONDER, crude rates.", fontsize=7)
    save(fig, "fig1_life_cycle.png")


def fig2_maps():
    shapes = county_shapes()
    states = shapes.dissolve(by="STATEFP")
    wins = ["2007-2009", "2016-2018", "2022-2024"]
    causes = [("suicide", "Suicide"), ("overdose", "Drug overdose"), ("k70", "Alcoholic liver disease")]
    fig, axes = plt.subplots(3, 3, figsize=(11, 6.6))
    for i, (c, title) in enumerate(causes):
        for j, w in enumerate(wins):
            ax = axes[i, j]
            d = shapes.merge(county(c, w), on="fips", how="left")
            ok = d.deaths >= 10
            d.loc[~ok].plot(ax=ax, color="#eeeeee", linewidth=0)
            q = pd.qcut(d.loc[ok, "rate"], 5, labels=False)
            d.loc[ok].assign(q=q).plot(ax=ax, column="q", cmap="YlOrRd", linewidth=0, vmin=0, vmax=4)
            states.boundary.plot(ax=ax, color="white", linewidth=0.3)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(w.replace("-", "–"))
            if j == 0:
                ax.text(-0.02, 0.5, title, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=9, weight="bold")
    fig.text(0.01, 0.0, "Each map: county rate in fifths (light = lowest fifth, dark red = highest), among counties with ≥10 deaths. "
             "Grey = too few deaths to show (NCHS rule). CDC WONDER.", fontsize=7)
    save(fig, "fig2_maps.png")


def fig3_supply():
    d = pd.read_csv(OUT / "r05_supply.csv")
    a = d[d.analysis == "pills_vs_cause"]
    b = d[d.analysis == "suicide_x_overdose_given_pills"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for c, color, label in [("overdose", "#b2182b", "overdose"), ("suicide", "#2166ac", "suicide"), ("k70", "#7f7f7f", "alcoholic liver disease")]:
        s = a[a.cause == c]
        axes[0].plot(s.window.map(MID), s.rho_pills_2006_12, color=color, marker="o", ms=3, label=label)
    axes[0].set_title("Pills shipped 2006–12 vs. each cause (ρ)")
    axes[0].legend(fontsize=7)
    axes[1].plot(b.window.map(MID), b.rho_raw, color="#b2182b", marker="o", ms=3, label="suicide × overdose")
    axes[1].plot(b.window.map(MID), b.rho_partial_pills, color="#b2182b", ls="--", marker="o", ms=3,
                 label="…with pills per resident held constant")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title("How much of the overlap runs through the pills")
    axes[1].legend(fontsize=7, loc="lower left")
    for ax in axes:
        ax.axvspan(2006, 2012, color="#e6d8a8", alpha=0.35, lw=0)
    fig.text(0.01, -0.05, "Pills = oxycodone + hydrocodone dosage units shipped per resident per year, 2006–2012. "
             "Source: The Washington Post, DEA ARCOS; CDC WONDER.", fontsize=7)
    save(fig, "fig3_supply.png")


def fig4_regions():
    d = pd.read_csv(OUT / "r09_recent_waves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    colors = {"West": "#b2182b", "Northeast": "#2166ac", "Midwest": "#4d9221", "South": "#7f7f7f"}
    for r, color in colors.items():
        s = d[d.region == r]
        axes[0].plot(s.year, s.all_drugs_rate, color=color, marker="o", ms=3, label=r)
        axes[1].plot(s.year, s.synthetic_opioids_rate, color=color, marker="o", ms=3)
    axes[0].set_title("Drug overdose deaths per 100,000")
    axes[1].set_title("…involving synthetic opioids (mostly fentanyl)")
    axes[0].legend(fontsize=7)
    fig.text(0.01, -0.05, "Census regions. 2024–2025 provisional (CDC VSRR, predicted counts). Rates use Census population "
             "estimates (2025 uses 2024).", fontsize=7)
    save(fig, "fig4_regions.png")


def fig5_robustness():
    r4 = pd.read_csv(OUT / "r04_age.csv")
    r3 = pd.read_csv(OUT / "r03_aian.csv")
    r7 = pd.read_csv(OUT / "r07_definitions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    so = r4[r4.pair == "suicide x overdose"]
    x = so.window.map(MID)
    crude, adj = so.crude_on_isr_set, so.isr  # one indirect method in every window
    axes[0].plot(x, crude, color="#b2182b", marker="o", ms=3, label="crude")
    axes[0].plot(x, adj, color="#b2182b", ls="--", marker="o", ms=3, label="age-adjusted")
    axes[0].set_title("Age-adjusted (suicide × overdose)")
    axes[0].legend(fontsize=7)
    sk = r3[r3.pair == "suicide x k70"]
    for col, ls, label in [("rho_all", "-", "all counties"), ("rho_no_aian5", "--", "without counties ≥5% AIAN"),
                           ("rho_partial_aian", ":", "AIAN share held constant")]:
        axes[1].plot(sk.window.map(MID), sk[col], color="#2166ac", ls=ls, marker="o", ms=3, label=label)
    axes[1].set_title("AIAN counties (suicide × liver)")
    axes[1].legend(fontsize=7)
    coarse_mid = {"1999-2005": 2002, "2006-2012": 2009, "2013-2018": 2015.5, "2019-2024": 2021.5}
    for series, color, ls in [("primary: suicide x overdose", "#b2182b", "-"), ("C&D: suicide x poisonings", "#b2182b", "--"),
                              ("ages 25-64: suicide x overdose", "#b2182b", ":")]:
        s = r7[r7.series == series]
        axes[2].plot(s.window.map(coarse_mid), s.spearman, color=color, ls=ls, marker="o", ms=3, label=series.split(":")[0])
    axes[2].set_title("C&D definitions, ages 25–64")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_ylim(-0.2, 0.8)
        ax.axhline(0, color="black", lw=0.5)
    fig.text(0.01, -0.05, "Rank correlations (ρ) across counties with ≥10 deaths of both causes. Age adjustment: indirect "
             "standardisation in every window (it ranks counties like WONDER's direct rates, ρ ≥ 0.987).", fontsize=7)
    save(fig, "fig5_robustness.png")


def fig6_burden():
    d = pd.read_csv(OUT / "r06_burden.csv")
    s = d[d.window == "2022-2024"].set_index("cause").loc[["suicide", "overdose", "k70"]]
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    y = np.arange(3)
    ax.barh(y + 0.18, s.deaths_share_in_top_fifth, height=0.34, color="#b2182b", label="share of the cause's deaths")
    ax.barh(y - 0.18, s.population_share_in_top_fifth, height=0.34, color="#bbbbbb", label="share of the population")
    ax.set_yticks(y, ["suicide", "overdose", "alcoholic liver disease"])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("The fifth of counties with the highest rates, 2022–2024")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "fig6_burden.png")


def fig7_waves():
    w = pd.read_csv(OUT / "r11_waves.csv")
    c = pd.read_csv(OUT / "r11_state_corr.csv")
    drugs = [("rx_opioids", "prescription-type opioids", "#7f7f7f"), ("heroin", "heroin", "#2166ac"),
             ("synthetic_opioids", "fentanyl & other synthetics", "#b2182b"), ("psychostimulants", "methamphetamine & other stimulants", "#4d9221"),
             ("cocaine", "cocaine", "#e08214")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    us = w[w.region == "US"]
    for key, label, color in drugs:
        s = us[us.measure == key]
        axes[0].plot(s.year, s.rate, color=color, marker="o", ms=2.5, label=label)
    axes[0].set_title("Overdose deaths per 100,000, by drug involved")
    axes[0].legend(fontsize=7)
    for key, label, color in [("overdose_ucd", "all drug overdoses", "black")] + drugs:
        s = c[c.measure == key]
        axes[1].plot(s.year, s.spearman, color=color, marker="o", ms=2.5, lw=2 if key == "overdose_ucd" else 1.2,
                     label=label if key == "overdose_ucd" else None)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].legend(fontsize=7, loc="lower left")
    axes[1].set_title("Across states: suicide rate vs. each drug (ρ)")
    fig.text(0.01, -0.05, "Deaths from drug overdose (CDC definition) involving each drug; a death involving two drugs counts twice. "
             "Right: Spearman correlation across states of age-adjusted rates, year by year. CDC WONDER, multiple cause of death.",
             fontsize=7)
    save(fig, "fig7_waves.png")


if __name__ == "__main__":
    fig1_life_cycle(); fig2_maps(); fig3_supply(); fig4_regions(); fig5_robustness(); fig6_burden(); fig7_waves()
