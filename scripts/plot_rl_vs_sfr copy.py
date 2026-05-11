#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Radio Luminosity vs Star Formation Rate
---------------------------------------
Separate populations:
- Non-AGN, non-radio-excess: blue circles
- AGN: red triangles
- Radio-excess (non-AGN): green squares

Radio-excess is treated as a distinct population, not an overlay.

Edited version:
- keeps the original main panel
- adds a lower offset sub-panel
- uses only the best-fit offset in the bottom panel
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import linregress

# Serif fonts for ApJ-style figures
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'

try:
    from astropy.table import Table
except Exception:
    Table = None


# =========================
# IDE DEFAULTS
# =========================
INPUT_FILE = "merged_matched.csv"   # <-- merged Delvecchio table
OUTFIG = "rl_vs_sfr_showcase"
# =========================


def load_table(path, fits_hdu=None):
    path = Path(path)
    if path.suffix.lower() in ['.csv', '.tsv']:
        return pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_table(path)
    elif path.suffix.lower() in ['.fits', '.fit', '.fz']:
        if Table is None:
            raise RuntimeError("Install astropy to read FITS files")
        return Table.read(str(path), hdu=fits_hdu).to_pandas()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def bin_median_iqr(values, x, edges, min_per_bin=5):
    meds, p16s, p84s = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() < min_per_bin:
            meds.append(np.nan)
            p16s.append(np.nan)
            p84s.append(np.nan)
        else:
            meds.append(np.median(values[mask]))
            p16s.append(np.percentile(values[mask], 16))
            p84s.append(np.percentile(values[mask], 84))
    return np.array(meds), np.array(p16s), np.array(p84s)


def main():

    df = load_table(INPUT_FILE)

    # ---- column detection ----
    logL_col = pick_col(df, [
        'logL_1p4GHz', 'logL1p4', 'log10_L1p4', 'L1p4_log',
        'log10_L_1.4GHz', 'Lradio_21cm'
    ])

    logSFR_col = pick_col(df, [
        'logSFR', 'log10_SFR', 'log10SFR', 'log_SFR'
    ])

    sfr_col = pick_col(df, [
        'SFR_IR', 'SFR', 'SFR_total'
    ])

    if logL_col is None:
        raise KeyError("No log10 radio luminosity column found.")

    if logSFR_col is None and sfr_col is None:
        raise KeyError("No SFR column found.")

    # ---- build arrays ----
    logL = pd.to_numeric(df[logL_col], errors='coerce').values

    if logSFR_col is not None:
        logSFR = pd.to_numeric(df[logSFR_col], errors='coerce').values
    else:
        sfr = pd.to_numeric(df[sfr_col], errors='coerce').values
        logSFR = np.full_like(sfr, np.nan, dtype=float)
        ok = np.isfinite(sfr) & (sfr > 0)
        logSFR[ok] = np.log10(sfr[ok])

    agn_cols = ['XRAY_AGN_my', 'MIR_AGN_my', 'SED_AGN_my']
    existing_agn_cols = [c for c in agn_cols if c in df.columns]
    if existing_agn_cols:
        print(df[existing_agn_cols].sum(numeric_only=True))
        print("Any AGN row?", df[existing_agn_cols].any(axis=1).any())
        print(df[existing_agn_cols].head(10))

    mfin = np.isfinite(logL) & np.isfinite(logSFR)
    logL, logSFR = logL[mfin], logSFR[mfin]
    df = df.loc[mfin].reset_index(drop=True)

    # ---- AGN flag ----
    agn_cols = [c for c in ['XRAY_AGN_my', 'MIR_AGN_my', 'SED_AGN_my'] if c in df.columns]
    if agn_cols:
        agn_flag = df[agn_cols].any(axis=1).values
    else:
        agn_flag = np.zeros(len(df), dtype=bool)

    # ---- Radio-excess flag (from Delvecchio) ----
    re_col = pick_col(df, ['Radio_excess', 'Radio_excess_delv'])
    if re_col is not None:
        radio_excess_flag = pd.to_numeric(df[re_col], errors='coerce').fillna(0).astype(int).astype(bool).values
    else:
        radio_excess_flag = np.zeros(len(df), dtype=bool)

    # ---- population masks (exclusive) ----
    mask_agn = agn_flag
    mask_radio_excess = radio_excess_flag & ~agn_flag
    mask_normal = ~agn_flag & ~radio_excess_flag

    print(f"Normal (SF): {mask_normal.sum()}")
    print(f"Radio-excess (non-AGN): {mask_radio_excess.sum()}")
    print(f"AGN: {mask_agn.sum()}")

    # ---- Fit only non-AGN ----
    mask_fit = ~mask_agn
    x_fit = logL[mask_fit]
    y_fit = logSFR[mask_fit]

    if x_fit.size < 3 or np.allclose(x_fit, x_fit[0]):
        raise RuntimeError("Not enough non-AGN points (or zero x-variance) to perform a fit.")

    lr = linregress(x_fit, y_fit)
    m_fit = lr.slope
    b_fit = lr.intercept
    m_err = lr.stderr
    b_err = lr.intercept_stderr
    r = lr.rvalue
    r2 = r**2

    # keep your original broad band style
    sigma = 1 * np.std(y_fit - (m_fit * x_fit + b_fit))

    print("\nNon-AGN fit (excluding AGN):")
    print(f"  N = {x_fit.size}")
    print(f"  m = {m_fit:.3f} ± {m_err:.3f}")
    print(f"  b = {b_fit:.3f} ± {b_err:.3f}")
    print(f"  Pearson r = {r:.3f}")
    print(f"  R^2 = {r2:.3f}")
    print(f"  p-value = {lr.pvalue:.3e}")
    print(f"  σ_resid = {sigma:.3f} dex")

    # ---- Main-panel lines ----
    xgrid = np.linspace(np.nanmin(logL), np.nanmax(logL), 200)
    y_line = m_fit * xgrid + b_fit
    y_upper = y_line + sigma
    y_lower = y_line - sigma

    # ---- Best-fit offset only ----
    y_pred_bestfit = m_fit * logL + b_fit
    offset_bestfit = y_pred_bestfit - logSFR

    n_bins = 15
    bin_edges = np.linspace(np.nanmin(logL), np.nanmax(logL), n_bins + 1)
    bin_ctrs = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    med_bestfit, p16_bestfit, p84_bestfit = bin_median_iqr(offset_bestfit, logL, bin_edges)

    # ---- plot with added sub-panel ----
    fig = plt.figure(figsize=(5.2, 5.8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1.0], hspace=0.05)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # -------------------------
    # TOP PANEL = original plot
    # -------------------------

    ax_top.scatter(
        logL[mask_normal], logSFR[mask_normal],
        s=12, color='#1A85FF', alpha=0.45,
        label='Star-forming'
    )

    ax_top.scatter(
        logL[mask_radio_excess], logSFR[mask_radio_excess],
        s=15, color='#019C5F', marker='s', alpha=0.40,
        label='Radio-excess'
    )

    ax_top.scatter(
        logL[mask_agn], logSFR[mask_agn],
        s=12, color='#D41159', marker='^', alpha=0.45,
        label='AGN'
    )

    ax_top.plot(xgrid, y_line, 'k-', label=f'Best-fit non-AGN (m={m_fit:.2f})')
    ax_top.fill_between(xgrid, y_lower, y_upper, alpha=0.2, color='gray', label=r'$\pm1\sigma$')

    ax_top.set_ylabel(r'$\log_{10}\mathrm{SFR}\ \mathrm{[M_\odot\,yr^{-1}]}$')
    ax_top.grid(alpha=0.3)
    ax_top.legend(
        fontsize=9,
        frameon=True,
        handlelength=1.5,
        handletextpad=0.6,
        labelspacing=0.4,
        markerscale=0.9,
        borderpad=0.6,
        loc='upper left'
    )
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # -------------------------
    # BOTTOM PANEL = best-fit offset only
    # -------------------------

    ax_bot.axhline(0, color='black', lw=1.0, ls=':')
    ax_bot.plot(
        bin_ctrs, med_bestfit,
        'o-', color='black', lw=1.4, ms=4.0,
        label='Best-fit offset'
    )
    ax_bot.fill_between(bin_ctrs, p16_bestfit, p84_bestfit, color='gray', alpha=0.2)

    ax_bot.set_xlabel(r'$\log_{10} L_{1.4\,\mathrm{GHz}}\ \mathrm{[W\,Hz^{-1}]}$')
    ax_bot.set_ylabel(r'$\Delta\log_{10}\mathrm{SFR}$')
    ax_bot.set_ylim(-1.5, 3.0)
    ax_bot.grid(alpha=0.3)
    ax_bot.legend(
        fontsize=8,
        frameon=True,
        handlelength=1.4,
        handletextpad=0.5,
        labelspacing=0.35,
        markerscale=0.9,
        borderpad=0.5,
        loc='upper left'
    )

    plt.tight_layout()
    plt.savefig(f"{OUTFIG}.png", dpi=400, bbox_inches='tight')
    plt.savefig(f"{OUTFIG}.pdf", bbox_inches='tight')
    plt.show()

    print(f"Saved {OUTFIG}.png and {OUTFIG}.pdf")


if __name__ == "__main__":
    main()