#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:58:34 2020

@author: Kabelo McKabuza
"""


"""
WISE cross-match + k-corrected rest-22µm SFRs via 4-band WISE SED fitting (Option A)

Fix included for Astropy UnitTypeError (deg2):
- VizieR RAJ2000/DEJ2000 columns can already carry degree units after vstack.
- We therefore coerce coordinates to plain float degrees before attaching u.deg.

Pipeline (per cluster field):
1) Read MGCLS compact catalog, filter sources by Field (= cluster_name)
2) Query AllWISE (VizieR II/328/allwise) around each radio position
3) Keep matches within 2 arcsec
4) Store matched W1–W4 mags
5) Fit a simple 2-component SED to W1–W4 (f_nu) and evaluate at λ_obs = 22(1+z) µm
   to estimate rest-frame 22µm νLν
6) Convert νLν(rest 22µm) to SFR via Cluver+2017
7) Save outputs: <cluster>_offsets_copy.csv and <cluster>_sfr.csv
"""

import os
import time
import random
import numpy as np
import pandas as pd
import astropy.units as u

from tqdm import tqdm
from scipy.optimize import least_squares
from astropy.constants import c
from astropy.table import Table, vstack
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo


# -----------------------------
# Settings
# -----------------------------
vizier_catalog = "II/328/allwise"
chunk_size = 100

input_file = "/Users/jason/Downloads/Table2_MGCLS_compactcat_DR1.csv"
output_dir = "./Output/"
os.makedirs(output_dir, exist_ok=True)

# Ask Vizier only for the columns we need (faster + smaller payloads)
Vizier.columns = ["RAJ2000", "DEJ2000", "W1mag", "W2mag", "W3mag", "W4mag"]

# -----------------------------
# Clusters and redshifts
# -----------------------------
clusters = {
    "Abell 133": 0.057,
    "Abell 141": 0.2300,
    "Abell 68": 0.2546,
    "Abell 194": 0.018,
    "Abell 209": 0.21,
    "Abell 22": 0.206,
    "Abell 2485": 0.247,
    "Abell 2597": 0.085,
    "Abell 2645": 0.251,
    "Abell 2667": 0.230,
    "Abell 2744": 0.308,
    "Abell 2751": 0.107,
    "Abell 2811": 0.108,
    "Abell 2895": 0.227,
    "Abell 3365": 0.093,
    "Abell 3376": 0.046,
    "Abell 33": 0.280,
    "Abell 3558": 0.048,
    "Abell 3562": 0.049,
    "Abell 3667": 0.056,
    "Abell 370": 0.375,
    "Abell 4038": 0.028,
    "Abell 521": 0.253,
    "Abell 545": 0.154,
    "Abell 548": 0.042,
    "Abell 85": 0.055,
    "Abell S1063": 0.348,
    "Abell S1121": 0.190,
    "Abell S295": 0.300,
    "ElGordo": 0.870,
    "J0014.3-6604": 0.155,
    "J0027.3-5015": 0.145,
    "J0051.1-4833": 0.187,
    "J0108.5-4020": 0.143,
    "J0117.8-5455": 0.251,
    "J0145.0-5300": 0.188,
    "J0145.2-6033": 0.184,
    "J0212.8-4707": 0.115,
    "J0216.3-4816": 0.163,
    "J0217.2-5244": 0.343,
    "J0225.9-4154": 0.220,
    "J0232.2-4420": 0.284,
    "J0303.7-7752": 0.274,
    "J0314.3-4525": 0.073,
    "J0317.9-4414": 0.075,
    "J0328.6-5542": 0.086,
    "J0336.3-4037": 0.062,
    "J0342.8-5338": 0.060,
    "J0351.1-8212": 0.061,
    "J0352.4-7401": 0.127,
    "J0406.7-7116": 0.229,
    "J0416.7-5525": 0.365,
    "J0431.4-6126": 0.059,
    "J0449.9-4440": 0.172,
    "J0510.2-4519": 0.200,
    "J0516.6-5430": 0.297,
    "J0525.8-4715": 0.191,
    "J0528.9-3927": 0.284,
    "J0540.1-4050": 0.036,
    "J0540.1-4322": 0.085,
    "J0542.8-4100": 0.640,
    "J0543.4-4430": 0.164,
    "J0545.5-4756": 0.130,
    "J0600.8-5835": 0.037,
    "J0607.0-4928": 0.056,
    "J0610.5-4848": 0.243,
    "J0616.8-4748": 0.116,
    "J0625.2-5521": 0.121,
    "J0626.3-5341": 0.051,
    "J0627.2-5428": 0.051,
    "J0631.3-5610": 0.054,
    "J0637.3-4828": 0.203,
    "J0638.7-5358": 0.233,
    "J0645.4-5413": 0.167,
    "J0658.5-5556": 0.296,
    "J0712.0-6030": 0.032,
    "J0738.1-7506": 0.111,
    "J0745.1-5404": 0.074,
    "J0757.7-5315": 0.043,
    "J0812.5-5714": 0.062,
    "J0820.9-5704": 0.061,
    "J0943.4-7619": 0.199,
    "J0948.6-8327": 0.198,
    "J1040.7-7047": 0.061,
    "J1130.0-4213": 0.155,
    "J1145.6-5420": 0.155,
    "J1201.0-4623": 0.118,
    "J1240.2-4825": 0.152,
    "J1358.9-4750": 0.074,
    "J1410.4-4246": 0.049,
    "J1423.7-5412": 0.300,
    "J1518.3-4632": 0.056,
    "J1535.1-4658": 0.036,
    "J1539.5-8335": 0.073,
    "J1601.7-7544": 0.153,
    "J1645.4-7334": 0.069,
    "J1653.0-5943": 0.048,
    "J1705.1-8210": 0.074,
    "J1840.6-7709": 0.019,
    "J2023.4-5535": 0.232,
    "J2104.9-8243": 0.097,
    "J2222.2-5235": 0.174,
    "J2319.2-6750": 0.029,
    "J2340.1-8510": 0.193,
    "MACSJ 0025.4-1222B": 0.584,
    "MACS J0257.6-2209": 0.322,
    "MACSJ0417.5-1155": 0.440,
    "PLCK G200.9-28.2": 0.220,
    "RXCJ0225.1-22928": 0.060,
    "RXCJ0510.7-0801": 0.220,
}


# -----------------------------
# Option A: WISE 4-band SED fit -> k-corrected rest 22µm νLν -> SFR
# -----------------------------
WISE_LAM_UM = np.array([3.4, 4.6, 12.0, 22.0], dtype=float)
WISE_F0_JY = np.array([309.540, 171.787, 31.674, 8.283], dtype=float)  # W1..W4 Vega ZPs


def wise_mags_to_fluxes_jy(w1, w2, w3, w4):
    mags = np.array([w1, w2, w3, w4], dtype=float)
    fluxes = np.full(4, np.nan, dtype=float)
    finite = np.isfinite(mags)
    if finite.any():
        fluxes[finite] = WISE_F0_JY[finite] * 10.0 ** (-mags[finite] / 2.5)
    return fluxes, np.isfinite(fluxes)


def model_fnu_jy(lam_um, A_rj, B_mir, alpha_mir):
    """
    f_nu model = RJ tail + MIR power-law, in terms of wavelength:
      RJ ~ nu^2  -> (lam0/lam)^2
      MIR ~ nu^alpha -> (lam0/lam)^alpha
    """
    lam_um = np.array(lam_um, dtype=float)
    lam0_rj = 3.4
    lam0_mir = 22.0
    rj = A_rj * (lam0_rj / lam_um) ** 2
    mir = B_mir * (lam0_mir / lam_um) ** alpha_mir
    return rj + mir


def fit_wise_sed(fluxes_jy, finite_mask):
    lam = WISE_LAM_UM[finite_mask]
    y = fluxes_jy[finite_mask]
    if len(y) < 3:
        return None

    A0 = max(y[0], 1e-6)
    B0 = max(y[-1], 1e-6)
    alpha0 = -1.0

    bounds_lo = [0.0, 0.0, -6.0]
    bounds_hi = [np.inf, np.inf, 4.0]

    def resid(p):
        A_rj, B_mir, alpha = p
        yhat = model_fnu_jy(lam, A_rj, B_mir, alpha)
        # log-space residuals to balance bands
        return np.log10(yhat + 1e-30) - np.log10(y + 1e-30)

    res = least_squares(
        resid,
        x0=[A0, B0, alpha0],
        bounds=(bounds_lo, bounds_hi),
        loss="soft_l1",
    )
    if not res.success:
        return None
    return tuple(res.x)


def nuLnu_rest22_from_sedfit(w1, w2, w3, w4, z):
    """
    Fit observed W1–W4 and evaluate best-fit f_nu at lambda_obs = 22*(1+z) µm.
    Then:
      Lnu_rest = 4π D_L^2 fnu_obs / (1+z)
      nuLnu_rest = nu_rest * Lnu_rest
    Returns nuLnu_rest in erg/s.
    """
    fluxes, finite = wise_mags_to_fluxes_jy(w1, w2, w3, w4)
    params = fit_wise_sed(fluxes, finite)
    if params is None:
        return np.nan

    A_rj, B_mir, alpha = params

    lam_obs = 22.0 * (1.0 + z)  # micron (observed)
    fnu_obs_jy = model_fnu_jy(lam_obs, A_rj, B_mir, alpha)
    if not np.isfinite(fnu_obs_jy) or fnu_obs_jy <= 0:
        return np.nan

    fnu_obs_cgs = fnu_obs_jy * 1e-23  # Jy -> erg/s/cm^2/Hz
    D_L_cm = cosmo.luminosity_distance(z).to(u.cm).value

    Lnu_rest = 4.0 * np.pi * D_L_cm**2 * fnu_obs_cgs / (1.0 + z)
    nu_rest_hz = (c / (22.0 * u.micron)).to(u.Hz).value

    return nu_rest_hz * Lnu_rest


def sfr_cluver17_from_nuLnu(nuLnu_erg_s):
    return 2.04e-43 * nuLnu_erg_s


# -----------------------------
# Vizier query helper (retry)
# -----------------------------
def query_with_retry(coord, retries=5, base_delay=1.0, max_delay=30.0):
    for attempt in range(retries):
        try:
            result = Vizier.query_region(coord, radius=4 * u.arcsec, catalog=vizier_catalog)
            time.sleep(0.1)  # polite delay
            return result
        except Exception as e:
            wait = min(max_delay, base_delay * (2**attempt) + random.uniform(0, 1))
            print(f"Retry {attempt+1}/{retries} after error: {e}. Waiting {wait:.1f}s.")
            time.sleep(wait)
    print(f"All {retries} retries failed for coordinate {coord}.")
    return None


def safe_float(x):
    """Convert VizieR/astropy table values into float, returning np.nan on failure."""
    try:
        if x is None:
            return np.nan
        # Handle masked values
        if hasattr(x, "mask") and x.mask:
            return np.nan
        # Handle strings like '--'
        if isinstance(x, str) and x.strip() in ("--", ""):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def to_deg_float(col):
    """
    Robustly convert an astropy column/quantity to plain float degrees.
    This prevents deg^2 errors (double-applying units).
    """
    arr = np.asarray(col)
    # If it's a Quantity with units, use .to_value(u.deg) when possible
    try:
        return np.asarray(col.to_value(u.deg), dtype=float)
    except Exception:
        return np.asarray(arr, dtype=float)


# -----------------------------
# Main execution
# -----------------------------
full_table = Table.read(input_file)

for cluster_name, z in clusters.items():
    print(f"\nProcessing cluster: {cluster_name} (z={z})")
    start_time = time.time()

    subset = full_table[full_table["Field"] == cluster_name]
    if len(subset) == 0:
        print(f"No sources found for {cluster_name}. Skipping.")
        continue

    all_entries = []

    for start_idx in range(0, len(subset), chunk_size):
        chunk = subset[start_idx : start_idx + chunk_size]
        radio_coords = SkyCoord(ra=chunk["RA_deg"] * u.deg, dec=chunk["Dec_deg"] * u.deg)

        match_tables = []

        print(f"Processing {len(radio_coords)} rows for {cluster_name}...")
        for coord in tqdm(radio_coords, desc=f"{cluster_name} chunk {start_idx}"):
            result = query_with_retry(coord)
            if result and len(result) > 0 and len(result[0]) > 0:
                r = result[0].copy()
                r["match_RA_deg"] = np.full(len(r), coord.ra.deg)
                r["match_Dec_deg"] = np.full(len(r), coord.dec.deg)
                match_tables.append(r)

        if not match_tables:
            continue

        combined = vstack(match_tables, metadata_conflicts="silent")

        # --- FIX: coerce to float degrees BEFORE multiplying by u.deg ---
        ra_wise_deg = to_deg_float(combined["RAJ2000"])
        dec_wise_deg = to_deg_float(combined["DEJ2000"])
        ra_match_deg = to_deg_float(combined["match_RA_deg"])
        dec_match_deg = to_deg_float(combined["match_Dec_deg"])

        wise_coords = SkyCoord(ra=ra_wise_deg * u.deg, dec=dec_wise_deg * u.deg)
        match_coords = SkyCoord(ra=ra_match_deg * u.deg, dec=dec_match_deg * u.deg)

        sep = match_coords.separation(wise_coords)
        keep = sep <= 2 * u.arcsec
        matched = combined[keep]

        if len(matched) == 0:
            continue

        x_off = (to_deg_float(matched["RAJ2000"]) - to_deg_float(matched["match_RA_deg"])) * 3600.0
        y_off = (to_deg_float(matched["DEJ2000"]) - to_deg_float(matched["match_Dec_deg"])) * 3600.0

        for i in range(len(matched)):
            w1 = safe_float(matched["W1mag"][i]) if "W1mag" in matched.colnames else np.nan
            w2 = safe_float(matched["W2mag"][i]) if "W2mag" in matched.colnames else np.nan
            w3 = safe_float(matched["W3mag"][i]) if "W3mag" in matched.colnames else np.nan
            w4 = safe_float(matched["W4mag"][i]) if "W4mag" in matched.colnames else np.nan

            ra_wise = safe_float(matched["RAJ2000"][i])
            dec_wise = safe_float(matched["DEJ2000"][i])

            all_entries.append(
                [
                    cluster_name,
                    round(float(x_off[i]), 3),
                    np.nan,
                    round(float(y_off[i]), 3),
                    np.nan,
                    ra_wise,
                    dec_wise,
                    w1,
                    w2,
                    w3,
                    w4,
                ]
            )

    if not all_entries:
        print(f"No valid matches for {cluster_name}. Skipping SFR calculation.")
        continue

    colnames = [
        "Field",
        "RAdiff_median (arcsec)",
        "RAdiff_median_Err (arcsec)",
        "DECdiff_median (arcsec)",
        "DECdiff_median_Err (arcsec)",
        "WISE_RA_deg",
        "WISE_Dec_deg",
        "W1mag",
        "W2mag",
        "W3mag",
        "W4mag",
    ]

    offsets_table = Table(rows=all_entries, names=colnames)

    cluster_basename = cluster_name.replace(" ", "_")
    offsets_path = os.path.join(output_dir, f"{cluster_basename}_offsets_copy.csv")
    sfr_path = os.path.join(output_dir, f"{cluster_basename}_sfr.csv")

    offsets_table.write(offsets_path, format="ascii.csv", overwrite=True)
    print(f"Offsets saved to {offsets_path}")

    # --- Compute SFRs using Option A (SED fit + k-correction) ---
    df = pd.read_csv(offsets_path)

    df["nuLnu_rest22_erg_s"] = df.apply(
        lambda row: nuLnu_rest22_from_sedfit(
            row["W1mag"], row["W2mag"], row["W3mag"], row["W4mag"], z
        ),
        axis=1,
    )

    df["SFR_Msun_per_yr"] = df["nuLnu_rest22_erg_s"].apply(
        lambda val: sfr_cluver17_from_nuLnu(val) if np.isfinite(val) and val > 0 else np.nan
    )

    df.to_csv(sfr_path, index=False)
    print(f"SFRs saved to {sfr_path}")
    print(f"Finished {cluster_name} in {(time.time() - start_time) / 60:.2f} minutes")

print("\nAll clusters processed.")




