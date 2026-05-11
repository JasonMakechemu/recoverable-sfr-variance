#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 12:53:28 2025

@author: jason
"""

'''
Importation of all our functions, including a variety of regression models
and tools from the scikit-learn machine learning library.

We load the .fits file path which contains sources from
the VLA-COSMOS 3 GHz Large Project.
'''

# --- Imports ---
import os
import pwlf
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from astropy.io import fits
from sklearn.svm import SVR
from scipy.constants import c  # Speed of light in m/s
from sklearn.base import clone
from scipy.integrate import quad
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, chi2
from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
from sklearn.pipeline import make_pipeline
from astropy.cosmology import FlatLambdaCDM
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap
from scipy.stats import median_abs_deviation
from astropy.cosmology import Planck18 as cosmo
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error

# ==== Load FITS data ====
fits_file_path = '/Users/jason/Downloads/VLA_3GHz_counterpart_array_20170210_paper_delvecchio_et_al.fits'

with fits.open(fits_file_path) as hdul:
    data_fits = hdul[1].data
    data = pd.DataFrame({col: data_fits[col].byteswap().newbyteorder() for col in data_fits.columns.names})

data = data[(data != -99).all(axis=1)].dropna()

print(data.columns)


'''
Cleaning the dataset by removing rows with invalid or missing values.
'''

# ==== Clean & Format ====
data = pd.DataFrame({
    'Radio_Luminosity_1.4GHz': data['Lradio_21cm'],  # 1.4 GHz radio
    'Radio_Luminosity_3GHz': data['Lradio_10cm'],    # 3 GHz radio
    'IR_Magnitude': data['L_TIR_SF'],                # Infrared magnitude
    'True_SFR': data['SFR_IR'],
    'Photometric_Redshift': data['Z_BEST'],
    'RA': data['RA_VLA3'],
    'Dec': data['DEC_VLA3'],
    'Stellar_Mass': data['Mstar']
})


data = data[(data != -99).all(axis=1)].dropna()

data_clean = data.copy()


'''
For the COSMOS-VLA dataset

This code performs a cosmological calculation and visualization. It computes the radio
flux of astronomical sources based on their observed radio luminosity and redshift

Then plots this calculated flux against redshift, colored by star formation rate (SFR).

For each object in the data:
    
1) Gets the photometric redshift
2) Computes the comoving distance (ΛCDM cosmological model)
3) Converts the radio luminosity from log to linear
4) Calculates radio flux using the inverse square law:
'''

# Cosmological parameters (flat ΛCDM model)
H0 = 67.4 * 1000 / (3.086e22)  # Hubble constant in s^-1 (70 km/s/Mpc)
Omega_m = 0.315
Omega_lambda = 1 - Omega_m


# Hubble parameter as a function of redshift
def H(z):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_lambda)

# Comoving distance integrand
def integrand(z):
    return c / H(z)

# Comoving distance function in meters
def comoving_distance(z):
    distance, _ = quad(integrand, 0, z)
    return distance


radio_flux_array = []

for i in range(len(data)):
    z = data['Photometric_Redshift'][i]
    distance_meters = comoving_distance(z)
    L_radio_linear = 10**(data['Radio_Luminosity_1.4GHz'][i])  # convert log luminosity to linear
    flux = L_radio_linear / (4 * np.pi * distance_meters**2)
    radio_flux_array.append(flux)
    
#Output flux: W m^-2 Hz^-1 (flux density)


# Plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    data['Photometric_Redshift'],
    radio_flux_array,
    c=np.log10(data['True_SFR']),         # Color by true SFR
    cmap='viridis',
    s=10,
    alpha=0.7
)
plt.xlabel('Redshift (z)')
plt.ylabel('Radio Flux (W/m²)')
plt.title('Radio Flux vs Redshift')
plt.yscale('log')

cbar = plt.colorbar(scatter)
cbar.set_label('log SFR M_s/yr')

plt.savefig('radio_flux_vs_redshift.png', dpi=300) 
plt.show()

'''

For the COSMOS-VLA dataset

This block of code is preparing and evaluating multiple machine learning models
to predict the star formation rate (SFR) of galaxies from various combinations of
input features. 

Multiple feature sets are defined:
    
X_full: Includes all available relevant features.

X_radio, X_radio_photo_z, X_1_4GHz, X_3GHz: Subsets for specific experiments.

Target y: log10 of the true star formation rate (True_SFR), making the regression
easier and more normally distributed.


Each feature set is split into training and testing sets (80% training, 20% testing)
for fair evaluation. random_state=42 ensures reproducibility.

We then define a dictionary of six regression models to compare:

Next we train each model on the training set, and then predict on the test set.
Converts both predictions and true values back to linear scale.

Compuntations of the Mean Absolute Error (MAE) in %, and Median Error in % are done next,
followed by printing the performance of each model; and returning results for further
analysis or plotting.

Stellar mass seems to be a  contributing factor to the break in relation between,
1.4GHz radio luminosity and star formation rate

'''

# ==== Features & Targets ====
X_full = data_clean[['Radio_Luminosity_1.4GHz', 'Radio_Luminosity_3GHz', 'IR_Magnitude', 'Photometric_Redshift']]
X_radio = data_clean[['Radio_Luminosity_1.4GHz', 'Radio_Luminosity_3GHz']]
X_1_4GHz_stellar_mass = data_clean[['Radio_Luminosity_1.4GHz', 'Stellar_Mass']]
X_radio_photo_z = data_clean[['Radio_Luminosity_1.4GHz', 'Radio_Luminosity_3GHz', 'Photometric_Redshift']]
X_1_4GHz = data_clean[['Radio_Luminosity_1.4GHz']]
X_3GHz = data_clean[['Radio_Luminosity_3GHz']]
y = np.log10(data_clean['True_SFR'])

# ==== Splits ====
def split_without_shuffle(X, y, train_ratio=0.8):
    split_idx = int(len(y) * train_ratio)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# Now apply it to each dataset
X_train_full, X_test_full, y_train_full, y_test_full = split_without_shuffle(X_full, y)
X_train_radio, X_test_radio, y_train_radio, y_test_radio = split_without_shuffle(X_radio, y)
X_train_radio_photo_z, X_test_radio_photo_z, y_train_radio_photo_z, y_test_radio_photo_z = split_without_shuffle(X_radio_photo_z, y)
X_train_1_4GHz, X_test_1_4GHz, y_train_1_4GHz, y_test_1_4GHz = split_without_shuffle(X_1_4GHz, y)
X_train_3GHz, X_test_3GHz, y_train_3GHz, y_test_3GHz = split_without_shuffle(X_3GHz, y)
X_train_1_4GHz_stellar_mass, X_test_1_4GHz_stellar_mass, y_train_1_4GHz_stellar_mass, y_test_1_4GHz_stellar_mass = split_without_shuffle(X_1_4GHz_stellar_mass, y)

# ==== Standardization ====
scaler_full = StandardScaler().fit(X_train_full)
scaler_radio = StandardScaler().fit(X_train_radio)
scaler_radio_photo_z = StandardScaler().fit(X_train_radio_photo_z)
scaler_1_4GHz = StandardScaler().fit(X_train_1_4GHz)
scaler_3GHz = StandardScaler().fit(X_train_3GHz)

X_train_full_scaled = scaler_full.transform(X_train_full)
X_test_full_scaled = scaler_full.transform(X_test_full)
X_train_radio_scaled = scaler_radio.transform(X_train_radio)
X_test_radio_scaled = scaler_radio.transform(X_test_radio)
X_train_radio_photo_z_scaled = scaler_radio_photo_z.transform(X_train_radio_photo_z)
X_test_radio_photo_z_scaled = scaler_radio_photo_z.transform(X_test_radio_photo_z)
X_train_1_4GHz_scaled = scaler_1_4GHz.transform(X_train_1_4GHz)
X_test_1_4GHz_scaled = scaler_1_4GHz.transform(X_test_1_4GHz)
X_train_3GHz_scaled = scaler_3GHz.transform(X_train_3GHz)
X_test_3GHz_scaled = scaler_3GHz.transform(X_test_3GHz)

scaler_1_4GHz_stellar_mass = StandardScaler().fit(X_train_1_4GHz_stellar_mass)
X_train_1_4GHz_stellar_mass_scaled = scaler_1_4GHz_stellar_mass.transform(X_train_1_4GHz_stellar_mass)
X_test_1_4GHz_stellar_mass_scaled = scaler_1_4GHz_stellar_mass.transform(X_test_1_4GHz_stellar_mass)

# ==== Models ====
model_dict = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'Support Vector Regression': SVR(kernel='rbf'),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

# ==== Evaluation ====

X_sets = {
    #'Full': (X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full),
    #'Radio': (X_train_radio_scaled, X_test_radio_scaled, y_train_radio, y_test_radio),
    #'Radio + Photo-z': (X_train_radio_photo_z_scaled, X_test_radio_photo_z_scaled, y_train_radio_photo_z, y_test_radio_photo_z),
    #'3GHz only': (X_train_3GHz_scaled, X_test_3GHz_scaled, y_train_3GHz, y_test_3GHz),
    #'1.4GHz + Stellar Mass': (X_train_1_4GHz_stellar_mass_scaled, X_test_1_4GHz_stellar_mass_scaled, y_train_1_4GHz_stellar_mass, y_test_1_4GHz_stellar_mass),
    '1.4GHz only': (X_train_1_4GHz_scaled, X_test_1_4GHz_scaled, y_train_1_4GHz, y_test_1_4GHz)
}


for set_name, (X_train, X_test, y_train, y_test) in X_sets.items():
    print(f"\n========== Feature Set: {set_name} ==========")
    for model_name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # R² score in log space
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Linear space for MAE and median errors
        y_train_lin = 10 ** y_train
        y_test_lin = 10 ** y_test
        y_test_pred_lin = 10 ** y_test_pred
        y_train_pred_lin = 10 ** y_train_pred

        mae_test = mean_absolute_error(y_test_lin, y_test_pred_lin)
        med_test = median_absolute_error(y_test_lin, y_test_pred_lin)
        pct_mae = np.mean(np.abs(y_test_pred_lin - y_test_lin) / y_test_lin) * 100
        pct_med = np.median(np.abs(y_test_pred_lin - y_test_lin) / y_test_lin) * 100

        train_mae = mean_absolute_error(y_train_lin, y_train_pred_lin)

        print(f"\n-- {model_name} --")
        print(f"R² (Train): {r2_train:.3f} | R² (Test): {r2_test:.3f}")
        print(f"MAE (Test): {mae_test:.3e} | Median Error: {med_test:.3e}")
        print(f"MAE %: {pct_mae:.2f}% | Median % Error: {pct_med:.2f}%")
        print(f"Train MAE vs Test MAE: {train_mae:.3e} vs {mae_test:.3e}")

        # Residuals
        residuals = y_test - y_test_pred
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, bins=30, kde=True, color='steelblue')
        plt.title(f"Residuals: {model_name} on {set_name}")
        plt.xlabel("Residual (log SFR)")
        plt.tight_layout()
        plt.show()


'''
Saving test set for the COSMOS VLA predictions
'''

# Create a copy of the test features
test_data = X_test_full.copy()

# Add the target variable
test_data['log10_True_SFR'] = y_test_full

# Add RA and Dec by matching the indices from data_clean
test_data['RA'] = data_clean.loc[X_test_full.index, 'RA']
test_data['Dec'] = data_clean.loc[X_test_full.index, 'Dec']

# Reset index if desired (optional)
test_data.reset_index(drop=True, inplace=True)

# Save to CSV
test_data.to_csv('test_dataset.csv', index=False)


'''
For COSMOS-VLA dataset

Trains and evaluates all previously defined machine learning models using the
full feature set. Visualizes each model's predicted vs. true values for log(SFR),
color-coded by user-chosen parameters like redshift or luminosity.

Calls the previously defined function. Stores predictions, MAE %, and median error %
for each model in results_full.

For plotting:
    
In each of our models, we do the following

Creates a subplot (2x3 grid for 6 models).

x-axis: true log(SFR)
y-axis: predicted log(SFR)
Point color: determined by color_by 'example parameter'
Adds:
Diagonal dashed 1-1 line 

Text box: shows MAE % and Median Error %
Colorbar: matches your chosen coloring variable

color_vals is log-transformed if needed (e.g., np.log10(z_vals)).
'''


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_test_linear = 10 ** y_test
        y_pred_linear = 10 ** y_pred

        abs_errors = np.abs(y_pred_linear - y_test_linear)
        mae = np.mean(abs_errors)
        med_err = np.median(abs_errors)

        mae_pct = np.mean(abs_errors / y_test_linear) * 100
        med_pct = np.median(abs_errors / y_test_linear) * 100

        results[name] = (y_test, y_pred, mae_pct, med_pct)
        preds[name] = y_pred_linear  # store linear scale predictions
        print(f"{name}: MAE% = {mae_pct:.2f}%, Median Error% = {med_pct:.2f}%")
    return results, preds

# Train and get predictions (everything greyed out except the 1.4GHz only model).

# Full feature set
'''results_full, preds_full = train_and_evaluate(
    model_dict, X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full)

# Radio luminosities only
results_radio, preds_radio = train_and_evaluate(
    model_dict, X_train_radio_scaled, X_test_radio_scaled, y_train_radio, y_test_radio)

# Radio luminosities + Photometric Redshift
results_radio_photo_z, preds_radio_photo_z = train_and_evaluate(
    model_dict, X_train_radio_photo_z_scaled, X_test_radio_photo_z_scaled, y_train_radio_photo_z, y_test_radio_photo_z)

# 3 GHz only
results_3GHz, preds_3GHz = train_and_evaluate(
    model_dict, X_train_3GHz_scaled, X_test_3GHz_scaled, y_train_3GHz, y_test_3GHz)

#stellar mass also
results_1_4GHz_stellar, preds_1_4GHz_stellar = train_and_evaluate(
    model_dict, X_train_1_4GHz_stellar_mass, X_test_1_4GHz_stellar_mass, y_train_full, y_test_full)
'''
# 1.4 GHz only
results_1_4GHz, preds_1_4GHz = train_and_evaluate(
    model_dict, X_train_1_4GHz_scaled, X_test_1_4GHz_scaled, y_train_1_4GHz, y_test_1_4GHz)

# Example for the 1.4 GHz model's predictions using Neural Network (MLP)
y_pred_linear_1_4GHz = preds_1_4GHz['Neural Network (MLP)']
#y_pred_linear_1_4GHz_stellar_mass = preds_1_4GHz_stellar['Neural Network (MLP)']


# Prepare DataFrame as before, using the 1.4 GHz test set
test_data_1_4GHz = X_test_1_4GHz.copy() #1.4ghz luminosity from the test set
test_data_1_4GHz['log10_True_SFR'] = y_test_1_4GHz
test_data_1_4GHz['RA'] = data_clean.loc[X_test_1_4GHz.index, 'RA']
test_data_1_4GHz['Dec'] = data_clean.loc[X_test_1_4GHz.index, 'Dec']
test_data_1_4GHz['Predicted_SFR'] = y_pred_linear_1_4GHz
test_data_1_4GHz['Redshift'] = data_clean['Photometric_Redshift']

test_data_1_4GHz.reset_index(drop=True, inplace=True)
test_data_1_4GHz.to_csv('test_dataset_1_4GHz_with_predictions.csv', index=False)

'''
inclusion of stellar mass is greyed out

test_data_1_4GHz_stellar = X_test_1_4GHz_stellar_mass.copy()
test_data_1_4GHz_stellar['log10_True_SFR'] = y_test_full
test_data_1_4GHz_stellar['RA'] = data_clean.loc[X_test_1_4GHz_stellar_mass.index, 'RA']
test_data_1_4GHz_stellar['Dec'] = data_clean.loc[X_test_1_4GHz_stellar_mass.index, 'Dec']
test_data_1_4GHz_stellar['Predicted_SFR'] = y_pred_linear_1_4GHz_stellar_mass
test_data_1_4GHz_stellar['Redshift'] = data_clean.loc[X_test_1_4GHz_stellar_mass.index, 'Photometric_Redshift']

test_data_1_4GHz_stellar.reset_index(drop=True, inplace=True)
test_data_1_4GHz_stellar.to_csv('test_dataset_1_4GHz_stellar_with_predictions.csv', index=False)
'''


'''
For COSMOS VLA dataset
'''

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'

def plot_true_vs_pred(results, title, color_by='redshift', bins=20, save_path=None, data_clean=None):
    plt.figure(figsize=(18, 10))

    if data_clean is None:
        raise ValueError("data_clean DataFrame must be provided to the function.")

    z_vals = data_clean['Photometric_Redshift'].values
    sfr_vals = data_clean['True_SFR'].values
    lum_1_4GHz_vals = data_clean['Radio_Luminosity_1.4GHz'].values
    lum_3GHz_vals = data_clean['Radio_Luminosity_3GHz'].values
    ir_mag_vals = data_clean['IR_Magnitude'].values
    stellar_mass_vals = data_clean['Stellar_Mass'].values  # Added stellar mass values

    # Handle coloring logic
    if color_by == 'redshift':
        color_vals = z_vals
        cmap = 'viridis'
        color_label = 'Photometric Redshift'
    elif color_by == 'sfr':
        color_vals = np.log10(sfr_vals)
        cmap = 'plasma'
        color_label = 'True SFR ($M_\odot$/yr)'
    elif color_by == '1.4GHz':
        color_vals = np.log10(lum_1_4GHz_vals)
        cmap = 'inferno'
        color_label = '1.4 GHz Luminosity'
    elif color_by == '3GHz':
        color_vals = np.log10(lum_3GHz_vals)
        cmap = 'cividis'
        color_label = '3 GHz Luminosity'
    elif color_by == 'IR':
        color_vals = np.log10(ir_mag_vals)
        cmap = 'magma'
        color_label = 'IR Magnitude ($L_{TIR_{SF}}$)'
    elif color_by == 'stellar_mass':
        color_vals = np.log10(stellar_mass_vals)
        cmap = 'magma'
        color_label = 'Stellar Mass ($M_\odot$)'
    elif color_by == 'none':
        color_vals = None
        cmap = None
        color_label = None
    else:
        raise ValueError("color_by must be 'redshift', 'sfr', '1.4GHz', '3GHz', 'IR', 'stellar_mass', or 'none'")

    for i, (name, (true, pred, mae_pct, med_pct)) in enumerate(results.items(), 1):
        df = pd.DataFrame({'true': true, 'pred': pred})
        df['residual'] = df['pred'] - df['true']
        df['bin'] = pd.qcut(df['true'], q=bins, duplicates='drop')

        plt.subplot(2, 3, i)

        if color_by == 'none':
            sc = plt.scatter(df['true'], df['pred'], color='gray', alpha=0.6, marker='o')
        else:
            true_linear = 10 ** df['true'].values
            idxs = [np.argmin(np.abs(data_clean['True_SFR'].values - val)) for val in true_linear]
            color_data = np.array(color_vals)[idxs]
            sc = plt.scatter(df['true'], df['pred'], c=color_data, cmap=cmap, alpha=0.6, marker='o')

        plt.plot([min(df['true']), max(df['true'])], [min(df['true']), max(df['true'])], 'k--', lw=2)

        plt.title(name)
        plt.xlabel(r'True $\log(\mathrm{SFR})$ $\log(M_\odot\,\mathrm{yr}^{-1})$', fontsize=14)
        plt.ylabel(r'Predicted $\log(\mathrm{SFR})$ $\log(M_\odot\,\mathrm{yr}^{-1})$', fontsize=14)

        plt.grid(True)
        plt.text(0.95, 0.95, f'MAE: {mae_pct:.2f}%\nMedian Error: {med_pct:.2f}%',
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.7))

        if color_by != 'none':
            plt.colorbar(sc, label=color_label)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


plot_true_vs_pred(results_1_4GHz, 
                  title="Predicted vs True SFR", 
                  color_by='none', 
                  save_path='pred_vs_true_sfr.png',
                  data_clean=data_clean)

'''
inclusion of stellar mass is greyed out

plot_true_vs_pred(results_1_4GHz_stellar,
                   title="Predicted vs True SFR (Colored by Stellar Mass)",
                   color_by='stellar_mass',
                   save_path='pred_vs_true_sfr_stellar_mass.png',
                   data_clean=data_clean)
'''


'''
For COSMOS VLA dataset

The plot_residuals() function visualizes errors made by each regression model:

Residuals = True SFR − Predicted SFR (in linear space, not log)
Colored by a variable of choice ('1.4GHz', 'redshift', etc.)
Gives insight into biases, trends, or systematic errors in the predictions.
'''

def plot_residuals(results, title, color_by='redshift', save_path=None, data_clean=data_clean):
    """
    Plots the residuals of a regression model, colored by a specified variable.

    Args:
        results (dict): A dictionary containing model results.
        title (str): The title for the entire plot.
        color_by (str, optional): The variable to color the points by.
                                  Must be 'redshift', 'sfr', '1.4GHz', '3GHz',
                                  'IR', or 'stellar_mass'. Defaults to 'redshift'.
        save_path (str, optional): The path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(18, 10))

    z_vals = data_clean['Photometric_Redshift'].values
    sfr_vals = data_clean['True_SFR'].values
    lum_1_4GHz_vals = data_clean['Radio_Luminosity_1.4GHz'].values
    lum_3GHz_vals = data_clean['Radio_Luminosity_3GHz'].values
    ir_mag_vals = data_clean['IR_Magnitude'].values
    stellar_mass_vals = data_clean['Stellar_Mass'].values

    if color_by == 'redshift':
        color_vals = np.log10(z_vals)
        cmap = 'viridis'
        color_label = 'Photometric Redshift'
    elif color_by == 'sfr':
        color_vals = np.log10(sfr_vals)
        cmap = 'plasma'
        color_label = 'True SFR (M⊙/yr)'
    elif color_by == '1.4GHz':
        color_vals = np.log10(lum_1_4GHz_vals)
        cmap = 'inferno'
        color_label = '1.4 GHz Luminosity'
    elif color_by == '3GHz':
        color_vals = np.log10(lum_3GHz_vals)
        cmap = 'cividis'
        color_label = '3 GHz Luminosity'
    elif color_by == 'IR':
        color_vals = np.log10(ir_mag_vals)
        cmap = 'magma'
        color_label = 'IR Magnitude (L_TIR_SF)'
    elif color_by == 'stellar_mass':
        color_vals = np.log10(stellar_mass_vals)
        cmap = 'cividis'  # Or any other suitable colormap
        color_label = 'Stellar Mass (M⊙)'
    else:
        raise ValueError("color_by must be 'redshift', 'sfr', '1.4GHz', '3GHz', 'IR', or 'stellar_mass'")

    for i, (name, (true_log, pred_log, _, _)) in enumerate(results.items(), 1):
        true = 10 ** true_log
        pred = 10 ** pred_log
        residuals = true - pred
        med_percent_err = np.median(np.abs(residuals) / true) * 100

        df = pd.DataFrame({'true': true, 'residuals': residuals})

        idxs = np.isin(data_clean['True_SFR'].values, true)
        color_data = color_vals[idxs]
        if len(color_data) != len(df):
            color_data = color_vals[:len(df)]

        plt.subplot(2, 3, i)
        sc = plt.scatter(df['true'], df['residuals'], c=color_data, cmap=cmap, alpha=0.6)
        plt.hlines(0, xmin=min(true), xmax=max(true), colors='black', linestyles='dashed')
        plt.title(f"{name} Residuals")
        plt.xlabel('True SFR (M⊙/yr)')
        plt.ylabel('Residuals (True - Predicted)')
        plt.grid(True)
        plt.text(0.95, 0.95, f'Median %Err: {med_percent_err:.2f}%',
                     transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round,pad=0.3'))
        plt.colorbar(sc, label=color_label)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



'''
For COSMOS VLA dataset

Evaluate models trained on different subsets of input features. such as different 
luminosities, or redshifts.

We then visualise performance and residuals for each model.

Color-code results based on astrophysical parameters (e.g., redshift, luminosity).

Save plots for easy inspection and reporting.
'''


'''
To avoid this error -> "ValueError: X has 1 features, but MLPRegressor is expecting 2 features as input."

Make sure the last feature set in the dictionary has the same numer of features (ideally the same model),
that is being reshaped in the variable log_sfr_pred.
'''

# ==== Evaluate All Feature Sets ====

#Only including the 1.4GHz model

feature_sets = {
    #'Full Features': (X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full),
    #'Radio Only': (X_train_radio_scaled, X_test_radio_scaled, y_train_radio, y_test_radio),
    #'Radio + z': (X_train_radio_photo_z_scaled, X_test_radio_photo_z_scaled, y_train_radio_photo_z, y_test_radio_photo_z),
    #'3 GHz Only': (X_train_3GHz_scaled, X_test_3GHz_scaled, y_train_3GHz, y_test_3GHz),
    #'1.4GHz + Stellar Mass': (X_train_1_4GHz_stellar_mass_scaled, X_test_1_4GHz_stellar_mass_scaled, y_train_1_4GHz_stellar_mass, y_test_1_4GHz_stellar_mass),
    '1.4 GHz Only': (X_train_1_4GHz_scaled, X_test_1_4GHz_scaled, y_train_1_4GHz, y_test_1_4GHz)
}

all_results = {}

color_modes = ['1.4GHz', '3GHz', 'IR', 'redshift', 'sfr', 'stellar_mass']


# Create an output directory
output_dir = "model_plots"
os.makedirs(output_dir, exist_ok=True)

for name, (X_train, X_test, y_train, y_test) in feature_sets.items():
    print(f"\n==== Evaluating: {name} ====")
    results, preds = train_and_evaluate(model_dict, X_train, X_test, y_train, y_test)
    all_results[name] = results

    for color_by in color_modes:
        safe_name = name.replace(" ", "_").lower()
        safe_color = color_by.replace(".", "").lower()

        true_pred_path = os.path.join(output_dir, f"{safe_name}_true_vs_pred_colored_by_{safe_color}.png")
        residuals_path = os.path.join(output_dir, f"{safe_name}_residuals_colored_by_{safe_color}.png")

        print(f"Saving: {true_pred_path} and {residuals_path}")

        plot_true_vs_pred(results,
                          f"{name} Model (Colored by {color_by})",
                          color_by=color_by,
                          save_path=true_pred_path,
                          data_clean=data_clean)

        plot_residuals(results,
                       f"{name} Model Residuals (Colored by {color_by})",
                       color_by=color_by,
                       save_path=residuals_path,
                       data_clean=data_clean)


'''
FOR COSMOS VLA dataset

This block adds a specialized diagnostic plot to visualize how model residuals
vary with redshift. For the 1.4GHz only model.
'''


def plot_residuals_vs_redshift(results, X_test, y_test_log, redshift_col='Photometric_Redshift', model_names=None, save_path=None):
    """
    Plot residuals (true - predicted) over redshift for selected models.

    Parameters:
        results: dict of model results from train_and_evaluate
        X_test: DataFrame containing redshift values
        y_test_log: true log10(SFR) values
        redshift_col: column name for redshift
        model_names: list of model names to include (optional)
        save_path: optional path to save the figure
    """
    if model_names is None:
        model_names = list(results.keys())

    redshifts = X_test[redshift_col].values
    n_models = len(model_names)

    plt.figure(figsize=(10, 4 * n_models))


    print(f"Type of results: {type(results)}")
    print(f"Type of results['{model_names[0]}']: {type(results[model_names[0]])}")
    print(f"Contents of results['{model_names[0]}']: {results[model_names[0]]}")


    for i, model_name in enumerate(model_names):
        true_log, pred_log, _, _ = results[model_name]
        residuals = true_log - pred_log

        ax = plt.subplot(n_models, 1, i + 1)
        sc = ax.scatter(redshifts, residuals, c=pred_log, cmap='viridis', alpha=0.6, s=20)
        ax.hlines(0, xmin=min(redshifts), xmax=max(redshifts), colors='gray', linestyles='dashed')
        ax.set_title(f"{model_name} - Residuals (True - Predicted) vs Redshift", fontsize=11)
        ax.set_ylabel("Residual (log SFR)", fontsize=10)
        ax.set_xlabel("Photometric Redshift", fontsize=10)
        ax.grid(True)
        plt.colorbar(sc, ax=ax, label='Predicted log(SFR)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Unpack the tuple returned by train_and_evaluate
results_1_4GHz, _ = train_and_evaluate(model_dict, X_train_1_4GHz_scaled, X_test_1_4GHz_scaled, y_train_1_4GHz, y_test_1_4GHz)
#results_1_4GHz_stellar, _ = train_and_evaluate(model_dict, X_train_1_4GHz_stellar_mass_scaled, X_test_1_4GHz_stellar_mass_scaled, y_train_1_4GHz_stellar_mass, y_test_1_4GHz_stellar_mass)

# Unscaled test set needed for redshift values
X_test_1_4GHz_unscaled = X_test_1_4GHz.copy()
X_test_1_4GHz_unscaled['Photometric_Redshift'] = X_test_full['Photometric_Redshift'].values

plot_residuals_vs_redshift(
    results_1_4GHz,  # now correctly a dict
    X_test=X_test_1_4GHz_unscaled,
    y_test_log=y_test_1_4GHz,
    model_names=list(model_dict.keys()),
    save_path="residuals_vs_redshift_1_4GHz_models.png"
)


'''
For COSMOS VLA dataset

This block introduces a side-by-side diagnostic plot that compares true vs. predicted
log(SFR) as a function of redshift. 
'''

def plot_predicted_vs_true_sfr_redshift(results, redshifts, save_path=None):
    """
    Make side-by-side plots of predicted log(SFR) vs redshift and true log(SFR) vs redshift
    for each ML model using 1.4GHz only input. Highlights 3-sigma outliers.

    Parameters:
        results: dict of model -> (true_log, pred_log, mae%, med%)
        redshifts: array of redshift values (same order as test set)
        save_path: optional path to save the figure
    """

    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 3.5 * n_models), sharex=True)

    vmin, vmax = -1, 3.5  # fixed limits for color and y-axis

    for i, (model_name, (true_log, pred_log, mae_pct, med_pct)) in enumerate(results.items()):
        ax_true = axes[i, 0]
        ax_pred = axes[i, 1]

        true_log = np.array(true_log)
        pred_log = np.array(pred_log)

        # Calculate 3-sigma thresholds (not used)
        #true_mean, true_std = np.mean(true_log), np.std(true_log)
        #pred_mean, pred_std = np.mean(pred_log), np.std(pred_log)

        #true_outliers = np.abs(true_log - true_mean) > (1 * true_std)
        #pred_outliers = np.abs(pred_log - pred_mean) > (1 * pred_std)

        # Plot true log(SFR)
        sc1 = ax_true.scatter(redshifts, true_log, c=true_log, cmap='plasma',
                              alpha=0.6, vmin=vmin, vmax=vmax, label='True SFR')
        #ax_true.scatter(redshifts[true_outliers], true_log[true_outliers],
                        #color='red', edgecolor='black', s=50, label='3σ Outliers')
        ax_true.set_ylabel("True log(SFR)", fontsize=10)
        ax_true.set_title(f"{model_name} - True log(SFR)", fontsize=11)
        ax_true.set_ylim(vmin, vmax)
        ax_true.grid(True)
        cbar1 = plt.colorbar(sc1, ax=ax_true)
        cbar1.set_label("True log(SFR)", fontsize=9)
        ax_true.legend(fontsize=8)

        # Plot predicted log(SFR)
        sc2 = ax_pred.scatter(redshifts, pred_log, c=pred_log, cmap='viridis',
                              alpha=0.6, vmin=vmin, vmax=vmax, label='Predicted SFR')
        #ax_pred.scatter(redshifts[pred_outliers], pred_log[pred_outliers],
                        #color='red', edgecolor='black', s=50, label='3σ Outliers')
        ax_pred.set_ylabel("Predicted log(SFR)", fontsize=10)
        ax_pred.set_title(f"{model_name} - Predicted log(SFR)", fontsize=11)
        ax_pred.set_ylim(vmin, vmax)
        ax_pred.grid(True)
        cbar2 = plt.colorbar(sc2, ax=ax_pred)
        cbar2.set_label("Predicted log(SFR)", fontsize=9)
        ax_pred.legend(fontsize=8)

        # Set x-axis labels for the last row
        if i == n_models - 1:
            ax_true.set_xlabel("Photometric Redshift", fontsize=10)
            ax_pred.set_xlabel("Photometric Redshift", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return true_log


# Grab the matching redshift values for the test set (unscaled)
redshifts_1_4GHz = X_test_full.loc[X_test_1_4GHz.index, 'Photometric_Redshift'].values

true_log = plot_predicted_vs_true_sfr_redshift(
    results_1_4GHz,
    redshifts=redshifts_1_4GHz,
    save_path="1_4GHz_predicted_vs_true_sfr_vs_redshift.png"
)


'''
For COSMOS VLA dataset

Plots the scatter of residuals against redshift for the 1.4GHz only model
'''


def plot_scatter_vs_redshift(model_name, y_true_log, y_pred_log, redshifts, bins=10):
    """
    Plot the standard deviation of residuals (scatter) in redshift bins.
    
    Parameters:
    - model_name: str, name of the model for labeling.
    - y_true_log: array-like, true log(SFR)
    - y_pred_log: array-like, predicted log(SFR)
    - redshifts: array-like, photometric redshift
    - bins: int, number of redshift bins
    """
    residuals = y_true_log - y_pred_log

    # Bin by redshift
    bin_edges = np.linspace(np.min(redshifts), np.max(redshifts), bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    scatter_per_bin = []

    for i in range(bins):
        in_bin = (redshifts >= bin_edges[i]) & (redshifts < bin_edges[i + 1])
        if np.sum(in_bin) > 1:
            scatter = np.std(residuals[in_bin])
        else:
            scatter = np.nan  # Skip bins with too few points
        scatter_per_bin.append(scatter)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, scatter_per_bin, marker='o', linestyle='-', color='tab:blue')
    plt.xlabel("Photometric Redshift")
    plt.ylabel("σ(residuals) in log(SFR)")
    plt.title(f"Scatter of Residuals vs Redshift — {model_name} (1.4 GHz only)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Get residuals and redshifts from the test set
best_model_name = 'Random Forest'  # Or whichever model you're interested in
y_true_log, y_pred_log, *_ = results_1_4GHz[best_model_name]

# You want redshift values *corresponding to the test set*
z_test = X_test_1_4GHz.index.to_numpy()
z_vals = data_clean.iloc[z_test]['Photometric_Redshift'].values

# Plot scatter vs redshift
plot_scatter_vs_redshift(best_model_name, y_true_log, y_pred_log, z_vals, bins=20)



'''
Reverses COSMOS VLA data trained model and gets predicted luminosity for Abell 133.

This code performs a reverse transformation using the ML model—specifically, 
it infers radio luminosity at 1.4 GHz from SFR values using a pre-trained 
Neural Network (MLP). That is, it inverts the model that originally predicted
log10(SFR) from log10(L_1.4GHz).

The sample that the model was trained on on has removed X-ray selected AGN identified
in Chandra and XMM-Newton. Additionally, they use an IR AGN Selection method. This uses
only the WISE W1 and W2 bands to identify AGN based on their W1-W2 colour

Lastly, to remove contamination radio loud galaxies... they apply a radio luminosity cut.
Some of which may be star-forming, but a relatively small number (<3%) of the total sample


These SFR values are calculated from W4 band magnitudes for sources in the Abell 133 
cluster.

1) WE have the model which was trained to predict log10(SFR) from log10(L_1.4GHz).
    
2) We refit the scaler to match the feature scaling used during training. This is
   critical to ensure consistent input format.

3) We sweep a very fine grid of radio luminosities to get smooth model predictions
   of log10(SFR) across the full plausible range.
    
4) WE numerically invert the MLP using interpolation.
   This gives us a function whereby log_10(SFR) -> log_10(L_1.4GHz). This is not
   analytically derived, but approximated by interpolating over the dense grid.
   
5) We load new SFR values (in linear units), convert to log10, and use the inverse
    function to get log10(L_1.4GHz).

6) A CSV is saved with the original SFRs and *predicted* radio luminosities.
    
7) Shows how the SFRs map to luminosities via the model.

8) Plots the original forward model curve (log SFR → log L) for visual validation.



Predict SFR from real radio data in COSMOS. True_SFR is true COSMOS SFR


True 1.4GHz Radio luminosity -> predicting forward the SFR -> taking residuals of
true vs predicted SFR in the COSMOS.

Variance of the residuals (errors) is not constant across all levels of the independent 
variable(s). (Heteroscedasticity)

Outputs:
    
The RMSE of 218 is very large. It indicates a high average prediction error.

R^2 Score: -0.2362

- R² means the model is performing worse than simply predicting the mean of the true SFR values.
This is a strong indication that the model has not learned a meaningful relationship between
radio luminosity and SFR.

'''


# ==== 1. Load pre-trained forward model and scaler (log L_1.4GHz → log SFR) ====
mlp_model = model_dict['Neural Network (MLP)']  # Pre-trained model

# Assume scaler_1_4 was saved during training; reload or recreate it consistently
scaler_1_4 = StandardScaler().fit(X_1_4GHz)  # Replace with saved scaler if possible

# ==== 2. Prepare training data for inverse model (log SFR → log L) ====
# Forward: log L → log SFR, so get predicted log SFR for training data
X_scaled = scaler_1_4.transform(X_1_4GHz)
log_sfr_pred = mlp_model.predict(X_scaled).reshape(-1, 1)


# True log L (target) and predicted log SFR (input) for inverse fit
log_L_true = X_1_4GHz.values.flatten()

# ==== 3. Train a new regression model for inverse mapping ====
# Use polynomial regression (degree 3) to map log SFR → log L
inverse_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
inverse_model.fit(log_sfr_pred, log_L_true)

# Calculate residuals on training set to estimate uncertainty
log_L_pred_train = inverse_model.predict(log_sfr_pred)
residuals_train = log_L_true - log_L_pred_train
residual_std = np.std(residuals_train)

print(f"Inverse model trained. Residual std: {residual_std:.4f} dex")

# ==== 4. Define cluster files ====
cluster_names = [
    'Abell_133', 'Abell_68', 'Abell_194', 'Abell_209', 'Abell_22',
    'Abell_2485', 'Abell_2597', 'Abell_2645', 'Abell_2667', 'Abell_2744', 'Abell_2751',
    'Abell_2811', 'Abell_2895', 'Abell_3365', 'Abell_3376', 'Abell_3558',
    'Abell_3562', 'Abell_3667', 'Abell_370', 'Abell_4038', 'Abell_521', 'Abell_545',
    'Abell_85', 'Abell_S1063', 'Abell_S1121', 'Abell_S295', 'ElGordo', 'J0014.3-6604'
]
cluster_files = [f'{name}_sfr.csv' for name in cluster_names]

# ==== 5. Process each cluster ====
for file_name in cluster_files:
    cluster_name = file_name.replace('_sfr.csv', '')
    
    try:
        new_sfr_df = pd.read_csv(f'/Users/jason/Downloads/Output/{file_name}')
        
        # Filter out zero or negative SFR values before log10
        positive_sfr_mask = new_sfr_df['SFR_Msun_per_yr'] > 0
        if not positive_sfr_mask.any():
            print(f"[{cluster_name}] No positive SFR values, skipping.")
            continue
        
        new_sfr_clean = new_sfr_df.loc[positive_sfr_mask].copy()
        new_sfr_log = np.log10(new_sfr_clean['SFR_Msun_per_yr'].values).reshape(-1, 1)
        
        # Check SFR log range to avoid extrapolation
        min_log_sfr_train, max_log_sfr_train = log_sfr_pred.min(), log_sfr_pred.max()
        new_sfr_log_clipped = np.clip(new_sfr_log, min_log_sfr_train, max_log_sfr_train)
        if not np.array_equal(new_sfr_log, new_sfr_log_clipped):
            print(f"[{cluster_name}] Some SFR values clipped to training range.")

        # Predict log L from inverse model
        predicted_logL = inverse_model.predict(new_sfr_log_clipped)
        
        # Add uncertainty estimate (± 1 sigma)
        new_sfr_clean['Predicted_log10_L_1.4GHz'] = predicted_logL
        new_sfr_clean['Predicted_log10_L_1.4GHz_uncertainty'] = residual_std
        
        # Save output with predictions and uncertainty
        output_csv_path = f'predicted_L_1.4GHz_from_SFR_{cluster_name}.csv'
        new_sfr_clean.to_csv(output_csv_path, index=False, float_format='%.10f')
        print(f"[{cluster_name}] Predictions saved to '{output_csv_path}'.")

        # Plot results
        plt.figure(figsize=(8, 6))
        plt.scatter(new_sfr_clean['SFR_Msun_per_yr'], 10**predicted_logL, 
                    c='dodgerblue', edgecolor='k', alpha=0.8)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('SFR [Msun/yr]')
        plt.ylabel('Predicted L_1.4GHz [W/Hz]')
        plt.title(f'{cluster_name}: Predicted Radio Luminosity vs. SFR')
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.tight_layout()
        plot_path = f'plot_predicted_L1.4_vs_SFR_{cluster_name}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"[{cluster_name}] Error processing file: {e}")

# ==== 6. Plot forward and inverse models for visualization ====
plt.figure(figsize=(8, 6))
plt.scatter(log_sfr_pred, log_L_true, s=1, color='darkorange', label='Training Data')
plt.plot(log_sfr_pred, inverse_model.predict(log_sfr_pred), '--', color='blue', label='Inverse Model Fit')
plt.xlabel('log10(SFR)')
plt.ylabel('log10(L_1.4GHz)')
plt.title('Inverse Model: log10(SFR) → log10(L_1.4GHz)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== 7. Evaluate forward model residuals ====
sfr_true = data['True_SFR']
X_real_scaled = scaler_1_4.transform(X_1_4GHz)
sfr_pred_forward = mlp_model.predict(X_real_scaled)

residuals = sfr_true - sfr_pred_forward

plt.figure(figsize=(8, 6))
plt.scatter(X_1_4GHz.iloc[:, 0], residuals, alpha=0.6, c='tomato', edgecolor='k')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('log10(L_1.4GHz)')
plt.ylabel('Residual: True SFR - Predicted SFR')
plt.title('Residuals of Forward Model')
plt.grid(True)
plt.tight_layout()
plt.savefig('residuals_forward_model.png', dpi=300)
plt.show()

# Check heteroscedasticity
plt.figure(figsize=(8, 6))
plt.scatter(sfr_pred_forward, np.abs(residuals), alpha=0.6, c='mediumseagreen', edgecolor='k')
plt.xlabel('Predicted log10(SFR)')
plt.ylabel('Absolute Residual')
plt.title('Residual Magnitude vs. Predicted SFR (Heteroscedasticity Check)')
plt.grid(True)
plt.tight_layout()
plt.savefig('heteroscedasticity_check.png', dpi=300)
plt.show()

# Print metrics
rmse = np.sqrt(mean_squared_error(sfr_true, sfr_pred_forward))
r2 = r2_score(sfr_true, sfr_pred_forward)
print(f"Forward model RMSE: {rmse:.4f}")
print(f"Forward model R² score: {r2:.4f}")


'''
From predicted luminosity 


This code calculates radio flux densities in mJy from predicted radio luminosities
(log₁₀ L₁.₄GHz) for sources at a fixed redshift using cosmological distance 
calculations.

 It’s the final physical step: turning model-predicted luminosities
into observable fluxes.
'''


# Cosmology (Planck 2018 parameters)
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)

# Cluster data no abell_68, abell_S1063, abell_S1121, abell_S295 or Abell 548
cluster_names = [
    'Abell_133', 'Abell_194', 'Abell_209', 'Abell_22',
    'Abell_2485', 'Abell_2597', 'Abell_2645', 'Abell_2667', 'Abell_2744', 'Abell_2751',
    'Abell_2811', 'Abell_2895', 'Abell_3365', 'Abell_3376', 'Abell_3558',
    'Abell_3562', 'Abell_3667', 'Abell_370', 'Abell_4038', 'Abell_521', 'Abell_545',
    'Abell_85', 'ElGordo', 'J0014.3-6604'
]


redshifts = [
    0.057, 0.018, 0.21, 0.142, 0.247, 0.085, 0.251, 0.230,
    0.308, 0.107, 0.108, 0.227, 0.093, 0.046, 0.048, 0.049, 0.056,
    0.375, 0.028, 0.253, 0.154, 0.055, 0.870, 0.155
]


#See if can implement varying spectral index (cannot implement due to lack of data)
alpha = 0.7  # Spectral index (same for all cluster fields)

# Conversion factor: 1 Mpc = 3.085677581e22 meters
MPC_TO_M = 3.085677581e22

# Conversion function
def luminosity_to_flux(log10_L, D_L_m):
    L = 10**log10_L  # Convert log luminosity to linear Watts/Hz
    S_W_m2_Hz = L / (4 * np.pi * D_L_m**2) * (1 + z)**(1 - alpha)  # Flux in W/m²/Hz
    S_mJy = S_W_m2_Hz * 1e29  # Convert W/m²/Hz to mJy
    return S_mJy

# Loop through each cluster
for name, z in zip(cluster_names, redshifts):
    # File path
    input_file = f'predicted_L_1.4GHz_from_SFR_{name}.csv'
    output_file = f'predicted_L_1.4GHz_from_SFR_flux_density_{name}.csv'

    # Load CSV
    df = pd.read_csv(input_file)

    # Calculate luminosity distance in meters
    D_L_m = cosmo.luminosity_distance(z).value * MPC_TO_M

    # Apply the conversion
    df['flux_density_mJy'] = df['Predicted_log10_L_1.4GHz'].apply(lambda logL: luminosity_to_flux(logL, D_L_m))

    # Save to new CSV
    df.to_csv(output_file, index=False)

    print(f"Processed {name}: saved to {output_file}")


'''
This final code block evaluates how well the ML-predicted radio flux densities
match actual observed fluxes from the radio catalog—specifically for all sources
in Abell 133.


'''

# Cluster names
clusters = [
    'Abell_133', 'Abell_194', 'Abell_209', 'Abell_22',
    'Abell_2485', 'Abell_2597', 'Abell_2645', 'Abell_2667', 'Abell_2744', 'Abell_2751',
    'Abell_2811', 'Abell_2895', 'Abell_3365', 'Abell_3376', 'Abell_3558',
    'Abell_3562', 'Abell_3667', 'Abell_370', 'Abell_4038', 'Abell_521', 'Abell_545',
    'Abell_85', 'ElGordo', 'J0014.3-6604'
]

# Load observed data (same for all)
# Make sure the path to your CSV is correct




try:
    observed = pd.read_csv('/Users/jason/Downloads/Table2_MGCLS_compactcat_DR1.csv')
except FileNotFoundError:
    print("Error: 'Table2_MGCLS_compactcat_DR1.csv' not found. Please check the file path.")
    exit() # Exit if the main observed file isn't found

predicted_clusters = {}  # dictionary to hold DataFrames keyed by cluster name

for cluster in clusters:
    # File paths
    predicted_file = f'predicted_L_1.4GHz_from_SFR_flux_density_{cluster}.csv'

    try:
        predicted = pd.read_csv(predicted_file)
    except FileNotFoundError:
        print(f"Missing file: {predicted_file}")
        continue

    # Save this DataFrame in the dictionary
    predicted_clusters[cluster] = predicted

    # Convert cluster name format (underscore -> space)
    cluster_field = cluster.replace('_', ' ')

    # Filter by cluster
    observed_cluster = observed[observed['Field'] == cluster_field].reset_index(drop=True)
    predicted_cluster = predicted[predicted['Field'] == cluster_field].reset_index(drop=True)

    # Skip if no matching entries
    if observed_cluster.empty or predicted_cluster.empty:
        print(f"Skipping {cluster}: no matching data after filtering by 'Field'.")
        continue

    # --- Cut off arrays to the minimum length ---
    min_len = min(len(observed_cluster), len(predicted_cluster))

    if len(observed_cluster) != len(predicted_cluster):
        print(f"Warning: Mismatch in number of sources for {cluster}. Observed: {len(observed_cluster)}, Predicted: {len(predicted_cluster)}. Truncating to min length: {min_len}.")

    observed_cluster_matched = observed_cluster.iloc[:min_len].copy()
    predicted_cluster_matched = predicted_cluster.iloc[:min_len].copy()

    # Manually align rows and assign Source_name based on the truncated data
    predicted_cluster_matched['# Source_name'] = observed_cluster_matched['# Source_name'].values

    # Merge using Source_name (now available in predicted_cluster_matched)
    merged = pd.merge(
        predicted_cluster_matched[['# Source_name', 'flux_density_mJy']],
        observed_cluster_matched[['# Source_name', 'Stot_mJy']],
        on='# Source_name'
    )
    # Extract values
    x = merged['Stot_mJy']          # observed
    y = merged['flux_density_mJy']  # predicted

    # Compute residuals and percentiles
    residuals = y - x
    p16 = np.percentile(residuals, 16)
    p84 = np.percentile(residuals, 84)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6, label='Sources')

    # --- Define plot limits and 1:1 line range based on the cluster ---
    if cluster == "Abell_22":
        x_min_limit, x_max_limit = 0, 2.5
        y_min_limit, y_max_limit = 0.4, 2.5
        plot_title_suffix = " (Matched by Truncation, Custom Limits)"
    else:
        x_min_limit, x_max_limit = 0, 1
        y_min_limit, y_max_limit = 0, 1
        plot_title_suffix = " (Matched by Truncation)"

    plt.plot([x_min_limit, x_max_limit], [y_min_limit, y_max_limit], 'r--', label='1:1 line')

    # Percentile shading
    x_vals_plot = np.linspace(x_min_limit, x_max_limit, 200)
    #plt.fill_between(x_vals_plot, x_vals_plot + p16, x_vals_plot + p84,
                     #where=(x_vals_plot + p84) >= y_min_limit, # Ensure shading starts at or above y_min_limit
                     #color='gray', alpha=0.3, label='16th–84th percentile range')

    # --- Set xlim and ylim ---
    plt.xlim(x_min_limit, x_max_limit)
    plt.ylim(y_min_limit, y_max_limit)

    plt.xlabel("Observed Flux Density (mJy)")
    plt.ylabel("Predicted Flux Density (mJy)")
    plt.title(f"Predicted vs Observed Flux Density — {cluster_field}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'predicted_vs_observed_flux_density_{cluster_field}.png', dpi=300)
    plt.show()
    
    

print("\nProcessing complete.")



'''
Plotting COSMOS SED fitting true vs predicted flux density
'''


# File paths
deblended_fits_file_path = '/Users/jason/Downloads/COSMOS_VLA_Deblended.fits'
fits_file_path = '/Users/jason/Downloads/VLA_3GHz_counterpart_array_20170210_paper_delvecchio_et_al.fits'

# Load FITS tables
deblended_table = Table.read(deblended_fits_file_path)
vla_table = Table.read(fits_file_path)

# Create SkyCoord objects
coords_deblended = SkyCoord(ra=deblended_table['RA']*u.deg, dec=deblended_table['Dec']*u.deg)
coords_vla = SkyCoord(ra=vla_table['RA_VLA3']*u.deg, dec=vla_table['DEC_VLA3']*u.deg)

# Match coordinates within 1 arcsec
idx, d2d, _ = coords_deblended.match_to_catalog_sky(coords_vla)
matched = d2d < 1 * u.arcsec

# Select matched rows
matched_deblended = deblended_table[matched]
matched_vla = vla_table[idx[matched]]

# Combine matched tables horizontally
matched_combined = hstack([matched_deblended, matched_vla], join_type='exact')

# Save result as FITS
matched_combined.write('/Users/jason/Downloads/cosmos_and_deblended_matched_sources.fits', overwrite=True)

print(f"Matched {len(matched_combined)} sources within 1 arcsec.")


'''
f20cm & df20cm --- VLA 1.4GHz flux and uncertainty [uJy]

xf20cm & xe20cm --- Super-deblended SED predicted flux density & uncertainty at 20cm [mJy]
'''

plt.figure(figsize=(10, 8))  # Set figure size

plt.scatter(deblended_table['f20cm']/100, deblended_table['xf20cm'], alpha=0.6, edgecolors='k', s=60, cmap='viridis')

# Set x and y axis ranges (adjust the numbers to your data)
plt.xlim(0, 1)    # example: x-axis from 0 to 10
plt.ylim(0, 1)     # example: y-axis from 0 to 5

# Grid for better readability
plt.grid(True, linestyle='--', alpha=0.5)


plt.xlabel('VLA-COSMOS 1.4 GHz flux (mJy)', fontsize=14)
plt.ylabel('SED Predicted Flux Density(mJy)', fontsize=14)
plt.title('Predicted vs observed 1.4 GHz radio luminosity', fontsize=14)
plt.tight_layout()
plt.savefig('Pred_vs_Obs_1.4GHz_RL_SED_fitting.png', dpi=300)
plt.show()



# Example: Set colorbar limits for SFR_IR_1 and flux ratio
sfr_vmin = -2      # Set according to your data
sfr_vmax = 2    # Example upper limit

ratio_vmin = 0  # Set according to your data
ratio_vmax = 2.0  # Example upper limit

# Scatter 1: color by SFR_IR_1 with limits
plt.figure(figsize=(6, 5))
sc1 = plt.scatter(deblended_table['f20cm'] / 100,
                  deblended_table['xf20cm'],
                  c=np.log10(deblended_table['SFR_IR_1']),
                  cmap='viridis',
                  alpha=0.6,
                  edgecolors='k',
                  s=60,
                  vmin=sfr_vmin,
                  vmax=sfr_vmax)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('VLA 1.4 GHz flux (mJy)', fontsize=12)
plt.ylabel('SED predicted (mJy)', fontsize=12)
plt.title('Predicted vs Observed 1.4 GHz\nColored by IR SFR', fontsize=14, weight='bold')
plt.colorbar(sc1, label='SFR_IR_1')
plt.tight_layout()
plt.savefig('Pred_vs_Obs_1.4GHz_colored_by_SFR_limited.png', dpi=300)
plt.show()



# Scatter 2: color by flux ratio with limits
flux_ratio = deblended_table['xf20cm'] / (deblended_table['f20cm'] / 100)

plt.figure(figsize=(6, 5))
sc2 = plt.scatter(deblended_table['f20cm'] / 100,
                  deblended_table['xf20cm'],
                  c=flux_ratio,
                  cmap='plasma',
                  alpha=0.6,
                  edgecolors='k',
                  s=60,
                  vmin=ratio_vmin,
                  vmax=ratio_vmax)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('VLA 1.4 GHz flux (mJy)', fontsize=12)
plt.ylabel('SED predicted (mJy)', fontsize=12)
plt.title('Predicted vs Observed 1.4 GHz\nColored by Predicted/Observed Flux Ratio', fontsize=14, weight='bold')
plt.colorbar(sc2, label='Predicted / Observed Flux Density')
plt.tight_layout()
plt.savefig('Pred_vs_Obs_1.4GHz_colored_by_flux_ratio_limited.png', dpi=300)
plt.show()




# Scatter 3: color by flux ratio with limits
redshift_colour = deblended_table['z_phot']

plt.figure(figsize=(6, 5))
sc2 = plt.scatter(deblended_table['f20cm'] / 100,
                  deblended_table['xf20cm'],
                  c=redshift_colour,
                  cmap='plasma',
                  alpha=0.6,
                  edgecolors='k',
                  s=60,
                  vmin=ratio_vmin,
                  vmax=ratio_vmax)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('VLA 1.4 GHz flux (mJy)', fontsize=12)
plt.ylabel('SED predicted (mJy)', fontsize=12)
plt.title('Predicted vs Observed 1.4 GHz\nColored by Redshift', fontsize=14, weight='bold')
plt.colorbar(sc2, label='Redshift')
plt.tight_layout()
plt.savefig('Pred_vs_Obs_1.4GHz_colored_by_redshift_limited.png', dpi=300)
plt.show()



'''
Reversing to get predicted luminosity again. In cosmos
'''


# === 1. Load saved inverse model and scalers ===
inverse_mlp = joblib.load("inverse_mlp_model.pkl")
sfr_scaler = joblib.load("sfr_scaler.pkl")
L_scaler = joblib.load("L_scaler.pkl")

# === 2. Load new predicted SFRs from CSV ===
input_csv = '/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/test_dataset_1_4GHz_with_predictions.csv'
#input_csv = '/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/test_dataset_1_4GHz_stellar_with_predictions.csv' including stellar mass in the model


df = pd.read_csv(input_csv)

# Ensure 'Predicted_SFR' column exists and values are positive
if 'Predicted_SFR' not in df.columns:
    raise ValueError("Column 'Predicted_SFR' not found in input CSV.")
predicted_sfr = df['Predicted_SFR'].values
if np.any(predicted_sfr <= 0):
    raise ValueError("Predicted_SFR values must be positive to compute log10.")

# === 3. Convert predicted SFR to log10 and scale ===
log10_sfr_input = np.log10(predicted_sfr).reshape(-1, 1)
log10_sfr_scaled = sfr_scaler.transform(log10_sfr_input)

# === 4. Predict log10(L_1.4GHz) with the inverse model ===
log10_L_scaled_pred = inverse_mlp.predict(log10_sfr_scaled).reshape(-1, 1)

# === 5. Inverse scale predictions back to original space ===
log10_L_pred = L_scaler.inverse_transform(log10_L_scaled_pred).flatten()

# === 6. Save predictions to CSV ===
df['Predicted_log10_L_1.4GHz_inverse_model'] = log10_L_pred
output_csv = 'predicted_L_1.4GHz_from_SFR_inverse_model.csv'
df.to_csv(output_csv, index=False, float_format="%.10f")

# including stellar mass in the model
#df['Predicted_log10_L_1.4GHz_stellar_inverse_model'] = log10_L_pred
#output_csv = 'predicted_L_1.4GHz_stellar_from_SFR_inverse_model.csv'
#df.to_csv(output_csv, index=False, float_format="%.10f")


print(f"✅ Inverse model prediction complete. Results saved to '{output_csv}'.")


# Plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    df['Redshift'],
    df['Radio_Luminosity_1.4GHz'],
    c=np.log10(df['Predicted_SFR']),         # Color by true SFR
    cmap='viridis',
    s=10,
    alpha=0.7
)
plt.xlabel('Redshift (z)')
plt.ylabel('Predicted Radio Luminosity (W/m²)')
plt.title('Radio Flux vs Redshift (Predicted)')
plt.yscale('log')

cbar = plt.colorbar(scatter)
cbar.set_label('log SFR M_s/yr')

plt.savefig('radio_flux_vs_redshift.png', dpi=300) 
plt.show()




# === 1. Load shared models and scalers ===
inverse_mlp = joblib.load("inverse_mlp_model.pkl")
forward_mlp = joblib.load("forward_mlp_model.pkl")
sfr_scaler = joblib.load("sfr_scaler.pkl")
L_scaler = joblib.load("L_scaler.pkl")

# === 2. Cluster info ===
clusters = [
    'Abell_133', 'Abell_194', 'Abell_209', 'Abell_22',
    'Abell_2485', 'Abell_2597', 'Abell_2645', 'Abell_2667', 'Abell_2744', 'Abell_2751',
    'Abell_2811', 'Abell_2895', 'Abell_3365', 'Abell_3376', 'Abell_3558',
    'Abell_3562', 'Abell_3667', 'Abell_370', 'Abell_4038', 'Abell_521', 'Abell_545',
    'Abell_85', 'ElGordo', 'J0014.3-6604'
]

input_dir = "/Users/jason/Downloads/Output/"
output_dir = input_dir  # You can separate if preferred

for cluster in clusters:
    print(f"\n🔁 Processing {cluster}...")

    # === Load SFR input CSV ===
    input_csv = os.path.join(input_dir, f"{cluster}_sfr.csv")
    if not os.path.exists(input_csv):
        print(f"❌ Missing input file for {cluster}")
        continue

    df = pd.read_csv(input_csv)

    # === Ensure SFR column exists ===
    if "SFR_Msun_per_yr" not in df.columns:
        print(f"❌ 'SFR_Msun_per_yr' missing in {cluster}")
        continue

    # === Prepare output columns ===
    valid_mask = df["SFR_Msun_per_yr"].notna() & (df["SFR_Msun_per_yr"] > 0)
    df[f'Predicted_log10_L_1.4GHz_inverse_model_{cluster}'] = np.nan

    # === Inverse prediction: SFR → L ===
    sfr_values = df.loc[valid_mask, "SFR_Msun_per_yr"].values
    log10_sfr = np.log10(sfr_values).reshape(-1, 1)
    log10_sfr_scaled = sfr_scaler.transform(log10_sfr)
    log10_L_scaled_pred = inverse_mlp.predict(log10_sfr_scaled).reshape(-1, 1)
    log10_L_pred = L_scaler.inverse_transform(log10_L_scaled_pred).flatten()

    df.loc[valid_mask, f'Predicted_log10_L_1.4GHz_inverse_model_{cluster}'] = log10_L_pred

    # === Save intermediate prediction ===
    inverse_csv = os.path.join(output_dir, f'predicted_L_1.4GHz_from_SFR_inverse_model_{cluster}.csv')
    df.to_csv(inverse_csv, index=False, float_format="%.10f")
    print(f"✅ Inverse model saved: {inverse_csv}")

    # === Forward prediction: L → SFR ===
    valid_mask_L = df[f'Predicted_log10_L_1.4GHz_inverse_model_{cluster}'].notna()
    df[f'{cluster}_Predicted_log10_SFR_from_L_model'] = np.nan
    df[f'{cluster}_Predicted_SFR_Msun_per_yr_from_L_model'] = np.nan

    log10_L_input = df.loc[valid_mask_L, f'Predicted_log10_L_1.4GHz_inverse_model_{cluster}'].values.reshape(-1, 1)
    log10_L_scaled = L_scaler.transform(log10_L_input)
    log10_sfr_scaled_pred = forward_mlp.predict(log10_L_scaled).reshape(-1, 1)
    log10_sfr_pred = sfr_scaler.inverse_transform(log10_sfr_scaled_pred).flatten()
    sfr_pred = 10 ** log10_sfr_pred

    df.loc[valid_mask_L, f'{cluster}_Predicted_log10_SFR_from_L_model'] = log10_sfr_pred
    df.loc[valid_mask_L, f'{cluster}_Predicted_SFR_Msun_per_yr_from_L_model'] = sfr_pred

    # === Save final result ===
    final_csv = os.path.join(output_dir, f'{cluster}_predicted_SFR_from_L_model.csv')
    df.to_csv(final_csv, index=False, float_format="%.10f")
    print(f"✅ Final forward model saved: {final_csv}")



'''
ABELL ML PREDICTIONS

Look at individual sources – check ratio of predicted to measured SFR gives us a
measure of SFR contribution in each source
'''



# === 1. Cluster list ===
clusters = [
    'Abell_133', 'Abell_194', 'Abell_209', 'Abell_22',
    'Abell_2485', 'Abell_2597', 'Abell_2645', 'Abell_2667', 'Abell_2744', 'Abell_2751',
    'Abell_2811', 'Abell_2895', 'Abell_3365', 'Abell_3376', 'Abell_3558',
    'Abell_3562', 'Abell_3667', 'Abell_370', 'Abell_4038', 'Abell_521', 'Abell_545',
    'Abell_85', 'ElGordo', 'J0014.3-6604'
]

# === 2. Define base paths ===
base_flux_dir = "/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project"
base_obs_dir = "/Users/jason/Downloads"
base_sfr_dir = "/Users/jason/Downloads/Output"

# === 3. File paths ===
observed_catalog_path = os.path.join(base_obs_dir, "Table2_MGCLS_compactcat_DR1.csv")

# === 4. Load full observed catalog ===
observed_cluster = pd.read_csv(observed_catalog_path)

# === 5. Assuming you have predicted_clusters dictionary ready ===
# predicted_clusters = {
#   'Abell_133': df_for_Abell_133,
#   'Abell_141': df_for_Abell_141,
#   ...
# }


print("Observed catalog unique Fields:")
print(observed_cluster['Field'].unique())

for cluster, df in predicted_clusters.items():
    print(f"\nPredicted catalog unique Fields for {cluster}:")
    print(df['Field'].unique())

for cluster in clusters:
    print(f"\n📊 Processing cluster: {cluster}")
    
    # Convert cluster name for 'Field' column match
    cluster_field = cluster.replace("_", " ")

    # Load SFR prediction CSV
    sfr_path = os.path.join(base_sfr_dir, f"{cluster}_predicted_SFR_from_L_model.csv")
    if not os.path.exists(sfr_path):
        print(f"❌ Skipping {cluster} — Missing SFR CSV")
        continue

    sfr_df = pd.read_csv(sfr_path)

    # Get predicted cluster DataFrame from dictionary
    if cluster not in predicted_clusters:
        print(f"❌ Skipping {cluster} — Missing predicted cluster data in dictionary")
        continue
    predicted_cluster = predicted_clusters[cluster]

    # Filter predicted and observed flux by Field
    obs_flux = observed[observed['Field'] == cluster_field].copy()
    pred_flux = predicted_clusters[cluster][predicted_clusters[cluster]['Field'] == cluster_field].copy()

    # Check for empty DataFrames
    if pred_flux.empty or obs_flux.empty:
        print(f"⚠️ Skipping {cluster}: no matching flux data.")
        continue

    # Truncate all DataFrames to same length
    min_len = min(len(pred_flux), len(obs_flux), len(sfr_df))
    pred_flux = pred_flux.iloc[:min_len].reset_index(drop=True)
    obs_flux = obs_flux.iloc[:min_len].reset_index(drop=True)
    sfr_df = sfr_df.iloc[:min_len].reset_index(drop=True)

    # Ensure '# Source_name' exists in both before merging
    if '# Source_name' not in pred_flux.columns:
        pred_flux['# Source_name'] = obs_flux['# Source_name'].values

    # Merge predicted and observed fluxes
    merged = pd.merge(
        pred_flux[['# Source_name', 'flux_density_mJy']],
        obs_flux[['# Source_name', 'Stot_mJy', 'RA_deg', 'Dec_deg']],
        on='# Source_name'
    )

    # Optionally merge with SFR
    if 'SFR_Msun_per_yr' in sfr_df.columns:
        merged['SFR_Msun_per_yr'] = sfr_df['SFR_Msun_per_yr']
    if f'{cluster}_Predicted_SFR_Msun_per_yr_from_L_model' in sfr_df.columns:
        merged['Predicted_SFR_Msun_per_yr_from_L'] = sfr_df[f'{cluster}_Predicted_SFR_Msun_per_yr_from_L_model']

    print(f"✅ Merged {len(merged)} rows for {cluster}")

    x = merged['Stot_mJy'].values        # observed
    y = merged['flux_density_mJy'].values  # predicted
    
    observed_flux_in_cluster = x
    predicted_flux_in_cluster = y

    # Flux ratio from merged (element-wise division)
    # After computing flux_ratio:
    flux_ratio = y / x  # y = predicted, x = observed
    
    # Add flux_ratio to merged DataFrame first
    merged['flux_ratio'] = flux_ratio
    merged['observed_flux_in_cluster'] = observed_flux_in_cluster
    merged['predicted_flux_in_cluster'] = predicted_flux_in_cluster

    # Add flux_ratio back to pred_flux and obs_flux using '# Source_name' alignment
    # Create a DataFrame for flux_ratio with '# Source_name' so it can be merged safely
    flux_ratio_df = pd.DataFrame({
        '# Source_name': merged['# Source_name'],
        'flux_ratio': flux_ratio,
        'observed_flux_in_cluster': observed_flux_in_cluster,
        'predicted_flux_in_cluster': predicted_flux_in_cluster,
        'RA_deg': merged['RA_deg'],
        'Dec_deg': merged['Dec_deg']
    })
    
    # Merge into pred_flux and obs_flux on '# Source_name'
    pred_flux = pred_flux.merge(flux_ratio_df, on='# Source_name', how='left')
    obs_flux = obs_flux.merge(flux_ratio_df, on='# Source_name', how='left')
    
    # Optional: add to sfr_df if it also has '# Source_name' and you want it there too
    if '# Source_name' in sfr_df.columns:
        sfr_df = sfr_df.merge(flux_ratio_df, on='# Source_name', how='left')
    
    
    print(f"Flux ratios for {cluster}:\n{merged[['# Source_name', 'flux_ratio']]}")
    # Or to save to a CSV:
    merged[['# Source_name', 'flux_ratio', 'observed_flux_in_cluster', 'predicted_flux_in_cluster', 'SFR_Msun_per_yr', 'Predicted_SFR_Msun_per_yr_from_L', 'RA_deg', 'Dec_deg']].to_csv(f"{cluster}_flux_ratio.csv", index=False)
        
    # For SFR ratio — if SFR is merged into `merged` DataFrame, use it there
    if 'SFR_Msun_per_yr' in merged.columns and 'Predicted_SFR_Msun_per_yr_from_L' in merged.columns:
        sfr_ratio = merged['Predicted_SFR_Msun_per_yr_from_L'] / merged['SFR_Msun_per_yr']
    else:
        # fallback or skip plotting SFR ratio
        sfr_ratio = None

    # === 5. Plot 1: Flux Ratio ===
    plt.figure(figsize=(8, 6))
    sc1 = plt.scatter(
        x, y,
        c=flux_ratio,
        cmap='plasma',
        vmin=0,
        vmax=2,
        alpha=0.6,
        edgecolors='k',
        s=60
    )
    plt.colorbar(sc1).set_label('Predicted / True Flux Ratio (mJy)', fontsize=12)
    plt.xlabel('Observed Flux (mJy)', fontsize=12)
    plt.ylabel('Predicted Flux (mJy)', fontsize=12)
    plt.title(f'{cluster}: Flux Comparison\nColored by Flux Ratio', fontsize=14, weight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{cluster}_Pred_vs_Obs_Flux_colored_by_flux_ratio_loglog.png", dpi=300)
    plt.close()
    
    # === 6. Plot 2: SFR Ratio ===
    plt.figure(figsize=(8, 6))
    sc2 = plt.scatter(
        x, y,
        c=sfr_ratio,
        cmap='plasma',
        vmin=0,
        vmax=0.5,
        alpha=0.6,
        edgecolors='k',
        s=60
    )
    plt.colorbar(sc2).set_label('Predicted / True SFR Ratio', fontsize=12)
    plt.xlabel('Observed Flux (mJy)', fontsize=12)
    plt.ylabel('Predicted Flux (mJy)', fontsize=12)
    plt.title(f'{cluster}: Flux Comparison\nColored by SFR Ratio', fontsize=14, weight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{cluster}_Pred_vs_Obs_Flux_colored_by_sfr_ratio_loglog.png", dpi=300)
    plt.close()


# Assuming predicted_clusters is your dict: cluster_name -> predicted DataFrame
# base_sfr_dir is the folder containing your SFR CSVs

for cluster in clusters:
    print(f"\nProcessing {cluster}...")

    # Load the SFR CSV for the cluster
    sfr_path = os.path.join(base_sfr_dir, f"{cluster}_predicted_SFR_from_L_model.csv")
    if not os.path.exists(sfr_path):
        print(f"❌ Missing SFR CSV for {cluster}, skipping.")
        continue

    sfr_df = pd.read_csv(sfr_path)

    # Check if required SFR columns are present
    required_sfr_cols = ['SFR_Msun_per_yr', f'{cluster}_Predicted_SFR_Msun_per_yr_from_L_model']
    missing_cols = [col for col in required_sfr_cols if col not in sfr_df.columns]
    if missing_cols:
        print(f"⚠️ {cluster}: Missing required SFR columns {missing_cols}, skipping.")
        continue

    # Get the predicted cluster DataFrame from your dict
    pred_df = predicted_clusters.get(cluster)
    if pred_df is None:
        print(f"⚠️ {cluster}: Predicted cluster DataFrame not found in dictionary, skipping.")
        continue

    # Merge predicted cluster and SFR DataFrame on '# Source_name' or other key
    # Check if '# Source_name' exists in both dfs for merge
    if 'Field' not in pred_df.columns or 'Field' not in sfr_df.columns:
        print(f"⚠️ {cluster}: 'Field' missing in predicted or SFR DataFrame, skipping.")
        continue
    

    merged_df = pd.merge(pred_df, sfr_df[['Field'] + required_sfr_cols], on='Field', how='inner')

    if merged_df.empty:
        print(f"⚠️ {cluster}: Merged DataFrame empty after merging predicted and SFR, skipping.")
        continue

    # Calculate SFR Ratio
    
    sfr_ratio = merged_df[f'{cluster}_Predicted_SFR_Msun_per_yr_from_L_model'] / merged_df['SFR_Msun_per_yr_x']

    plt.figure(figsize=(8,5))
    plt.hist(np.log10(sfr_ratio), bins=20, color='steelblue', edgecolor='black')
    plt.xlabel('SFR Contribution Ratio (Predicted / Observed) log10')
    plt.ylabel('Number of Sources')
    plt.title(f'Histogram of SFR Contribution Ratios - {cluster}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



'''
For VLA-COSMOS data
'''

# Load the CSV file
csv_df = pd.read_csv('predicted_L_1.4GHz_from_SFR_inverse_model.csv')

# Load the FITS file
with fits.open('/Users/jason/Downloads/COSMOS_VLA_Deblended.fits') as hdul:
    fits_data = hdul[1].data
    
    # Convert each field to native byte order (little-endian)
    native_fits_data = {}
    for name in fits_data.names:
        col = fits_data[name]
        if col.dtype.byteorder not in ('=', '|'):
            col = col.byteswap().newbyteorder()
        native_fits_data[name] = col
    
    fits_df = pd.DataFrame(native_fits_data)

# Clean FITS data (if necessary)
fits_df = fits_df[(fits_df['Dec'] >= -90) & (fits_df['Dec'] <= 90)]
fits_df = fits_df.dropna(subset=['RA', 'Dec', 'z_phot'])  # also drop if z_phot missing

# Create SkyCoord objects
csv_coords = SkyCoord(ra=csv_df['RA'].values * u.degree, dec=csv_df['Dec'].values * u.degree)
fits_coords = SkyCoord(ra=fits_df['RA'].values * u.degree, dec=fits_df['Dec'].values * u.degree)

# Match catalogs within 2 arcseconds
idx, d2d, _ = csv_coords.match_to_catalog_sky(fits_coords)
max_sep = 2 * u.arcsec
matched = d2d < max_sep

# Filter matched sources in csv and add z_phot from fits using the matched indices
csv_matched = csv_df[matched].reset_index(drop=True)
matched_fits_indices = idx[matched]

# Add z_phot column from fits catalog
# Add desired columns from fits catalog
for col in ['z_phot', 'XRAY_AGN', 'MIR_AGN', 'SED_AGN']:
    csv_matched[col] = fits_df.iloc[matched_fits_indices][col].values

# Save output CSV
csv_matched.to_csv('csv_matched_with_zphot.csv', index=False)

fits_matched = fits_df.iloc[idx[matched]].reset_index(drop=True)

# Example: scatter plot of a column from csv vs a column from fits
# Replace 'Predicted_L' and 'Measured_L' with actual column names you want to plot
plt.scatter(csv_matched['Radio_Luminosity_1.4GHz'], csv_matched['Predicted_log10_L_1.4GHz_inverse_model'], s=10, alpha=0.6)
plt.xlabel('true L')
plt.ylabel('Predicted L (from COSMOS VLA)')
plt.title('True Luminosity vs Predicted Luminosity')
plt.xlim(20, 30)
plt.ylim(20, 30)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(csv_matched['log10_True_SFR'], np.log10(csv_matched['Predicted_SFR']), s=10, alpha=0.6)
plt.xlabel('true SFR')
plt.ylabel('Predicted SFR (from COSMOS VLA)')
plt.title('True SFR vs Predicted SFR')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.show()

plt.scatter(csv_matched['log10_True_SFR'], csv_matched['Predicted_log10_L_1.4GHz_inverse_model'], s=10, alpha=0.6)
plt.xlabel('true SFR')
plt.ylabel('Predicted L (from COSMOS VLA)')
plt.title('True SFR vs Predicted Luminosity')
plt.ylim(20, 25)
plt.xlim(0, 5)
plt.show()

plt.scatter(np.log10(csv_matched['Predicted_SFR']), csv_matched['Predicted_log10_L_1.4GHz_inverse_model'], s=10, alpha=0.6)
plt.xlabel('Predicted SFR')
plt.ylabel('Predicted L (from COSMOS VLA)')
plt.title('Predicted SFR vs Predicted Luminosity')
plt.ylim(20, 25)
plt.xlim(0, 5)
plt.show()


'''
Convert predicted luminosity to flux density in VLA-COSMOS test set

Cosmology (Planck 2018 parameters)
'''

final_cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)

final_alpha = 0.7  # spectral index

def luminosity_to_flux(row):
    final_redshift = row['z_phot']
    final_log10_L = row['Predicted_log10_L_1.4GHz_inverse_model']
    
    final_D_L_m = final_cosmo.luminosity_distance(final_redshift).value * 3.085677581e22
    final_L = 10**final_log10_L  # W/Hz
    final_S_W_m2_Hz = final_L / (4 * np.pi * final_D_L_m**2) * (1 + final_redshift)**(1 - final_alpha)
    final_S_mJy = final_S_W_m2_Hz * 1e29
    
    return final_S_mJy

def true_luminosity_to_flux(row):
    final_redshift = row['z_phot']
    # FIX: convert from log scale
    true_L = 10**row['Radio_Luminosity_1.4GHz']  # log10(W/Hz) → W/Hz

    final_D_L_m = final_cosmo.luminosity_distance(final_redshift).value * 3.085677581e22
    true_S_W_m2_Hz = true_L / (4 * np.pi * final_D_L_m**2) * (1 + final_redshift)**(1 - final_alpha)
    true_S_mJy = true_S_W_m2_Hz * 1e29
    
    return true_S_mJy


# Apply both calculations row-wise
csv_matched['Predicted_flux_mJy'] = csv_matched.apply(luminosity_to_flux, axis=1)
csv_matched['True_flux_mJy'] = csv_matched.apply(true_luminosity_to_flux, axis=1)

# Save to CSV
csv_matched.to_csv('the_matched_predicted_L_1.4GHz_from_SFR_flux_density.csv', index=False)


mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'

# Set style
#sns.set(style="whitegrid")

# Define min/max for consistent axes
min_val = min(csv_matched['Predicted_flux_mJy'].min(), csv_matched['True_flux_mJy'].min())
max_val = max(csv_matched['Predicted_flux_mJy'].max(), csv_matched['True_flux_mJy'].max())

# ---------- Original Plot ----------
plt.figure(figsize=(10, 8))
plt.scatter(deblended_table['f20cm'][-1546:]/100,
            csv_matched['Predicted_flux_mJy'],
            c='dodgerblue', alpha=0.7, edgecolors='k', s=60)
#plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Observed Flux Density (mJy)', fontsize=14)
plt.ylabel('ML Predicted Flux Density (mJy)', fontsize=14)
plt.title('Predicted vs Observed 1.4 GHz Flux Density', fontsize=14)
plt.tight_layout()
plt.savefig('pred_vs_true_flux_density_cosmos.png', dpi=300)
plt.legend()
plt.show()



# ---------- Colored by SFR (True or Predicted) ----------
plt.figure(figsize=(8, 6))
sc = plt.scatter(csv_matched['True_flux_mJy'],
                 csv_matched['Predicted_flux_mJy'],
                 c=np.log10(csv_matched['Predicted_SFR']/csv_matched['log10_True_SFR']),  # or 'True_SFR'
                 cmap='plasma', alpha=0.7,
                 edgecolors='k', s=60,
                 vmin=-1,
                 vmax=2.5)
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('True Flux Density [mJy]', fontsize=12)
plt.ylabel('Predicted Flux Density [mJy]', fontsize=12)
plt.title('Colored by Predicted log10 SFR Ratio  (Predicted / True)\nCOSMOS Test set', fontsize=14)
plt.colorbar(sc, label='Predicted SFR')
plt.legend()
plt.tight_layout()
plt.show()


# ---------- Colored by Flux Ratio (Predicted / True) ----------
flux_ratio = csv_matched['Predicted_flux_mJy'] / csv_matched['True_flux_mJy']

plt.figure(figsize=(8, 6))
sc = plt.scatter(csv_matched['True_flux_mJy'],
                 csv_matched['Predicted_flux_mJy'],
                 c=np.log10(flux_ratio),
                 cmap='coolwarm', alpha=0.7,
                 edgecolors='k', s=60,
                 vmin=-3,
                 vmax=1)
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('True Flux Density [mJy]', fontsize=12)
plt.ylabel('Predicted Flux Density [mJy]', fontsize=12)
plt.title('Colored by Flux Ratio (Predicted / True) COSMOS Test set', fontsize=14)
plt.colorbar(sc, label='Flux Ratio')
plt.legend()
plt.tight_layout()
plt.show()





# ---------- Colored by Redshift ----------
redshift_colour_matched = csv_matched['z_phot']

plt.figure(figsize=(8, 6))
sc = plt.scatter(csv_matched['True_flux_mJy'],
                 csv_matched['Predicted_flux_mJy'],
                 c=redshift_colour_matched,
                 cmap='plasma', alpha=0.7,
                 edgecolors='k', s=60,
                 vmin=0,
                 vmax=2)
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('True Flux Density [mJy]', fontsize=12)
plt.ylabel('Predicted Flux Density [mJy]', fontsize=12)
plt.title('Colored by Redshift COSMOS Test set', fontsize=14)
plt.colorbar(sc, label='Redshift')
plt.legend()
plt.tight_layout()
plt.show()





'''
COSMOS test-set
'''

# Set style
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))

# Plot histograms
bins = 30  # You can adjust this depending on your data distribution
plt.hist(csv_matched['log10_True_SFR'], bins=bins, alpha=0.6, label='Log10 true SFR', color='steelblue', edgecolor='black')
plt.hist(np.log10(csv_matched['Predicted_SFR']), bins=bins, alpha=0.6, label='Predicted SLog10 FR', color='orange', edgecolor='black')

# Labels and title
plt.xlabel('Log10 Star Formation Rate (SFR)', fontsize=12)
plt.ylabel('No. of Sources', fontsize=12)
plt.title('Histogram of Sources by Log10 SFR COSMOS', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))

# Compute the ratio (True SFR / Predicted SFR)
sfr_ratio = np.log10(csv_matched['Predicted_SFR'] / 10**csv_matched['log10_True_SFR'] )

# Optionally, handle any infinities or NaNs (e.g., predicted SFR = 0)
sfr_ratio = sfr_ratio.replace([np.inf, -np.inf], np.nan).dropna()

# Set style
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
plt.hist(sfr_ratio, bins=30, color='purple', alpha=0.7, edgecolor='black')

plt.xlabel('Predicted SFR / True SFR', fontsize=12)
plt.ylabel('Number of Sources', fontsize=12)
plt.title('Histogram of Predicted/True Log10 SFR Ratio COSMOS', fontsize=14)
plt.tight_layout()
plt.show()





# Load the CSV file
final_csv_df = pd.read_csv('/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/the_matched_predicted_L_1.4GHz_from_SFR_flux_density.csv')

# Convert RA, Dec columns to numeric explicitly and drop rows with invalid values
final_csv_df['RA'] = pd.to_numeric(final_csv_df['RA'], errors='coerce')
final_csv_df['Dec'] = pd.to_numeric(final_csv_df['Dec'], errors='coerce')
final_csv_df = final_csv_df.dropna(subset=['RA', 'Dec'])

# Load the FITS file
with fits.open('/Users/jason/Downloads/COSMOS_VLA_Deblended.fits') as final_hdul:
    final_fits_data = final_hdul[1].data
    
    # Convert each field to native byte order (little-endian)
    final_native_fits_data = {}
    for final_name in final_fits_data.names:
        final_col = final_fits_data[final_name]
        if final_col.dtype.byteorder not in ('=', '|'):
            final_col = final_col.byteswap().newbyteorder()
        final_native_fits_data[final_name] = final_col
    
    final_fits_df = pd.DataFrame(final_native_fits_data)

# Clean FITS data: keep Dec in range, and drop rows missing RA, Dec, or z_phot
final_fits_df = final_fits_df[(final_fits_df['Dec'] >= -90) & (final_fits_df['Dec'] <= 90)]
final_fits_df = final_fits_df.dropna(subset=['RA', 'Dec', 'z_phot'])

# Convert RA, Dec columns to numeric explicitly just in case, then to numpy arrays
final_fits_df['RA'] = pd.to_numeric(final_fits_df['RA'], errors='coerce')
final_fits_df['Dec'] = pd.to_numeric(final_fits_df['Dec'], errors='coerce')
final_fits_df = final_fits_df.dropna(subset=['RA', 'Dec'])  # re-drop if numeric coercion introduced NaNs

# Create SkyCoord objects safely using numpy arrays
final_csv_coords = SkyCoord(ra=final_csv_df['RA'].values * u.degree, dec=final_csv_df['Dec'].values * u.degree)
final_fits_coords = SkyCoord(ra=final_fits_df['RA'].values * u.degree, dec=final_fits_df['Dec'].values * u.degree)

# Match catalogs within 2 arcseconds
final_idx, final_d2d, _ = final_csv_coords.match_to_catalog_sky(final_fits_coords)
final_max_sep = 2 * u.arcsec
final_matched = final_d2d < final_max_sep

# Filter matched sources in csv and add z_phot from fits using matched indices
final_csv_matched = final_csv_df[final_matched].reset_index(drop=True)
final_matched_fits_indices = final_idx[final_matched]

# Add z_phot column from fits catalog to matched CSV rows
final_csv_matched['z_phot'] = final_fits_df.iloc[final_matched_fits_indices]['z_phot'].values
final_csv_matched['xf20cm'] = final_fits_df.iloc[final_matched_fits_indices]['xf20cm'].values

# Save output CSV
final_csv_matched.to_csv('csv_matched_with_zphot.csv', index=False)

# Also get matched FITS rows if needed
final_fits_matched = final_fits_df.iloc[final_matched_fits_indices].reset_index(drop=True)


# Initialize the new columns with NaNs in the full CSV DataFrame
final_csv_df['z_phot'] = np.nan
final_csv_df['xf20cm'] = np.nan

# Assign values to matched rows using matched indices
final_csv_df.loc[final_matched, 'z_phot'] = final_fits_df.iloc[final_matched_fits_indices]['z_phot'].values
final_csv_df.loc[final_matched, 'xf20cm'] = final_fits_df.iloc[final_matched_fits_indices]['xf20cm'].values



plt.scatter(final_csv_df['Predicted_log10_L_1.4GHz_inverse_model'], final_csv_df['xf20cm'])
plt.xlabel('Log10 of Predicted Flux Density (mJy ML)')
plt.ylabel('Log10 of Predicted Flux Density (mJy SED)')
plt.title('Scatter Plot of Log10 Predicted Flux Density vs Log10 xf20cm')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid(True)
plt.show()


'''
For ML: The ML model might not have converged, been trained on insufficient data,
or had hyperparameter settings that prevented it from learning any meaningful patterns.
It could be "underfitting" severely.

For SED Fitting: The SED fitting code might be failing to converge for most sources, 
using an inappropriate set of templates, or encountering severe degeneracies that lead
to essentially random parameter recovery for the flux density.

Correlation Analysis (Across Flux Bins)
Plot true and predicted
'''


# Bin the data by true flux
bins = np.arange(0, 1.05, 0.05)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
correlations = []

for i in range(len(bins)-1):
    bin_data = csv_matched[(csv_matched['True_flux_mJy'] >= bins[i]) & (csv_matched['True_flux_mJy'] < bins[i+1])]
    if len(bin_data) > 2:
        # Replace 'SFR' with your column name for Star Formation Rate
        corr, _ = spearmanr(bin_data['True_flux_mJy'], 10**bin_data['log10_True_SFR'])
        correlations.append(corr)
    else:
        correlations.append(np.nan)

plt.figure(figsize=(8, 5))
plt.plot(bin_centers, correlations, marker='o')
plt.xlabel("Obs. Flux Density [mJy]")
plt.ylabel("Spearman Correlation (True Flux vs SFR)")
plt.title("SFR Influence on Radio Flux (per Bin) COSMOS!")
plt.savefig("Spearman correlation Cosmos.png", dpi=300)
plt.grid(True)
plt.show()




#Piecewise Linear Regression (Breakpoint Detection)

#Drop rows with NaN or infinite values in relevant columns

cleaned_data = csv_matched[['True_flux_mJy', 'Predicted_flux_mJy']].replace([np.inf, -np.inf], np.nan).dropna()

# Sort by True Flux

sorted_data = cleaned_data.sort_values('True_flux_mJy')
x = sorted_data['True_flux_mJy'].values
y = sorted_data['Predicted_flux_mJy'].values

# Optional: Remove zero or near-zero fluxes to avoid log/fit issues
x = x[x > 0]
y = y[:len(x)]  # Ensure y matches x after filtering

# Fit two-piece linear model
model = pwlf.PiecewiseLinFit(x, y)
breakpoints = model.fit(2)  # Returns [x0, x_break, xN]

# Predict for plotting
x_hat = np.linspace(x.min(), x.max(), 100)
y_hat = model.predict(x_hat)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.5, s=20, label='Data')
plt.plot(x_hat, y_hat, 'r-', label='Piecewise Fit')
plt.axvline(breakpoints[1], color='purple', linestyle='--', label=f'Breakpoint: {breakpoints[1]:.2f} mJy')
plt.xlabel("True Flux [mJy]")
plt.ylabel("Predicted Flux [mJy]")
plt.legend()
plt.title("Piecewise Regression for SFR/AGN Transition")
plt.grid(True)
plt.show()


# Add ratio column
csv_matched['Flux_Ratio'] = csv_matched['Predicted_flux_mJy'] / csv_matched['True_flux_mJy']

# Sort and compute rolling median
sorted_flux = csv_matched.sort_values('True_flux_mJy')
rolling_ratio = sorted_flux['Flux_Ratio'].rolling(window=20, center=True).median()

plt.figure(figsize=(8, 5))
plt.plot(sorted_flux['True_flux_mJy'], rolling_ratio, label='Rolling Median (Ratio)')
plt.axhline(1.0, color='gray', linestyle='--', label='Ratio = 1')
plt.xlabel('True Flux [mJy]')
plt.ylabel('Predicted / True Flux Ratio')
plt.title('Flux Ratio Trend vs True Flux')
plt.grid(True)
plt.legend()
plt.show()

cluster_colors = ListedColormap(['red', 'blue'])

# Prepare data for clustering
X_raw = csv_matched[['True_flux_mJy', 'Flux_Ratio']].dropna()
X = np.log10(X_raw.values + 1e-5)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# Align labels with original data
filtered_csv = csv_matched.loc[X_raw.index].copy()
filtered_csv['cluster'] = labels
filtered_csv['log_flux'] = X[:, 0]
filtered_csv['log_ratio'] = X[:, 1]

# Determine AGN type
def agn_type(row):
    if row['XRAY_AGN']:
        return 'XRAY_AGN'
    elif row['MIR_AGN']:
        return 'MIR_AGN'
    elif row['SED_AGN']:
        return 'SED_AGN'
    else:
        return 'Non-AGN'

filtered_csv['AGN_Type'] = filtered_csv.apply(agn_type, axis=1)

# Marker map
marker_map = {
    'XRAY_AGN': 'o',
    'MIR_AGN': '^',
    'SED_AGN': 's',
    'Non-AGN': 'x'
}

# Plot clustered sources with different AGN markers
plt.figure(figsize=(10, 8))
for agn_type, marker in marker_map.items():
    subset = filtered_csv[filtered_csv['AGN_Type'] == agn_type]
    plt.scatter(
        10**subset['log_flux'], 10**subset['log_ratio'],
        c=subset['cluster'], cmap=cluster_colors,
        marker=marker, alpha=0.6, label=agn_type
    )

# Create custom black markers for the legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='XRAY_AGN',
           markerfacecolor='black', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='^', color='w', label='MIR_AGN',
           markerfacecolor='black', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='s', color='w', label='SED_AGN',
           markerfacecolor='black', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='x', color='black', label='Non-AGN',
           markersize=8, linestyle='None')
]

# Add the custom legend to the plot
plt.legend(handles=legend_elements, title='AGN Type')


plt.xlabel('True Flux [mJy]')
plt.ylabel('Flux Ratio (Predicted/True)')
plt.xscale('log')
plt.yscale('log')
plt.title('GMM Clustering with AGN Types (COSMOS)')
#plt.legend(title='AGN Type')
plt.grid(True)
# Save before showing
plt.savefig('gmm_clustering_with_agn_markers.png', dpi=300, bbox_inches='tight')
plt.show()



# Step 1: Extract the means (in log10 space)
means = gmm.means_  # shape (2, 2): [log_flux, log_ratio]

# Step 2: Sort means by log_flux to ensure consistent order
means = means[np.argsort(means[:, 0])]

log_flux1 = means[0, 0]
log_flux2 = means[1, 0]

# Step 3: Compute midpoint and uncertainty
log_flux_break = (log_flux1 + log_flux2) / 2
log_flux_error = np.abs(log_flux2 - log_flux1) / 2

# Convert back to linear space for human-readable result
flux_break = 10**log_flux_break
flux_error = 10**(log_flux_break + log_flux_error) - flux_break

print(f"GMM flux breakpoint: {flux_break:.4f} mJy ± {flux_error:.4f} mJy")
print(f"(Log-space: log10(flux) = {log_flux_break:.4f} ± {log_flux_error:.4f})")


# ------
# Task 2: Fraction of non-AGN sources in each cluster
# ------
# Count AGN per cluster (XRAY_AGN, MIR_AGN, SED_AGN)
cluster_stats = (
    filtered_csv.groupby('cluster')['AGN_Type']
    .apply(lambda g: g.isin(['XRAY_AGN', 'MIR_AGN', 'SED_AGN']).sum() / len(g))
    .reset_index(name='AGN_Fraction')
)

# Sort by AGN fraction descending
cluster_stats_sorted = cluster_stats.sort_values(by='AGN_Fraction', ascending=True).reset_index(drop=True)

# Use the same cluster colors as the scatter plot
# Map cluster to its original color
color_map = {0: 'red', 1: 'blue'}
bar_colors = [color_map[c] for c in cluster_stats_sorted['cluster']]

# Plot bar chart with consistent colors
# Manually define labels for each bar
xtick_labels = ['Cluster 0', 'Cluster 1']  # <-- Your custom labels

# Plot
plt.figure(figsize=(5, 4))
plt.bar(
    [0, 1],  # bar positions
    cluster_stats_sorted['AGN_Fraction'],
    color=bar_colors
)
plt.xticks(ticks=[0, 1], labels=xtick_labels)
plt.ylabel('Fraction of AGN Sources')
plt.title('AGN Fraction per GMM Cluster (COSMOS)', pad=20)  # default is ~6
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('agn_fraction_per_cluster.png', dpi=300, bbox_inches='tight')
plt.show()



# --- Helper functions ---
def r500_to_angular_deg(r500_mpc, z):
    """
    Convert r500 (in Mpc) at redshift z to an angular size in degrees using Planck18 cosmology.
    """
    ang_rad = r500_mpc / cosmo.angular_diameter_distance(z).value
    return np.degrees(ang_rad)

def angular_distance_deg(ra1, dec1, ra2, dec2):
    """
    Great-circle separation between (ra1, dec1) and (ra2, dec2) in degrees.
    """
    c1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    c2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
    return c1.separation(c2).deg


# --- Main analysis ---
def run_gmm_spearman_analysis(cluster_name, data, exclude_outliers, save_dir='./'):
    """
    Run GMM + binned Spearman correlation analysis and save a 2-panel plot.

    Parameters
    ----------
    cluster_name : str
        Name of the cluster (used for plot title & filename).
    data : pandas.DataFrame
        Must contain columns:
          - 'Predicted_SFR_Msun_per_yr_from_L'
          - 'SFR_Msun_per_yr'
          - 'observed_flux_in_cluster'   (mJy, positive)
          - 'flux_ratio'                 (Predicted/Observed flux density ratio or similar)
    exclude_outliers : bool
        If True, exclude points with SFR ratio > 2σ from the mean before analysis.
    save_dir : str
        Directory where the PNG will be saved.

    Returns
    -------
    dict with keys:
      - 'cluster'
      - 'gmm_breakpoint' (linear mJy)
      - 'spearman_breakpoint' (linear mJy, center of bin with |rho| max)
      - 'max_rho'
      - 'combined_pval'
    """

    if data is None or data.empty:
        print(f"⚠️ No data for {cluster_name} in this subset, skipping.")
        return None

    # --- Compute SFR ratio and outliers ---
    gmm_sfr_ratio = data['Predicted_SFR_Msun_per_yr_from_L'] / data['SFR_Msun_per_yr']
    sfr_ratio_mean = gmm_sfr_ratio.mean()
    sfr_ratio_std = gmm_sfr_ratio.std()
    outlier_mask = np.abs(gmm_sfr_ratio - sfr_ratio_mean) > 2 * sfr_ratio_std

    if exclude_outliers:
        data_filtered = data.loc[~outlier_mask].copy()
    else:
        data_filtered = data.copy()

    if data_filtered.empty:
        print(f"⚠️ No data left after excluding outliers for {cluster_name}, skipping.")
        return None

    # --- Prepare arrays for Spearman binning ---
    # Add small epsilon to avoid log(0)
    eps = 1e-5
    flux = data_filtered['observed_flux_in_cluster'].to_numpy()
    log_flux = np.log10(flux + eps)
    ratio = data_filtered['flux_ratio'].to_numpy()

    # Sort by log_flux for contiguous bins
    sort_idx = np.argsort(log_flux)
    log_flux_sorted = log_flux[sort_idx]
    ratio_sorted = ratio[sort_idx]

    # Bin parameters
    bin_fraction = 0.10           # 10% per bin
    min_points_per_bin = 20
    N = len(log_flux_sorted)
    bin_size = max(int(N * bin_fraction), min_points_per_bin)

    bin_centers = []
    spearman_rhos = []
    spearman_pvals = []

    for start in range(0, N, bin_size):
        end = start + bin_size
        if end > N:
            break
        x_bin = log_flux_sorted[start:end]
        y_bin = ratio_sorted[start:end]
        rho, pval = spearmanr(x_bin, y_bin)
        if np.isnan(rho):
            rho, pval = 0.0, 1.0
        # Convert median in log-space back to linear mJy for plotting on log-x
        bin_centers.append(np.median(10**x_bin))
        spearman_rhos.append(rho)
        spearman_pvals.append(pval)

    # --- GMM on log10(flux) & log10(ratio) ---
    X_for_gmm = np.log10(data_filtered[['observed_flux_in_cluster', 'flux_ratio']].to_numpy() + eps)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(X_for_gmm)
    labels = gmm.predict(X_for_gmm)

    print(f"✅ GMM applied for {cluster_name}. Found {gmm.n_components} components.")

    # Make labels consistent by mean flux_ratio in original (linear) space
    # Map each component id -> mean flux_ratio
    comp_ids = np.unique(labels)
    comp_means = {cid: data_filtered['flux_ratio'].to_numpy()[labels == cid].mean()
                  for cid in comp_ids}
    # Sort components so that component 0 has lower mean flux_ratio
    order = sorted(comp_means, key=lambda cid: comp_means[cid])
    remap = {order[0]: 0, order[1]: 1}
    consistent_labels = np.vectorize(remap.get)(labels)

    # --- GMM breakpoint + half-distance band (computed in log space) ---
    # Means are in (log10(flux), log10(ratio)); take the flux axis (col 0)
    cluster_centers = gmm.means_
    x_means_log = cluster_centers[:, 0]
    x_sep_log = np.mean(x_means_log)  # midpoint in log10 space
    half_dist_log = 0.5 * np.abs(x_means_log[1] - x_means_log[0])

    # Convert back to linear for plotting on log x-axis (asymmetric in linear space by design)
    x_sep_linear = 10**x_sep_log
    x_low_linear = 10**(x_sep_log - half_dist_log)
    x_high_linear = 10**(x_sep_log + half_dist_log)
    gmm_breakpoint = x_sep_linear

    # --- Determine Spearman "breakpoint" as bin with max |rho| ---
    if spearman_rhos:
        max_rho_idx = int(np.argmax(np.abs(spearman_rhos)))
        spearman_breakpoint = float(bin_centers[max_rho_idx])
        max_rho = float(spearman_rhos[max_rho_idx])
    else:
        spearman_breakpoint = np.nan
        max_rho = np.nan

    # --- Combine p-values with Fisher's method ---
    pvals_array = np.clip(np.asarray(spearman_pvals, dtype=float), 1e-15, 1.0)
    statistic = -2.0 * np.sum(np.log(pvals_array))
    dof = 2 * len(pvals_array)
    combined_pval = 1 - chi2.cdf(statistic, dof)

    print(f"Combined Fisher p-value for cluster {cluster_name}: {combined_pval:.3e}")
    print(f"Spearman p-values for each bin in cluster {cluster_name}:")
    for center, pval in zip(bin_centers, spearman_pvals):
        print(f"  Bin center (Observed Flux) ≈ {center:.4f} mJy, p-value = {pval:.4e}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
    ax1, ax2 = axes

    # Panel 1: Scatter with GMM coloring
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Observed Flux Density [mJy]', fontsize=14)
    ax1.set_ylabel('Flux Density Ratio (Predicted/Observed)', fontsize=14)
    ax1.set_title(f'GMM Clustering — {cluster_name}', fontsize=14)
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    if exclude_outliers:
        # plot only non-outliers
        sc = ax1.scatter(
            data.loc[~outlier_mask, 'observed_flux_in_cluster'],
            data.loc[~outlier_mask, 'flux_ratio'],
            c=consistent_labels[outlier_mask.values == False],
            cmap='coolwarm', alpha=0.7, marker='o'
        )
    else:
        # plot non-outliers
        ax1.scatter(
            data.loc[~outlier_mask, 'observed_flux_in_cluster'],
            data.loc[~outlier_mask, 'flux_ratio'],
            c=consistent_labels[outlier_mask.values == False],
            cmap='coolwarm', alpha=0.7, marker='o'
        )
        # (Optional) plot outliers differently; uncomment to show triangles
        # ax1.scatter(
        #     data.loc[outlier_mask, 'observed_flux_in_cluster'],
        #     data.loc[outlier_mask, 'flux_ratio'],
        #     c='black', alpha=0.7, marker='^', label='>2σ SFR Ratio'
        # )
        
    err_minus_linear = x_sep_linear - x_low_linear
    err_plus_linear  = x_high_linear - x_sep_linear
    err_linear_avg   = 0.5 * (err_minus_linear + err_plus_linear)

    # GMM breakpoint & half-distance band on scatter panel
    ax1.axvline(x_sep_linear, color='k', linestyle='--', linewidth=1, zorder=3,
                label=f'GMM break = {x_sep_linear:.2g} mJy')
    ax1.axvspan(x_low_linear, x_high_linear, color='gray', alpha=0.2, zorder=0,
            label=f'Half-distance ±{err_linear_avg:.2g} mJy')
    ax1.legend(loc='best')

    # Panel 2: Spearman vs flux (log-x)
    display_pval = "<1e-15" if combined_pval < 1e-15 else f"{combined_pval:.2e}"
    spearman_label = f"Spearman ρ (combined p={display_pval})"
    ax2.plot(bin_centers, spearman_rhos, marker='o', label=spearman_label)

    # GMM breakpoint & half-distance band on Spearman panel
    ax2.axvline(x_sep_linear, color='k', linestyle='--', linewidth=1,
                label=f'GMM break = {x_sep_linear:.2g} mJy')
    ax2.axvspan(x_low_linear, x_high_linear, color='gray', alpha=0.3,
            label=f'Half-distance ±{err_linear_avg:.2g} mJy')

    ax2.set_xscale('log')
    ax2.set_xlabel("Observed Flux Density (mJy)", fontsize=14)
    ax2.set_ylabel("Spearman Correlation", fontsize=14)
    ax2.set_title(f"Spearman Correlation: {cluster_name}", fontsize=14)
    ax2.grid(True)

    # Match x-limits across panels
    ax2.set_xlim(ax1.get_xlim())
    ax2.legend(loc='best')

    plt.tight_layout()

    # --- Save plot ---
    os.makedirs(save_dir, exist_ok=True)
    filename_safe_cluster = str(cluster_name).replace(' ', '_')
    out_filepath = os.path.join(save_dir, f'gmm_spearman_{filename_safe_cluster}.png')
    plt.savefig(out_filepath, dpi=300)
    plt.close()

    print(f"📊 Saved plot: {out_filepath}")

    return {
        'cluster': cluster_name,
        'gmm_breakpoint': float(gmm_breakpoint),
        'spearman_breakpoint': float(spearman_breakpoint),
        'max_rho': float(max_rho),
        'combined_pval': float(combined_pval),
    }



# --- Main execution loop ---

# Your clusters, redshifts, r500 (Mpc) already defined
# You also need to define cluster_centers dictionary here with (RA_deg, Dec_deg) for each cluster

cluster_centers = {
    'Abell_133': (15.6879, -21.8800),
    'Abell_194': (21.4458, -1.3731),
    'Abell_209': (22.9896, -13.5764),
    'Abell_22': (5.1608, -25.7220),
    'Abell_2485': (342.1371, -16.1062),
    'Abell_2597': (351.3321, -12.1244),
    'Abell_2645': (355.3200, -2.0975),
    'Abell_2667': (357.9196, -26.0836),
    'Abell_2744': (3.5671, -30.3830),
    'Abell_2751': (4.0580, -31.3885),
    'Abell_2811': (10.5368, -28.5358),
    'Abell_2895': (19.6454, -26.9731),
    'Abell_3365': (87.0500, -21.9350),
    'Abell_3376': (90.4256, -39.9851),
    'Abell_3558': (201.9783, -31.4922),
    'Abell_3562': (202.7833, -31.6713),
    'Abell_3667': (303.1403, -56.8406),
    'Abell_370': (39.9604, -1.5856),
    'Abell_4038': (356.8796, -28.2028),
    'Abell_521': (73.5358, -10.2442),
    'Abell_545': (86.7571, -25.6164),
    'Abell_85': (10.4529, -9.3180),
    'ElGordo': (15.7188, -49.2495),
    'J0014.3-6604': (3.5767, -66.0775)  # Interpolated name as it’s cut off
}

r500 = [1.002, 0.9824/1.5, 1.330/1.5, 1.056, 1.0622, 1.1066, 1.0897, 1.3031, 1.2359, 0.7431, 
        1.0355, 1.0896, 0.81, 0.8726, 1.101, 0.9265, 1.199, 1.64/1.5, 0.8863,
        1.240, 1.35, 1.2, 6.8, 1.0117]


csv_directory = './'
exclude_outliers = False  # Toggle as needed

results = []

for i, cluster in enumerate(clusters):
    print(f"\n🔍 Processing cluster: {cluster}")

    if cluster not in cluster_centers:
        print(f"❌ No cluster center coordinates for {cluster}, skipping.")
        continue

    file_path = os.path.join(csv_directory, f"{cluster}_flux_ratio.csv")
    if not os.path.exists(file_path):
        print(f"❌ File not found for {cluster}: {file_path}")
        continue

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading {cluster} CSV: {e}")
        continue

    required_cols = [
        'observed_flux_in_cluster', 'predicted_flux_in_cluster',
        'flux_ratio', 'SFR_Msun_per_yr',
        'Predicted_SFR_Msun_per_yr_from_L', 'RA_deg', 'Dec_deg'
    ]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Missing required columns for {cluster}, skipping.")
        continue

    data = df[required_cols].dropna()
    if data.empty:
        print(f"⚠️ No valid data points for {cluster} after dropping NaNs.")
        continue

    # Get cluster center coords and r500 in angular degrees
    ra_center, dec_center = cluster_centers[cluster]
    z = redshifts[i]
    r500_mpc = r500[i]
    r500_deg = r500_to_angular_deg(r500_mpc, z)

    # Calculate angular distances for sources
    distances_deg = angular_distance_deg(data['RA_deg'].values, data['Dec_deg'].values, ra_center, dec_center)

    # Create masks
    inside_mask = distances_deg <= r500_deg
    outside_mask = distances_deg > r500_deg

    subsets = {
        'inside_r500': data.loc[inside_mask],
        'outside_r500': data.loc[outside_mask],
        'all_sources': data
    }

    for subset_name, subset_data in subsets.items():
        print(f"\n▶️ Running analysis for {cluster} — subset: {subset_name} ({len(subset_data)} sources)")
        res = run_gmm_spearman_analysis(
            cluster_name=f"{cluster}_{subset_name}",
            data=subset_data,
            exclude_outliers=exclude_outliers,
            save_dir=csv_directory
        )
        if res:
            results.append(res)

print("\n✨ All clusters processed!")





# ---- After the loop finishes ----

# Convert results list of dicts to DataFrame
breakpoints_df = pd.DataFrame(results)

# Extract subset (inside_r500, outside_r500, all_sources) from cluster column
# Assuming cluster name format: "<cluster_name>_<subset>"
breakpoints_df['subset'] = breakpoints_df['cluster'].apply(lambda x: x.split('_')[-2] + '_' + x.split('_')[-1] if len(x.split('_')) > 2 else 'all_sources')

# Or, if cluster names are like 'Abell_133_inside_r500', 'Abell_133_outside_r500'
# you can just do:
breakpoints_df['subset'] = breakpoints_df['cluster'].apply(lambda x: '_'.join(x.split('_')[-2:]))

# Drop rows with NaN gmm_breakpoint or spearman_breakpoint if any
breakpoints_df_clean = breakpoints_df.dropna(subset=['gmm_breakpoint', 'spearman_breakpoint'])

# Group by subset and compute statistics
summary_stats = []

for subset_name, group in breakpoints_df_clean.groupby('subset'):
    mean_gmm = group['gmm_breakpoint'].mean()
    std_gmm = group['gmm_breakpoint'].std()
    median_gmm = group['gmm_breakpoint'].median()
    mad_gmm = median_abs_deviation(group['gmm_breakpoint'], scale='normal')

    mean_spearman = group['spearman_breakpoint'].mean()
    std_spearman = group['spearman_breakpoint'].std()
    median_spearman = group['spearman_breakpoint'].median()
    mad_spearman = median_abs_deviation(group['spearman_breakpoint'], scale='normal')

    summary_stats.append({
        'subset': subset_name,
        'mean_gmm': mean_gmm,
        'std_gmm': std_gmm,
        'median_gmm': median_gmm,
        'mad_gmm': mad_gmm,
        'mean_spearman': mean_spearman,
        'std_spearman': std_spearman,
        'median_spearman': median_spearman,
        'mad_spearman': mad_spearman
    })

# Convert summary to DataFrame for easier viewing
summary_df = pd.DataFrame(summary_stats)

# Print summary nicely
for idx, row in summary_df.iterrows():
    print(f"\n📌 Summary statistics for subset: {row['subset']}")
    print(f"Mean GMM Breakpoint: {row['mean_gmm']:.3f} ± {row['std_gmm']:.3f}")
    print(f"Median GMM Breakpoint: {row['median_gmm']:.3f} ± {row['mad_gmm']:.3f}")
    print(f"Mean Spearman Breakpoint: {row['mean_spearman']:.3f} ± {row['std_spearman']:.3f}")
    print(f"Median Spearman Breakpoint: {row['median_spearman']:.3f} ± {row['mad_spearman']:.3f}")



# Optional: sort by subset and cluster for readability
breakpoints_df_clean = breakpoints_df_clean.sort_values(by=['subset', 'cluster'])

# Print individual breakpoints per cluster
for subset_name, group in breakpoints_df_clean.groupby('subset'):
    print(f"\n📌 Breakpoints for subset: {subset_name}")
    print(" ──────────────────────────────────────────────────────────────")
    for _, row in group.iterrows():
        print(f" {row['cluster']:<30} | GMM: {row['gmm_breakpoint']:.3f} | Spearman: {row['spearman_breakpoint']:.3f}")


# Ensure DataFrame is clean and sorted
breakpoints_df_clean = breakpoints_df.dropna(subset=['gmm_breakpoint', 'spearman_breakpoint'])
breakpoints_df_clean = breakpoints_df_clean.sort_values(by=['subset', 'cluster'])

# Group by subset first, then by cluster within each subset
for subset_name, subset_group in breakpoints_df_clean.groupby('subset'):
    print(f"\n📌 Breakpoints for subset: {subset_name}")
    print(" ───────────────────────────────────────────────────────────────────────────────")
    print(f"{'Cluster':<30} | {'GMM (mean ± std)':<25} | {'Spearman (mean ± std)':<25}")
    print("-" * 85)
    
    for cluster_name, cluster_group in subset_group.groupby('cluster'):
        gmm_mean = cluster_group['gmm_breakpoint'].mean()
        gmm_std = cluster_group['gmm_breakpoint'].std()

        spearman_mean = cluster_group['spearman_breakpoint'].mean()
        spearman_std = cluster_group['spearman_breakpoint'].std()

        print(f"{cluster_name:<30} | {gmm_mean:.3f} ± {gmm_std:.3f}         | {spearman_mean:.3f} ± {spearman_std:.3f}")

print(group['gmm_breakpoint'])
print(group['spearman_breakpoint'])



# Prepare labels and x-axis positions
labels = summary_df['subset']
x = np.arange(len(labels))

# -------- 3. Scatter Plot with Mean/Median and Error Bars --------
fig, ax = plt.subplots(figsize=(10,6))

# Plot mean with std
ax.errorbar(x - 0.1, summary_df['mean_gmm'], yerr=summary_df['std_gmm'], fmt='o', label='Mean GMM ± Std', color='b')
ax.errorbar(x + 0.1, summary_df['median_gmm'], yerr=summary_df['mad_gmm'], fmt='s', label='Median GMM ± MAD', color='b', alpha=0.7)

ax.errorbar(x - 0.1, summary_df['mean_spearman'], yerr=summary_df['std_spearman'], fmt='o', label='Mean Spearman ± Std', color='orange')
ax.errorbar(x + 0.1, summary_df['median_spearman'], yerr=summary_df['mad_spearman'], fmt='s', label='Median Spearman ± MAD', color='orange', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Breakpoint Value')
ax.set_title('Mean and Median Breakpoints with Error Bars')
ax.legend()
plt.tight_layout()
plt.show()





labels = summary_df['subset']
x = np.arange(len(labels))

# Extract values and errors
mean_gmm = summary_df['mean_gmm']
median_gmm = summary_df['median_gmm']
mean_gmm_err = summary_df['std_gmm']
median_gmm_err = summary_df['mad_gmm']

mean_spearman = summary_df['mean_spearman']
median_spearman = summary_df['median_spearman']
mean_spearman_err = summary_df['std_spearman']
median_spearman_err = summary_df['mad_spearman']

# Create side-by-side subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

# GMM subplot
axs[0].bar(x - 0.15, mean_gmm, 0.3, yerr=mean_gmm_err, capsize=5,
           label='Mean', color='steelblue')
axs[0].bar(x + 0.15, median_gmm, 0.3, yerr=median_gmm_err, capsize=5,
           label='Median', color='skyblue')
axs[0].set_title('GMM Breakpoints')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=45, ha='right')
axs[0].legend()
axs[0].grid(True, axis='y', linestyle='--', alpha=0.7)

# Spearman subplot
axs[1].bar(x - 0.15, mean_spearman, 0.3, yerr=mean_spearman_err, capsize=5,
           label='Mean', color='indianred')
axs[1].bar(x + 0.15, median_spearman, 0.3, yerr=median_spearman_err, capsize=5,
           label='Median', color='salmon')
axs[1].set_title('Spearman Breakpoints')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels, rotation=45, ha='right')
axs[1].legend()
axs[1].grid(True, axis='y', linestyle='--', alpha=0.7)

# Main title and layout
fig.suptitle('Breakpoints Comparison (Mean vs Median)', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

plt.savefig("breakpoint_comparison.png", dpi=300)
plt.show()


'''
Test plots just to visualise different features and statistics
'''


# ======= Load your merged dataset =======
df = pd.read_csv("the_matched_predicted_L_1.4GHz_from_SFR_flux_density.csv")

# Remove invalid values
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[
    'Radio_Luminosity_1.4GHz', 'log10_True_SFR', 'Predicted_SFR'
])

# ======= Matplotlib ApJ-like style =======
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# ======= 1. SFR vs Radio Luminosity =======
fig, ax = plt.subplots(figsize=(6,5))

# scatter
ax.scatter(10**df["Radio_Luminosity_1.4GHz"], 10**df["log10_True_SFR"],
           s=20, c='k', alpha=0.7, edgecolors='none')

# regression in log-log space with uncertainties
x_log = df["Radio_Luminosity_1.4GHz"]
y_log = df["log10_True_SFR"]

coeffs, cov = np.polyfit(x_log, y_log, 1, cov=True)
slope, intercept = coeffs
slope_err, intercept_err = np.sqrt(np.diag(cov))

xvals_log = np.linspace(x_log.min(), x_log.max(), 100)
yfit_log = intercept + slope * xvals_log

# 1σ bounds
yfit_upper = (intercept + intercept_err) + (slope + slope_err) * xvals_log
yfit_lower = (intercept - intercept_err) + (slope - slope_err) * xvals_log

# plot regression line
ax.plot(10**xvals_log, 10**yfit_log, color='red', lw=1,
        label=f"Slope={slope:.2f}±{slope_err:.2f}")

# shaded 1σ uncertainty band
ax.fill_between(10**xvals_log, 10**yfit_lower, 10**yfit_upper,
                color='red', alpha=0.2)

# ======= Pearson correlation with error =======
r, _ = pearsonr(x_log, y_log)
n = len(x_log)
r_err = (1 - r**2) / np.sqrt(n - 3)  # Fisher's z standard error

# Legend with slope and r
ax.legend([f"Slope={slope:.2f}±{slope_err:.2f}",
           f"r = {r:.2f} ± {r_err:.2f}"],
          frameon=False)

# log scales and labels
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L_{\mathrm{1.4\,GHz}}\,[\mathrm{W\,Hz^{-1}}]$")
ax.set_ylabel(r"SFR$_\mathrm{total}$ [$M_\odot\,\mathrm{yr^{-1}}$]")
fig.tight_layout()

# save figure
plt.savefig("plot1_sfr_vs_radio_with_uncertainty.png", dpi=300)
plt.show()




# ======= 2. Fractional Radio SFR Contribution =======
fig, ax = plt.subplots(figsize=(6,5))
frac = df["Predicted_SFR"] / df["log10_True_SFR"]**10
ax.scatter(df["log10_True_SFR"]**10, frac, c='k', alpha=0.7, s=20)
ax.axhline(0.1, color='gray', linestyle='--', lw=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Total SFR [$M_\odot\,\mathrm{yr^{-1}}$]")
ax.set_ylabel(r"$\mathrm{SFR_{radio}} / \mathrm{SFR_{total}}$")
fig.tight_layout()
plt.savefig("plot2_fractional_contribution.pdf")



# ======= 3. Cumulative Distribution of Radio Contribution =======
fig, ax = plt.subplots(figsize=(6,5))
frac_sorted = np.sort(frac)
cdf = np.arange(1, len(frac_sorted)+1) / len(frac_sorted)
ax.plot(frac_sorted, cdf, color='k')
ax.set_xscale("log")
ax.set_xlabel(r"$\mathrm{SFR_{radio}} / \mathrm{SFR_{total}}$")
ax.set_ylabel("Cumulative Fraction")
fig.tight_layout()
plt.savefig("plot3_cdf_radio_contribution.pdf")



# ======= 4. Stacked Bar Chart of Median Energy Budget =======
fig, ax = plt.subplots(figsize=(6,5))
median_ir = np.median(df["log10_True_SFR"]**10) if "SFR_IR" in df.columns else 0
median_radio = np.median(df["Predicted_SFR"])
median_total = np.median(df["log10_True_SFR"]**10)
median_uv = max(median_total - median_ir - median_radio, 0)

ax.bar(0, median_ir, color='orange', label='IR')
if median_uv > 0:
    ax.bar(0, median_uv, bottom=median_ir, color='blue', label='UV')
ax.bar(0, median_radio, bottom=median_ir+median_uv, color='green', label='Radio')

ax.set_ylabel(r"Median SFR [$M_\odot\,\mathrm{yr^{-1}}$]")
ax.set_xticks([])
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig("plot4_stacked_bar_budget.pdf")



# ======= 5. Radio vs IR SFR Residuals =======
if "log10_True_SFR" in df.columns:
    fig, ax = plt.subplots(figsize=(6,5))
    residuals = df["Predicted_SFR"] - df["log10_True_SFR"]**10
    ax.scatter(df["log10_True_SFR"]**10, residuals, c='k', alpha=0.7, s=20)
    ax.axhline(0, color='gray', linestyle='--', lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"SFR$_\mathrm{IR}$ [$M_\odot\,\mathrm{yr^{-1}}$]")
    ax.set_ylabel(r"SFR$_\mathrm{radio}$ - SFR$_\mathrm{total}$ [$M_\odot\,\mathrm{yr^{-1}}$]")
    fig.tight_layout()
    plt.savefig("plot5_radio_residuals.pdf")

print("✅ All plots saved as PDF.")


