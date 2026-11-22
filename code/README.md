# Sr-90 Radioactivity: Data Analysis and Physical Modelling

## Overview

This project investigates how the measured count rate from a Strontium- 90 (Sr-90) beta source varies with source– detector separation. The experimental dataset contains measurements at 32 separations, ranging from 0.010 m to 0.850 m.

The main objective is to investigate whether the measured distance dependence follows an inverse-square relationship, and whether a more complete description requires additional effects such as finite source and detector geometry, air attenuation and detector dead time.

The project combines experimental data analysis, statistical modelling, numerical integration and Monte Carlo simulation. Four Python scripts investigate the experimental measurements and complementary physical modelling approaches.

## Objectives

- Process repeated count measurements and calculate count rates with Poisson statistical uncertainties.
- Investigate inverse- square behaviour and compare alternative mathematical descriptions of the distance dependence.
- Account for background contributions, detector dead time and correlated measurement uncertainties.
- Model finite source and detector dimensions using numerical integration.
- Use Monte Carlo sampling to investigate geometric acceptance and approximate beta- electron transmission through air.
- Compare alternative model configurations and assess their agreement with the experimental data using goodness- of- fit statistics.

## Methods

### 1. Experimental analysis and five- model comparison — `Radioactivity Analysis.py`

The main analysis processes repeated measurements at each source– detector separation. Counts and acquisition times are pooled to calculate count rates, with uncertainties estimated using Poisson counting statistics. Background measurements are used to estimate and subtract the background contribution, and a non- paralysable dead- time correction is applied.

Five distance- dependence models are compared:

1. Inverse- square model.
2. Free power- law model.
3. Power law with exponential attenuation.
4. Finite geometry with attenuation and an additional power- law correction.
5. Finite geometry with attenuation.

The finite- geometry calculation accounts for the circular source window and square detector active area rather than treating them as point objects. Numerical quadrature is used to evaluate the contributions from pairs of source and detector surface points.

The fitting procedure uses covariance- aware generalised least squares. The covariance matrix accounts for Poisson counting uncertainty, shared background uncertainty, distance uncertainty and the common distance- offset contribution. Cholesky factorisation is used to whiten the residuals before nonlinear least- squares optimisation.

The script also investigates the corrected quantity R(d) × d², fits a constant over selected distance intervals and examines an effective exponential description of the long- distance behaviour. The interval scan is exploratory because the overlapping intervals are not independent tests.

### 2. Finite- geometry Monte Carlo simulation — `Monte Carlo Exponent Fit.py`

This script investigates how finite source and detector dimensions affect geometric acceptance.

A circular source with a diameter of 9.75 mm and a square detector measuring 10 × 10 mm are modelled. Source and detector positions are sampled uniformly over their respective surfaces. Each sampled source– detector pair is weighted according to its isotropic solid- angle contribution.

The simulation evaluates 24 separations between 0.05 m and 1.00 m, using 100,000 sampled source–detector pairs per separation. A power- law model is fitted to the resulting geometric acceptance curve, and the simulation is repeated independently 100 times.

The fitted exponents are collected to examine their distribution and the variability of the estimated geometric acceptance.

This simulation isolates geometric effects. It does not include energy loss, air attenuation, multiple scattering, detector efficiency or dead time.

### 3. Beta- spectrum air- transmission Monte Carlo — `Monte Carlo Air Attenuation.py`

This script estimates the approximate transmission of Sr-90/ Y-90 beta electrons through air.

Electron kinetic energies are sampled from approximate allowed beta spectra for Sr-90 and Y-90, using endpoint energies of 0.546 MeV and 2.28 MeV respectively, with an assumed 50:50 mixture.

The sampled energies are converted into approximate continuous- slowing- down approximation (CSDA) ranges using an empirical range– energy relationship. For each air path length, an electron is counted as transmitted if its initial CSDA range exceeds the corresponding air areal density.

The simulation evaluates 18 path lengths between 0 and 0.85 m, using 1,000,000 sampled beta energies per run and 100 independent simulations. The transmission curves are fitted with an effective exponential model to estimate an effective attenuation coefficient.

The model uses a simplified straight- line range approximation. It does not include a Fermi- function correction, detailed energy- loss straggling, multiple scattering, source encapsulation, backscatter or detector response. The fitted coefficient is therefore an effective result of this approximate model, rather than a universal attenuation coefficient or a direct measurement.

### 4. Systematic model comparison — `All Models Test.py`

This script performs a separate comparison of 64 model configurations against the 32- distance experimental dataset.

The configurations combine four distance- dependence families:

- Inverse- square.
- Free power law.
- Finite geometry.
- Finite geometry with an additional empirical power adjustment.

Each family is evaluated with different combinations of exponential attenuation, an additional constant signal component, non- paralysable dead- time correction and distance- offset assumptions. The two distance- offset scenarios are 0 mm and 10.5 mm.

The fits use covariance-weighted generalised least squares. The uncertainty treatment includes Poisson counting uncertainty, correlated background uncertainty, ruler- distance uncertainty and the shared distance- offset contribution. Gauss– Hermite quadrature is used to account for distance uncertainty.

The successful configurations are compared using chi-squared, reduced chi-squared, degrees of freedom and p- values. This provides a systematic assessment of how different modelling assumptions and experimental corrections affect the agreement with the measured data.

The 10.5 mm distance offset and 2.5 μs dead-time value are fixed scenarios rather than independently fitted measurements. The best- ranked configuration should not automatically be interpreted as the physically correct model.

## Results

The main five- model analysis found that all five full- range models were rejected at the 5% significance level.

Key results include:

- The inverse- square model gave a reduced chi-squared of 50.19, indicating a poor fit across the full measurement range.
- The free power- law model gave a fitted exponent of 2.098 ± 0.004. Although this is close to the inverse- square exponent of 2, its goodness of fit remained poor.
- Including effective exponential attenuation substantially improved the fit, giving a reduced chi-squared of 12.15.
- Adding finite geometry and an additional power- law correction produced a further improvement, giving a reduced chi-squared of 11.77.
- The corrected R(d) × d² quantity was consistent with a constant over the selected interval from 0.200 m to 0.325 m, with a reduced chi-squared of 1.56 and p- value of 0.166. This is an exploratory interval result, not evidence that inverse- square behaviour holds across the full range.
- Applying the dead- time correction improved the best- model chi-squared by approximately 2.2%, but did not resolve the remaining disagreement.

Overall, attenuation improved the description of the measured distance dependence, but none of the tested full- range models provided a satisfactory complete explanation of the experimental data.

The supplementary simulations investigate geometric acceptance and approximate beta- electron transmission separately. They provide insight into individual physical effects but do not constitute a complete simulation of the experimental apparatus.

## Technologies

- **Python** — implementation of the analysis and simulation workflows.
- **NumPy** — numerical arrays, data processing, vectorised calculations and random sampling.
- **SciPy** — nonlinear curve fitting, least- squares optimisation, numerical integration, statistical distributions and linear algebra.
- **Matplotlib** — visualisation of experimental measurements, fitted models and Monte Carlo results.


## Running the code

### Requirements

Use Python 3 with the following packages installed:

```bash
python -m pip install numpy scipy matplotlib
```

### Running the scripts

From the root directory of the repository, run each script using:

```bash
python "code/Radioactivity Analysis.py"
python "code/Monte Carlo Exponent Fit.py"
python "code/Monte Carlo Air Attenuation.py"
python "code/All Models Test.py"
```

The experimental measurements required by `Radioactivity Analysis.py` and `All Models Test.py` are embedded directly in the scripts, so separate CSV input files are not required.

### Figure outputs

All four scripts generate figures using Matplotlib. The figures are both displayed and saved as PNG files in the current working directory.

- **`Radioactivity Analysis.py`** saves three figures: `Log Plot of Models.png`, `Rd^2 Plot.png` and `Range Scan.png`.
- **`Monte Carlo Exponent Fit.py`** saves `Exponent Fit.png` and `Exponent Histogram.png`.
- **`Monte Carlo Air Attenuation.py`** saves `Air Attenuation Fit.png` and `Air Attenuation Histogram.png`.
- **`All Models Test.py`** saves `All Models Comparison.png`.

The saved figures are written to the directory from which the scripts are executed, rather than automatically being placed in the repository's `figures/` folder.

### Reproducibility and computational requirements

The Monte Carlo scripts use fixed random seeds to make their sampled results reproducible. The air-attenuation simulation is computationally intensive, sampling 1,000,000 beta particles in each of 100 independent simulations.
