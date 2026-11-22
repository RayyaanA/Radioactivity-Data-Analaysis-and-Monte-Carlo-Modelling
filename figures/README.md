# Figures

This folder contains eight figures generated during the Sr-90 radioactivity investigation. They provide visual representations of the experimental data analysis, theoretical model comparisons and Monte Carlo simulations.

The figures are organised into four groups according to the analysis they support.

## 1. Main Experimental Analysis (`Radioactivity Analysis.py`)

**Range Scan – `Range Scan.png`**

A two-dimensional heatmap showing the goodness of fit of the inverse-square model across different contiguous intervals of the experimental dataset. The colour scale represents the logarithm of the p-value for each interval, highlighting regions where the model provides a more satisfactory description of the measured count rates.

**Logarithmic Model Comparison – `Log Plot of Models.png`**

A logarithmic plot comparing the measured Sr-90 count rates with five fitted theoretical models: the inverse-square model, free power-law model, power law with air attenuation, finite geometry with attenuation, and finite geometry with power correction and attenuation. Measurement uncertainties and the p-values associated with each fit are also shown.

**\(Rd^2\) Analysis – `Rd^2 Plot.png`**

A plot of the measured count rate multiplied by the square of the source-detector separation, \(Rd^2\), against separation. It illustrates the approximately constant region at shorter distances, the selected interval for the constant fit, and the subsequent decrease associated with air attenuation.

## 2. Extended Model Comparison (`All Models Test.py`)

**All-Model Reduced Chi-Squared Comparison – `All Models Comparison.png`**

A horizontal bar chart comparing the reduced chi-squared values of 64 model configurations. The configurations combine different distance-dependence models with variations in attenuation, additive constants, dead-time corrections and distance offsets. This provides a broader comparison of how these assumptions affect agreement with the experimental data.

## 3. Monte Carlo Geometric Acceptance (`Monte Carlo Exponent Fit.py`)

**Geometric Acceptance Fit – `Exponent Fit.png`**

A plot of the simulated probability of a beta particle reaching the detector against source-detector separation. The Monte Carlo results are compared with a fitted power-law relationship, illustrating the effect of finite source and detector dimensions on geometric acceptance.

**Fitted Exponent Distribution – `Exponent Histogram.png`**

A histogram showing the distribution of fitted power-law exponents from 100 independent Monte Carlo simulations. The mean fitted exponent is compared with the inverse-square reference value of \(n=2\).

## 4. Monte Carlo Air Attenuation (`Monte Carlo Air Attenuation.py`)

**Air Attenuation Fit – `Air Attenuation Fit.png`**

A plot showing the simulated fraction of beta particles not stopped in air as a function of air path length. An exponential attenuation model is fitted to the simulated transmission data to estimate an effective attenuation coefficient.

**Air Attenuation Coefficient Distribution – `Air Attenuation Histogram.png`**

A histogram showing the distribution of effective air-attenuation coefficients obtained from repeated Monte Carlo simulations. The mean fitted coefficient is indicated by a vertical dashed line.

---

Together, these figures illustrate the progression from experimental data processing and theoretical model fitting to numerical simulation. They provide visual evidence for the investigation's analysis of geometric effects, air attenuation and the limitations of different models in describing the measured count rates.
