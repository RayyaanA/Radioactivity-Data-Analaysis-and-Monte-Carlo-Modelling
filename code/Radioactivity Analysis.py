#Distance dependence of Sr- 90 count rate

#Fits five candidate models to the complete experimental dataset and
# evaluates their goodness of fit using chi- squared statistics. It also
# examines the distance- normalised rate and estimates the contributions
# from finite source- detector geometry and air attenuation.

#The analysis includes Poisson counting uncertainties, background
# uncertainty, ruler- distance uncertainty, and shared distance- offset
# uncertainty. Finite- geometry integrals are precomputed and reused.

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.stats import chi2
from scipy.special import roots_hermitenorm
from numpy.polynomial.legendre import leggauss

#-------- Settings --------
DETECTOR_WINDOW_TO_BASE_M = 0.00200
SOURCE_CASING_TO_BASE_M = 0.00200
SOURCE_WINDOW_TO_CASING_M = 0.00605

DISTANCE_OFFSET_M = (
    DETECTOR_WINDOW_TO_BASE_M
    + SOURCE_CASING_TO_BASE_M
    + SOURCE_WINDOW_TO_CASING_M
)

OFFSET_COMPONENT_TOLERANCE_M = 0.0005
OFFSET_COMPONENT_SIGMA_M = OFFSET_COMPONENT_TOLERANCE_M / np.sqrt(3)
SHARED_OFFSET_SIGMA_M = np.sqrt(2) * OFFSET_COMPONENT_SIGMA_M

DISTANCE_TOLERANCE_M = 0.0005
DISTANCE_MEASUREMENT_SIGMA_M = DISTANCE_TOLERANCE_M / np.sqrt(3)

SOURCE_DIAMETER_M = 0.00975
DETECTOR_WIDTH_M = 0.010
DETECTOR_HEIGHT_M = 0.010

GEOMETRY_SOURCE_ORDER = 7
GEOMETRY_DETECTOR_ORDER = 7

APPLY_DEAD_TIME = True
DEAD_TIME_S = 2.50e-6

BACKGROUND_DATA = [
    (5, 60.0),
    (7, 60.0),
    (7, 60.0),
    (8, 65.535),
]

U_CONSTANT_MIN_M = 0.20
U_CONSTANT_MAX_M = 0.325
U_EXPONENTIAL_MIN_M = 0.40

MU_FIXED = 0.67093

DISTANCE_QUADRATURE_POINTS = 7
P_ACCEPTANCE_THRESHOLD = 0.05

#-------- Experimental Data --------

#Experimental counts grouped by ruler separation; each value stores
# acquisition time per repeat and the list of repeat counts.
DATA = {
    0.010: (5,  [91382, 91115, 91714]),
    0.025: (5,  [42738, 42267, 42483, 42281, 42753, 42536]),
    0.050: (10, [34301, 34099, 33789, 34021, 34072, 33969]),
    0.075: (10, [17685, 17992, 17861, 17714, 17825, 17611]),
    0.100: (10, [10828, 10872, 10898]),
    0.125: (10, [7322, 7387, 7325]),
    0.150: (10, [5196, 5044, 5075, 5235, 5261, 5078]),
    0.175: (10, [3910, 3817, 3846, 4132, 4082, 4096]),
    0.200: (10, [2926, 3000, 2920, 3065, 3060, 2999]),
    0.225: (10, [2444, 2390, 2383, 2468, 2454, 2363]),
    0.250: (10, [1916, 1916, 1863, 1950, 2023, 2036]),
    0.275: (14, [2288, 2232, 2219, 2371, 2334, 2396]),
    0.300: (16, [2091, 2130, 2292, 2279, 2250, 2175]),
    0.325: (18, [2223, 2120, 2250]),
    0.350: (20, [2126, 2122, 2173]),
    0.375: (25, [2426, 2413, 2375, 2359, 2358, 2393]),
    0.400: (30, [2488, 2455, 2491]),
    0.425: (35, [2432, 2430, 2488]),
    0.450: (40, [2586, 2519, 2515]),
    0.475: (45, [2557, 2755, 2731, 2675, 2613, 2597]),
    0.500: (50, [2578, 2492, 2477]),
    0.525: (50, [2256, 2301, 2322]),
    0.550: (55, [1982, 2170, 2232]),
    0.575: (60, [2197, 2184, 2230]),
    0.600: (60, [1888, 1925, 2033]),
    0.625: (60, [1910, 1799, 1919]),
    0.650: (60, [1558, 1573, 1551]),
    0.675: (65, [1710, 1671, 1626]),
    0.700: (65, [1509, 1504, 1476]),
    0.750: (40, [791, 810, 743, 849, 793, 738]),
    0.800: (45, [702, 776, 781, 704, 747, 712]),
    0.850: (55, [759, 741, 783, 762, 731, 751]),
}

#-------- Pool Measurements And Calculate Background --------
distances = np.array(sorted(DATA), dtype=float)

counts = np.array([
    sum(DATA[d][1]) for d in distances
], dtype=float)

times = np.array([
    DATA[d][0] * len(DATA[d][1]) for d in distances
], dtype=float)

rates_observed = counts / times
rate_uncertainties = np.sqrt(counts) / times

background_counts = sum(row[0] for row in BACKGROUND_DATA)
background_time = sum(row[1] for row in BACKGROUND_DATA)

background_rate_observed = background_counts / background_time
background_rate_unc = np.sqrt(background_counts) / background_time

#Map the underlying true rate to the observed rate for a non- paralysable
# detector, accounting for the fraction of time the detector is unavailable.
def deadtime_forward(rate):
    rate = np.asarray(rate, dtype=float)
    if APPLY_DEAD_TIME:
        return rate / (1 + DEAD_TIME_S * rate)
    return rate

#Undo the non- paralysable response to estimate the true rate from
# the measured rate (valid while tau*R_observed is below one).
def deadtime_inverse(rate):
    rate = np.asarray(rate, dtype=float)
    if APPLY_DEAD_TIME:
        return rate / (1 - DEAD_TIME_S * rate)
    return rate

background_rate_true = float(deadtime_inverse(background_rate_observed))

if APPLY_DEAD_TIME:
    background_rate_true_sigma = (
        background_rate_unc
        / (1 - DEAD_TIME_S * background_rate_observed)**2
    )
else:
    background_rate_true_sigma = background_rate_unc

#Gauss- Hermite quadrature for independent distance uncertainty.
quad_x, quad_w = roots_hermitenorm(DISTANCE_QUADRATURE_POINTS)
quad_w /= np.sqrt(2 * np.pi)

distance_deviations = DISTANCE_MEASUREMENT_SIGMA_M * quad_x

#-------- Finite- Geometry Integration --------
def make_geometry_points():
    #Create weighted source- disk and detector- rectangle points.
    gx, gw = leggauss(GEOMETRY_SOURCE_ORDER)

    radial_u = 0.5 * (gx + 1)
    radial_w = 0.5 * gw

    source_radius = SOURCE_DIAMETER_M / 2
    radial_positions = source_radius * np.sqrt(radial_u)

    theta_count = 2 * GEOMETRY_SOURCE_ORDER + 1
    theta = 2 * np.pi * (
        np.arange(theta_count) + 0.5
    ) / theta_count

    source_points = []
    source_weights = []

    for radius, weight in zip(radial_positions, radial_w):
        for angle in theta:
            source_points.append((
                radius * np.cos(angle),
                radius * np.sin(angle)
            ))
            source_weights.append(
                weight * source_radius**2 / 2
                * 2 * np.pi / theta_count
            )

    gx, gw = leggauss(GEOMETRY_DETECTOR_ORDER)

    detector_x = gx * DETECTOR_WIDTH_M / 2
    detector_y = gx * DETECTOR_HEIGHT_M / 2

    detector_wx = gw * DETECTOR_WIDTH_M / 2
    detector_wy = gw * DETECTOR_HEIGHT_M / 2

    detector_points = []
    detector_weights = []

    for i, xx in enumerate(detector_x):
        for j, yy in enumerate(detector_y):
            detector_points.append((xx, yy))
            detector_weights.append(detector_wx[i] * detector_wy[j])

    source_points = np.asarray(source_points)
    detector_points = np.asarray(detector_points)

    #All source- detector lateral separations.
    dx = source_points[:, None, 0] - detector_points[None, :, 0]
    dy = source_points[:, None, 1] - detector_points[None, :, 1]

    lateral_squared = (dx**2 + dy**2).ravel()

    pair_weights = (
        np.asarray(source_weights)[:, None]
        * np.asarray(detector_weights)[None, :]
    ).ravel()

    return lateral_squared, pair_weights

LATERAL_SQUARED, GEOMETRY_WEIGHTS = make_geometry_points()

#Sum the geometric solid- angle contribution over all source/ detector
# point pairs, including attenuation along each path.
def geometry_kernel(distance, mu):
    #G(h,mu) = integral integral [h/r^3 exp(-mu*r)] dAs dAd
    #The source and detector quadrature points are constructed once.
    
    h = np.asarray(distance, dtype=float).ravel()

    r_squared = (
        h[:, None]**2 + LATERAL_SQUARED[None, :]
    )

    r = np.sqrt(r_squared)

    values = (
        h[:, None] / r**3
        * np.exp(-mu * r)
        * GEOMETRY_WEIGHTS[None, :]
    )

    return np.sum(values, axis=1).reshape(np.shape(distance))

#-------- Precompute All Geometry Values Needed For The Fits --------
effective_distances = distances + DISTANCE_OFFSET_M

fit_distances = (
    effective_distances[:, None]
    + distance_deviations[None, :]
)

#Geometry is independent of fitted parameters because mu is fixed.
GEOMETRY_FIT_NODES = geometry_kernel(fit_distances, MU_FIXED)

OFFSET_STEP_M = 1e-5

GEOMETRY_CENTRE = geometry_kernel(effective_distances, MU_FIXED)

GEOMETRY_PLUS = geometry_kernel(
    effective_distances + OFFSET_STEP_M, MU_FIXED
)

GEOMETRY_MINUS = geometry_kernel(
    effective_distances - OFFSET_STEP_M, MU_FIXED
)

#-------- Five Model Definitions --------
MODEL_NAMES = [
    "Inverse square",
    "Power law",
    "Power law + air attenuation",
    "Finite geometry + power correction + attenuation",
    "Finite geometry + attenuation",
]

#Return the source- only rate for one of the five candidate distance laws.
def source_rate(model, parameters, distance, geometry=None):

    distance = np.maximum(np.asarray(distance, dtype=float), 1e-12)

    A = np.exp(parameters[0])

    if model == 0:
        return A / distance**2

    if model == 1:
        n = parameters[1]
        return A / distance**n

    if model == 2:
        n = parameters[1]
        return A * distance**(-n) * np.exp(-MU_FIXED * distance)

    if model == 3:
        q = parameters[1]
        return A * geometry * distance**(-q)

    if model == 4:
        return A * geometry

    raise ValueError("Unknown model number.")

#Convert the source prediction into an observed rate by adding background
# and applying the detector dead- time response.
def predict_observed(model, parameters, distance, geometry=None):

    source = source_rate(model, parameters, distance, geometry)

    total_true = source + background_rate_true

    return deadtime_forward(total_true)

def initial_parameters(model):

    first_distance = effective_distances[0]

    A0 = max(
        rates_observed[0] * first_distance**2,
        1e-12
    )

    if model == 0:
        return [np.log(A0)], [np.log(1e-12)], [np.log(1e12)]

    if model in (1, 2):
        return (
            [np.log(A0), 2.0],
            [np.log(1e-12), 0.0],
            [np.log(1e12), 8.0]
        )

    geometry_first = GEOMETRY_CENTRE[0]

    A_geometry = max(
        rates_observed[0] / geometry_first,
        1e-12
    )

    if model == 3:
        return (
            [np.log(A_geometry), 0.0],
            [np.log(1e-12), -3.0],
            [np.log(1e20), 8.0]
        )

    if model == 4:
        return (
            [np.log(A_geometry)],
            [np.log(1e-12)],
            [np.log(1e20)]
        )

#-------- Model Predictions And Full Covariance --------

#Calculate model predictions and propagate the measurement uncertainties.
def prediction_and_covariance(model, parameters):

    if model >= 3:
        geometry_nodes = GEOMETRY_FIT_NODES
    else:
        geometry_nodes = None

    predicted_nodes = predict_observed(
        model,
        parameters,
        fit_distances,
        geometry_nodes
    )

    predicted_mean = np.sum(
        predicted_nodes * quad_w[None, :],
        axis=1
    )

    #Independent distance uncertainty.
    distance_variance = np.sum(
        quad_w[None, :]
        * (predicted_nodes - predicted_mean[:, None])**2,
        axis=1
    )

    covariance = np.diag(
        rate_uncertainties**2 + distance_variance
    )

    #Shared background uncertainty.
    if model >= 3:
        geometry_centre = GEOMETRY_CENTRE
    else:
        geometry_centre = None

    central_source = source_rate(
        model,
        parameters,
        effective_distances,
        geometry_centre
    )

    total_true = central_source + background_rate_true

    if APPLY_DEAD_TIME:
        background_derivative = (
            1 / (1 + DEAD_TIME_S * total_true)**2
        )
    else:
        background_derivative = np.ones_like(total_true)

    background_vector = (
        background_derivative * background_rate_true_sigma
    )

    covariance += np.outer(
        background_vector,
        background_vector
    )

    #Shared uncertainty in the physical distance offset.
    if model >= 3:
        geometry_plus = GEOMETRY_PLUS
        geometry_minus = GEOMETRY_MINUS
    else:
        geometry_plus = None
        geometry_minus = None

    prediction_plus = predict_observed(
        model,
        parameters,
        effective_distances + OFFSET_STEP_M,
        geometry_plus
    )

    prediction_minus = predict_observed(
        model,
        parameters,
        effective_distances - OFFSET_STEP_M,
        geometry_minus
    )

    offset_derivative = (
        prediction_plus - prediction_minus
    ) / (2 * OFFSET_STEP_M)

    offset_vector = offset_derivative * SHARED_OFFSET_SIGMA_M

    covariance += np.outer(offset_vector, offset_vector)

    return predicted_mean, covariance

#-------- Fit All Five Models Using Generalized Least Squares --------
#Fit one candidate model using generalized least squares.
def fit_model(model):

    p0, lower, upper = initial_parameters(model)

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    def whitened_residuals(parameters):

        prediction, covariance = prediction_and_covariance(
            model, parameters
        )

        L = np.linalg.cholesky(covariance)

        return np.linalg.solve(
            L,
            rates_observed - prediction
        )

    result = least_squares(
        whitened_residuals,
        p0,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=3000
    )

    parameters = result.x

    prediction, covariance = prediction_and_covariance(
        model, parameters
    )

    L = np.linalg.cholesky(covariance)

    whitened = np.linalg.solve(
        L,
        rates_observed - prediction
    )

    chi_squared = float(whitened @ whitened)

    dof = len(distances) - len(parameters)

    p_value = chi2.sf(chi_squared, dof)

    return {
        "model": model,
        "name": MODEL_NAMES[model],
        "parameters": parameters,
        "prediction": prediction,
        "covariance": covariance,
        "chi2": chi_squared,
        "dof": dof,
        "reduced_chi2": chi_squared / dof,
        "p": p_value,
        "success": result.success,
        "message": result.message,
    }

print("\nFITTING FIVE MODELS")
print("This uses the full dataset and the full covariance matrix.")

results = []

for model in range(5):
    fit = fit_model(model)
    results.append(fit)

    print("\n" + "-" * 65)
    print(fit["name"])

    print("Fit successful:", fit["success"])

    print("A =", f"{np.exp(fit['parameters'][0]):.7g}")

    if len(fit["parameters"]) > 1:
        print("Shape parameter =", f"{fit['parameters'][1]:.7g}")

    print("Chi-squared =", f"{fit['chi2']:.6g}")
    print("Degrees of freedom =", fit["dof"])
    print("Reduced chi-squared =", f"{fit['reduced_chi2']:.6g}")
    print("p-value =", f"{fit['p']:.6g}")

    if fit["p"] >= P_ACCEPTANCE_THRESHOLD:
        print("Goodness of fit: NOT REJECTED at 5% significance")
    else:
        print("Goodness of fit: REJECTED at 5% significance")

#-------- Summary Table --------
print("\n\nMODEL COMPARISON")
print(
    f"{'Model':<58s} {'chi2':>12s} "
    f"{'dof':>6s} {'red chi2':>12s} {'p-value':>12s}"
)

for fit in results:
    print(
        f"{fit['name']:<58s}"
        f"{fit['chi2']:12.4g}"
        f"{fit['dof']:6d}"
        f"{fit['reduced_chi2']:12.4g}"
        f"{fit['p']:12.4g}"
    )

#-------- Five- Model Comparison (Log Scale) --------
#Combined marginal horizontal uncertainty:
# independent distance uncertainty + shared physical offset uncertainty.
distance_x_uncertainty = np.sqrt(
    DISTANCE_MEASUREMENT_SIGMA_M**2
    + SHARED_OFFSET_SIGMA_M**2
)

fig_models_log, ax_models_log = plt.subplots(figsize=(10, 6))

#Measured data points.
ax_models_log.scatter(
    effective_distances,
    rates_observed,
    marker="o",
    color="black",
    s=10,
    label="Measured data",
    zorder=10
)

#Measurement uncertainties, shown separately in the legend.
ax_models_log.errorbar(
    effective_distances,
    rates_observed,
    xerr=distance_x_uncertainty,
    yerr=rate_uncertainties,
    fmt="none",
    ecolor="black",
    capsize=2.5,
    linewidth=0.9,
    label="Measurement uncertainties",
    zorder=9
)

#Smooth grid of physical source- detector separations.
separation_grid = np.linspace(
    effective_distances.min(),
    effective_distances.max(),
    400
)

geometry_grid = geometry_kernel(
    separation_grid,
    MU_FIXED
)

#Plot all five fitted models, including their p- values.
for fit in results:

    model = fit["model"]
    parameters = fit["parameters"]

    geometry_for_plot = (
        geometry_grid if model >= 3 else None
    )

    fitted_rates = predict_observed(
        model,
        parameters,
        separation_grid,
        geometry_for_plot
    )

    ax_models_log.plot(
        separation_grid,
        fitted_rates,
        linewidth=2,
        label=(
            f"{fit['name']} model "
            f"(p = {fit['p']:.2e})"
        )
    )

ax_models_log.set_xlabel(
    "Source- detector separation, $d$ (m)", fontsize = 16
)

ax_models_log.set_ylabel(
    "Observed count rate (counts s$^{-1}$)", fontsize = 16
)

ax_models_log.set_title(
    "Log plot comparison of fitted models with measured Sr- 90 count rates", fontsize = 16
)

ax_models_log.set_yscale("log")

ax_models_log.grid(
    True,
    which="both",
    alpha=0.3
)

#Increase axis number font size
ax_models_log.tick_params(
    axis="both",
    which="major",
    labelsize=14
)

ax_models_log.tick_params(
    axis="both",
    which="minor",
    labelsize=12
)

handles, labels = ax_models_log.get_legend_handles_labels()

priority = {"Measured data": 0, "Measurement uncertainties": 1}
order = sorted(
    range(len(labels)),
    key=lambda i: priority.get(labels[i], 2)
)

ax_models_log.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    fontsize=12,
    loc="best"
)

fig_models_log.tight_layout()
plt.savefig("Log plot of models.png")

#-------- Calculate Rd^2 and its Uncertainty --------

#Correct the measured total rate before subtracting the estimated true background.
rates_true_total = deadtime_inverse(rates_observed)

rates_net = rates_true_total - background_rate_true

if APPLY_DEAD_TIME:
    rate_correction_derivative = (
        1 / (1 - DEAD_TIME_S * rates_observed)**2
    )
else:
    rate_correction_derivative = np.ones_like(rates_observed)

rate_uncertainty_true = (
    rate_uncertainties * rate_correction_derivative
)

u_values = rates_net * effective_distances**2

#Independent Poisson counting uncertainty.
u_counting_variance = (
    effective_distances**4 * rate_uncertainty_true**2
)

#Independent distance uncertainty.
u_distance_variance = (
    2 * rates_net * effective_distances
    * DISTANCE_MEASUREMENT_SIGMA_M
)**2

u_covariance = np.diag(
    u_counting_variance + u_distance_variance
)

#Shared background uncertainty.
u_background_vector = (
    effective_distances**2 * background_rate_true_sigma
)

u_covariance += np.outer(
    u_background_vector,
    u_background_vector
)

#Shared physical distance- offset uncertainty.
u_offset_vector = (
    2 * rates_net * effective_distances
    * SHARED_OFFSET_SIGMA_M
)

u_covariance += np.outer(
    u_offset_vector,
    u_offset_vector
)

u_uncertainties = np.sqrt(np.diag(u_covariance))

#-------- Constant Fit to Rd^2 --------

#Test constancy of R*d^2 only over the preselected distance interval.
constant_mask = (
    (distances >= U_CONSTANT_MIN_M)
    & (distances <= U_CONSTANT_MAX_M)
)

constant_indices = np.where(constant_mask)[0]

u_constant = np.nan
u_constant_uncertainty = np.nan

if len(constant_indices) >= 2:

    u_selected = u_values[constant_mask]

    C_constant = u_covariance[
        np.ix_(constant_indices, constant_indices)
    ]

    ones = np.ones(len(u_selected))

    C_inv_ones = np.linalg.solve(C_constant, ones)
    C_inv_u = np.linalg.solve(C_constant, u_selected)

    u_constant = (
        ones @ C_inv_u
    ) / (
        ones @ C_inv_ones
    )

    u_constant_uncertainty = np.sqrt(
        1 / (ones @ C_inv_ones)
    )

    constant_residual = u_selected - u_constant

    chi_constant = (
        constant_residual
        @ np.linalg.solve(C_constant, constant_residual)
    )

    dof_constant = len(u_selected) - 1
    p_constant = chi2.sf(chi_constant, dof_constant)

    print("\nCONSTANT Rd^2 FIT")
    print("Distance interval:", U_CONSTANT_MIN_M, "to", U_CONSTANT_MAX_M, "m")
    print("u =", u_constant, "+/-", u_constant_uncertainty)
    print("Chi-squared =", chi_constant)
    print("Degrees of freedom =", dof_constant)
    print("p-value =", p_constant)

#-------- Exponential Fit to the Rd^2 Tail --------

#Fit the long- distance R*d^2 tail with an effective exponential model.
exponential_mask = (
    (distances >= U_EXPONENTIAL_MIN_M)
    & (u_values > 0)
)

exponential_indices = np.where(exponential_mask)[0]

exponential_fit = None

if len(exponential_indices) >= 3:

    d_exp = distances[exponential_mask]
    u_exp = u_values[exponential_mask]

    C_exp = u_covariance[
        np.ix_(exponential_indices, exponential_indices)
    ]

    L_exp = np.linalg.cholesky(C_exp)

    def exponential_function(distance, amplitude, mu):
        return amplitude * np.exp(
            -mu * (distance - U_EXPONENTIAL_MIN_M)
        )

    def exponential_residuals(parameters):
        predicted = exponential_function(d_exp, *parameters)

        return np.linalg.solve(
            L_exp,
            u_exp - predicted
        )

    exponential_fit = least_squares(
        exponential_residuals,
        [max(np.max(u_exp), 1e-12), 0.5],
        bounds=([0, 0], [np.inf, 100]),
        max_nfev=3000
    )

    amplitude_fit, mu_fit = exponential_fit.x

    exponential_whitened = exponential_residuals(
        exponential_fit.x
    )

    chi_exponential = float(
        exponential_whitened @ exponential_whitened
    )

    dof_exponential = len(d_exp) - 2
    p_exponential = chi2.sf(
        chi_exponential,
        dof_exponential
    )

    #Approximate parameter uncertainties from the fit Jacobian.
    parameter_covariance = np.linalg.inv(
        exponential_fit.jac.T @ exponential_fit.jac
    )

    amplitude_uncertainty, mu_uncertainty = np.sqrt(
        np.maximum(np.diag(parameter_covariance), 0)
    )

    print("\nEXPONENTIAL Rd^2 TAIL FIT")
    print("Fit starts at:", U_EXPONENTIAL_MIN_M, "m")
    print("Amplitude =", amplitude_fit, "+/-", amplitude_uncertainty)
    print("mu =", mu_fit, "+/-", mu_uncertainty, "m^-1")
    print("Chi-squared =", chi_exponential)
    print("Degrees of freedom =", dof_exponential)
    print("Reduced chi-squared =", chi_exponential / dof_exponential)
    print("p-value =", p_exponential)

#-------- Rd^2 with Constant and Exponential Fits --------

#Combined marginal uncertainty in physical separation.
#The offset uncertainty is shared between all measurements.
distance_x_uncertainty = np.sqrt(
    DISTANCE_MEASUREMENT_SIGMA_M**2
    + SHARED_OFFSET_SIGMA_M**2
)

fig_rd2, ax_rd2 = plt.subplots(figsize=(10, 6))

#Measured data points.
ax_rd2.scatter(
    effective_distances,
    u_values,
    marker="o",
    color="black",
    s=10,
    label=r"Measured data",
    zorder=10
)

#Horizontal and vertical measurement uncertainties.
ax_rd2.errorbar(
    effective_distances,
    u_values,
    xerr=distance_x_uncertainty,
    yerr=u_uncertainties,
    fmt="none",
    ecolor="black",
    capsize=2.5,
    linewidth=0.9,
    label="Measurement uncertainties",
    zorder=9
)

#Constant- u fit.
if np.isfinite(u_constant):

    ax_rd2.axhline(
        u_constant,
        linestyle="--",
        linewidth=2,
        label=(
            rf"Constant $u$ fit: "
            rf"$u$ = {u_constant:.3f} "
            rf"$\pm$ {u_constant_uncertainty:.3f}"
        )
    )

    #Physical separation limits of the constant- fit interval.
    constant_start = U_CONSTANT_MIN_M
    constant_end = U_CONSTANT_MAX_M

    ax_rd2.axvspan(
        constant_start,
        constant_end,
        alpha=0.10,
        color="skyblue",
        label=(
            f"Constant $u$ fit interval: "
            f"{constant_start:.3f}–{constant_end:.3f} m"
        )
    )

#Exponential- fit cutoff line.
#This is outside the constant- fit condition so it is always displayed.
exponential_cutoff = (
    U_EXPONENTIAL_MIN_M + DISTANCE_OFFSET_M
)

ax_rd2.axvline(
    exponential_cutoff,
    color="black",
    linestyle=":",
    linewidth=1.3,
    label=(
        f"Exponential fit cutoff: "
        f"{U_EXPONENTIAL_MIN_M:.2f} m"
    )
)

#Exponential air- attenuation fit.
if exponential_fit is not None:

    exponential_grid = np.linspace(
        d_exp.min(),
        d_exp.max(),
        250
    )

    ax_rd2.plot(
        exponential_grid + DISTANCE_OFFSET_M,
        exponential_function(
            exponential_grid,
            amplitude_fit,
            mu_fit
        ),
        linewidth=2,
        label=(
            rf"Exponential air-attenuation fit: "
            rf"$\mu$ = {mu_fit:.3f} "
            rf"$\pm$ {mu_uncertainty:.3f} m$^{{-1}}$"
        )
    )

ax_rd2.set_xlabel(
    "Source- detector separation, $d$ (m)", fontsize = 16
)

ax_rd2.set_ylabel(
    r"$Rd^2$ (counts m$^2$ s$^{-1}$)", fontsize = 16
)

ax_rd2.set_title(
    r"$Rd^2$ against source- detector separation", fontsize = 16
)

ax_rd2.grid(
    True,
    alpha=0.3
)

#Increase axis number font size
ax_rd2.tick_params(
    axis="both",
    which="major",
    labelsize=14
)

handles, labels = ax_rd2.get_legend_handles_labels()

priority = {
    "Measured data": 0,
    "Measurement uncertainties": 1
}

order = sorted(
    range(len(labels)),
    key=lambda i: priority.get(labels[i], 2)
)

ax_rd2.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    fontsize=12,
    loc="best"
)

fig_rd2.tight_layout()
plt.savefig("Rd^2 plot.png")

#-------- Final Settings And Completion --------
print("\nUNCERTAINTY AND MODEL SETTINGS")
print("Distance offset:", DISTANCE_OFFSET_M, "m")
print("Shared offset uncertainty:", SHARED_OFFSET_SIGMA_M, "m")
print("Independent distance uncertainty:", DISTANCE_MEASUREMENT_SIGMA_M, "m")
print("Background rate:", background_rate_observed, "+/-", background_rate_unc, "counts/s")
print("Dead time applied:", APPLY_DEAD_TIME)
print("Dead time:", DEAD_TIME_S, "s")
print("Fixed air attenuation coefficient:", MU_FIXED, "m^-1")
print("Source diameter:", SOURCE_DIAMETER_M, "m")
print("Detector width:", DETECTOR_WIDTH_M, "m")
print("Detector height:", DETECTOR_HEIGHT_M, "m")

print("\nAnalysis complete.")

plt.show()

#-------- Inverse- Square Contiguous Distance- Range Scan --------

#This section tests the inverse- square model over every possible
# contiguous interval of measured distances.

# It uses the same:
#-Poisson counting uncertainties
#-Independent distance uncertainty
#-Shared background uncertainty
#-Shared physical distance-offset uncertainty
#-Dead- time correction

#The scan is exploratory: testing many overlapping ranges means
# that a good p- value in one interval is not independent confirmation.

#------------------------- USER SETTINGS -------------------------
MIN_RANGE_POINTS = 6
MAX_RANGE_ROWS_PRINT = 30
HEATMAP_P_FLOOR = 1e-12

#Significance threshold
RANGE_P_THRESHOLD = 0.05

#------------------------- RANGE FITTING -------------------------
def fit_inverse_square_range(indices):

    indices = np.asarray(indices, dtype=int)

    d_eff = effective_distances[indices]
    d_nodes = fit_distances[indices]

    observed = rates_observed[indices]
    observed_sigma = rate_uncertainties[indices]
   
    #Initial estimate of the amplitude.
    A_initial = max(
        (observed[0] - background_rate_observed)
        * d_eff[0]**2,
        1e-12
    )

    p0 = np.array([np.log(A_initial)])

    lower = np.array([np.log(1e-12)])
    upper = np.array([np.log(1e12)])

    def prediction_and_covariance_range(parameters):

        #Inverse- square source model.
        predicted_nodes = predict_observed(
            0,
            parameters,
            d_nodes,
            None
        )

        predicted_mean = np.sum(
            predicted_nodes * quad_w[None, :],
            axis=1
        )

        #Independent distance uncertainty.
        distance_variance = np.sum(
            quad_w[None, :]
            * (predicted_nodes - predicted_mean[:, None])**2,
            axis=1
        )

        covariance = np.diag(
            observed_sigma**2 + distance_variance
        )

        #Shared background uncertainty.
        central_source = source_rate(
            0,
            parameters,
            d_eff,
            None
        )

        total_true = central_source + background_rate_true

        if APPLY_DEAD_TIME:
            background_derivative = (
                1 / (1 + DEAD_TIME_S * total_true)**2
            )
        else:
            background_derivative = np.ones_like(total_true)

        background_vector = (
            background_derivative * background_rate_true_sigma
        )

        covariance += np.outer(
            background_vector,
            background_vector
        )

        #Shared physical distance- offset uncertainty.
        prediction_plus = predict_observed(
            0,
            parameters,
            d_eff + OFFSET_STEP_M,
            None
        )

        prediction_minus = predict_observed(
            0,
            parameters,
            d_eff - OFFSET_STEP_M,
            None
        )

        offset_derivative = (
            prediction_plus - prediction_minus
        ) / (2 * OFFSET_STEP_M)

        offset_vector = (
            offset_derivative * SHARED_OFFSET_SIGMA_M
        )

        covariance += np.outer(
            offset_vector,
            offset_vector
        )

        return predicted_mean, covariance

    def whitened_residuals(parameters):

        prediction, covariance = prediction_and_covariance_range(
            parameters
        )

        L = np.linalg.cholesky(covariance)

        return np.linalg.solve(
            L,
            observed - prediction
        )

    fit = least_squares(
        whitened_residuals,
        p0,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=3000
    )

    prediction, covariance = prediction_and_covariance_range(
        fit.x
    )

    L = np.linalg.cholesky(covariance)

    residuals = np.linalg.solve(
        L,
        observed - prediction
    )

    chi_squared = float(residuals @ residuals)

    #One fitted parameter: the inverse- square amplitude.
    dof = len(indices) - 1

    p_value = chi2.sf(chi_squared, dof)

    return {
        "start": float(distances[indices[0]]),
        "end": float(distances[indices[-1]]),
        "n_points": len(indices),
        "A": float(np.exp(fit.x[0])),
        "chi2": chi_squared,
        "dof": dof,
        "reduced_chi2": chi_squared / dof,
        "p": p_value,
        "accepted": p_value >= RANGE_P_THRESHOLD,
        "success": fit.success
    }

#------------------------- TEST ALL RANGES -------------------------
n_distances = len(distances)

range_rows = []

p_value_matrix = np.full(
    (n_distances, n_distances),
    np.nan
)

for start_index in range(n_distances):

    for end_index in range(
        start_index + MIN_RANGE_POINTS - 1,
        n_distances
    ):

        indices = np.arange(
            start_index,
            end_index + 1
        )

        result = fit_inverse_square_range(indices)

        range_rows.append(result)

        p_value_matrix[
            start_index,
            end_index
        ] = result["p"]

#Sort by p- value, highest first.
range_rows.sort(
    key=lambda row: row["p"],
    reverse=True
)

#------------------------- PRINT RESULTS -------------------------
accepted_ranges = [
    row for row in range_rows
    if row["accepted"]
]

print("\n" + "=" * 85)
print("INVERSE-SQUARE CONTIGUOUS DISTANCE-RANGE SCAN")
print("=" * 85)

print(
    f"Minimum points per interval: {MIN_RANGE_POINTS}"
)

print(
    f"Total contiguous intervals tested: {len(range_rows)}"
)

print(
    f"Intervals not rejected at 5% significance: "
    f"{len(accepted_ranges)}"
)

print(
    f"Intervals rejected at 5% significance: "
    f"{len(range_rows) - len(accepted_ranges)}"
)

print("\nBEST-FITTING DISTANCE INTERVALS")
print(
    f"{'Start (m)':>11s} "
    f"{'End (m)':>11s} "
    f"{'Points':>7s} "
    f"{'Chi2':>11s} "
    f"{'dof':>5s} "
    f"{'Red. chi2':>11s} "
    f"{'p-value':>12s} "
    f"{'Status':>12s}"
)

for row in range_rows[:MAX_RANGE_ROWS_PRINT]:

    status = (
        "ACCEPTABLE"
        if row["accepted"]
        else "REJECTED"
    )

    print(
        f"{row['start']:11.3f} "
        f"{row['end']:11.3f} "
        f"{row['n_points']:7d} "
        f"{row['chi2']:11.4g} "
        f"{row['dof']:5d} "
        f"{row['reduced_chi2']:11.4g} "
        f"{row['p']:12.4g} "
        f"{status:>12s}"
    )

#Print all intervals that pass the 5% test.
if accepted_ranges:

    print("\nALL INTERVALS NOT REJECTED AT 5% SIGNIFICANCE")

    for row in accepted_ranges:

        print(
            f"{row['start']:.3f}–{row['end']:.3f} m | "
            f"{row['n_points']} points | "
            f"reduced chi2 = {row['reduced_chi2']:.4g} | "
            f"p = {row['p']:.4g}"
        )

else:

    print(
        "\nNo contiguous interval with at least "
        f"{MIN_RANGE_POINTS} points was acceptable "
        "at the 5% significance level."
    )

#------------------------- P- VALUE HEATMAP -------------------------
plot_p_values = np.ma.masked_invalid(
    np.maximum(p_value_matrix, HEATMAP_P_FLOOR)
)

fig_range_scan, ax_range_scan = plt.subplots(
    figsize=(10, 8)
)

image = ax_range_scan.imshow(
    np.log10(plot_p_values),
    origin="lower",
    aspect="auto",
    interpolation="nearest"
)

colourbar = fig_range_scan.colorbar(
    image,
    ax=ax_range_scan
)

colourbar.set_label(r"$\log_{10}(p)$")

tick_indices = np.arange(
    0,
    n_distances,
    max(1, n_distances // 10)
)

ax_range_scan.set_xticks(tick_indices)
ax_range_scan.set_yticks(tick_indices)

ax_range_scan.set_xticklabels(
    [f"{distances[i]:.3f}" for i in tick_indices],
    rotation=45
)

ax_range_scan.set_yticklabels(
    [f"{distances[i]:.3f}" for i in tick_indices]
)

ax_range_scan.set_xlabel("End distance of interval (m)")
ax_range_scan.set_ylabel("Start distance of interval (m)")

ax_range_scan.set_title(
    "Inverse-square contiguous-range scan"
)

fig_range_scan.tight_layout()

#------------------------- FINAL INTERPRETATION -------------------------
print("\nRANGE-SCAN INTERPRETATION")

print(
    "1. Each cell in the heatmap represents one contiguous "
    "distance interval."
)

print(
    "2. Larger p-values indicate that the inverse-square model "
    "is less strongly contradicted over that interval."
)

print(
    "3. A p-value above 0.05 means the model is not rejected "
    "at the 5% significance level; it does not prove the model "
    "is correct."
)

print(
    "4. The intervals overlap and are therefore not independent "
    "tests. A favourable interval may occur by chance when many "
    "intervals are examined."
)

print(
    "5. These results are exploratory and should not replace "
    "the full-dataset five-model comparison."
)

print("\nRange scan complete.")

plt.savefig("Range scan.png")
plt.show()
