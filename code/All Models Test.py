#Systematic comparison of distance laws and correction models

#Fits a grid of candidate models to the 32- distance experimental dataset.
#The comparison covers inverse- square, power- law, and finite- geometry
# descriptions, with optional attenuation, an additive signal component,
# detector dead time, and distance- offset scenarios.

#Counting, background, ruler- distance, and shared- offset uncertainties
# are included in the covariance- weighted fitting procedure. The model
# grid contains 64 configurations, and the script displays a comparison
# of their reduced chi- squared values.

import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.linalg import solve_triangular
from scipy.special import roots_hermitenorm
from numpy.polynomial.legendre import leggauss
from scipy.stats import chi2

#-------- Experimental Data --------

#Format: ruler distance [m]: (acquisition time per repeat [s], counts per repeat)
#The repeated measurements are embedded here so the comparison runs without
# an external data file; each entry stores time per repeat and repeat counts.
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

BACKGROUND_DATA = [(5, 60.0), (7, 60.0), (7, 60.0), (8, 65.535)]

#-------- Fixed Experimental/ Model Settings --------

#Distance offset of 0 or 10.5 mm. This is 0.0105 m.
DISTANCE_OFFSETS_M = [0.0, 0.0105]

#Ruler- distance tolerance is treated as a rectangular +/- tolerance.
DISTANCE_TOLERANCE_M = 0.0005
DISTANCE_SIGMA_M = DISTANCE_TOLERANCE_M / np.sqrt(3)

#The common offset is also treated as a rectangular +/- tolerance.
OFFSET_COMPONENT_TOLERANCE_M = 0.0005
OFFSET_COMPONENT_SIGMA_M = OFFSET_COMPONENT_TOLERANCE_M / np.sqrt(3)
SHARED_OFFSET_SIGMA_M = np.sqrt(2) * OFFSET_COMPONENT_SIGMA_M

SOURCE_DIAMETER_M = 0.00975
DETECTOR_WIDTH_M = 0.010
DETECTOR_HEIGHT_M = 0.010

#Fixed non- paralysable dead time when the correction is enabled.
DEAD_TIME_S = 2.5e-6

GEOMETRY_SOURCE_ORDER = 7
GEOMETRY_DETECTOR_ORDER = 7
DISTANCE_QUADRATURE_ORDER = 7

#Four model families.
FAMILIES = ["inverse", "power", "geometry", "geometry_power"]

#Keep the extra constant as an optional diagnostic model.
INCLUDE_EXTRA_CONSTANT_MODELS = True

#Parameter bounds.
MU_BOUNDS = (0.0, 20.0)
POWER_BOUNDS = (0.1, 4.0)

#-------- Pool Repeats And Calculate Observed Rates --------

distances = np.array(sorted(DATA), dtype=float)
counts = np.array([sum(DATA[d][1]) for d in distances], dtype=float)
times = np.array([DATA[d][0] * len(DATA[d][1]) for d in distances], dtype=float)

observed_rates = counts / times
counting_variance = counts / times**2
counting_sigma = np.sqrt(counting_variance)

bg_counts = float(sum(item[0] for item in BACKGROUND_DATA))
bg_time = float(sum(item[1] for item in BACKGROUND_DATA))
bg_rate_observed = bg_counts / bg_time
bg_variance_observed = bg_counts / bg_time**2
bg_sigma_observed = np.sqrt(bg_variance_observed)

#Gauss- Hermite nodes/ weights for a standard- normal distance uncertainty.
quad_x, quad_w = roots_hermitenorm(DISTANCE_QUADRATURE_ORDER)
quad_w = quad_w / np.sqrt(2.0 * np.pi)
distance_shifts = DISTANCE_SIGMA_M * quad_x

#-------- Finite Source- Detector Geometry --------

#Build quadrature points once because source/ detector geometry is shared
# by every finite- geometry configuration.
def make_geometry_points():
    #Construct weighted points over a uniform circular source and square detector.
    #The geometric kernel integrates cos(theta)/r^2 over source and detector
    # areas. With attenuation enabled, exp(-mu*r) is evaluated for each
    # source- detector path, rather than applying attenuation only at axial
    # distance after the geometry integration.
    
    gx, gw = leggauss(GEOMETRY_SOURCE_ORDER)

    #Uniform- area radial integration for a disk: r = R*sqrt(u).
    radial_u = 0.5 * (gx + 1.0)
    radial_w = 0.5 * gw
    source_radius = SOURCE_DIAMETER_M / 2.0
    radial_positions = source_radius * np.sqrt(radial_u)

    theta_count = 2 * GEOMETRY_SOURCE_ORDER + 1
    theta = 2.0 * np.pi * (np.arange(theta_count) + 0.5) / theta_count

    source_points = []
    source_weights = []
    for radius, weight in zip(radial_positions, radial_w):
        for angle in theta:
            source_points.append((radius * np.cos(angle),
                                  radius * np.sin(angle)))
            source_weights.append(
                weight * source_radius**2 / 2.0 * 2.0 * np.pi / theta_count
            )

    gx, gw = leggauss(GEOMETRY_DETECTOR_ORDER)
    detector_x = gx * DETECTOR_WIDTH_M / 2.0
    detector_y = gx * DETECTOR_HEIGHT_M / 2.0
    detector_wx = gw * DETECTOR_WIDTH_M / 2.0
    detector_wy = gw * DETECTOR_HEIGHT_M / 2.0

    detector_points = []
    detector_weights = []
    for i, x in enumerate(detector_x):
        for j, y in enumerate(detector_y):
            detector_points.append((x, y))
            detector_weights.append(detector_wx[i] * detector_wy[j])

    source_points = np.asarray(source_points, dtype=float)
    detector_points = np.asarray(detector_points, dtype=float)

    sx = source_points[:, None, 0] - detector_points[None, :, 0]
    sy = source_points[:, None, 1] - detector_points[None, :, 1]

    lateral2 = (sx**2 + sy**2).ravel()
    weights = (
        np.asarray(source_weights)[:, None]
        * np.asarray(detector_weights)[None, :]
    ).ravel()

    return lateral2, weights

LATERAL2, GEOMETRY_WEIGHTS = make_geometry_points()

@lru_cache(maxsize=128)
#Cache path lengths and geometric weights; only exp(-mu*r) changes as
# the attenuation parameter is varied during a fit.
def _geometry_distance_terms(distance_key):
    #Cache geometry- only terms for a particular distance array.
    #Distances are fixed throughout a given model fit (only mu changes), so
    # repeatedly calculating sqrt(h**2 + lateral**2) is unnecessary. The cache
    # stores the path lengths and unattenuated integration weights once.
    
    h = np.asarray(distance_key, dtype=float)
    if np.any(h <= 0):
        raise ValueError("Effective source-detector distance must be positive.")

    r = np.sqrt(h[:, None]**2 + LATERAL2[None, :])
    base = (h[:, None] / r**3) * GEOMETRY_WEIGHTS[None, :]
    return r, base

#Integrate geometric acceptance over both finite surfaces, optionally
# applying attenuation along each individual source- to- detector path.
def geometry_kernel(distance, mu=0.0):
    #Finite- area geometric kernel, optionally including path- wise attenuation.
    original_shape = np.shape(distance)
    h = np.asarray(distance, dtype=float).ravel()
    key = tuple(h.tolist())
    r, base = _geometry_distance_terms(key)

    #Only the attenuation factor changes during optimisation.
    if mu == 0.0:
        values = np.sum(base, axis=1)
    else:
        values = np.einsum("ij,ij->i", base, np.exp(-mu * r),
                           optimize=False)
    return values.reshape(original_shape)

#-------- Dead- Time Model --------

#Forward detector response: a non- paralysable dead time reduces the
# recorded rate relative to the underlying true event rate.
def apply_dead_time(true_rate, dead_model, tau):
    #Convert true event rate to observed rate.
    rate = np.asarray(true_rate, dtype=float)

    if dead_model == "none" or tau == 0.0:
        return rate
    if dead_model == "nonparalysable":
        return rate / (1.0 + rate * tau)

    raise ValueError(f"Unknown dead-time model: {dead_model}")

#Invert the non- paralysable response when converting measured background
# rates to estimated true rates.
def invert_dead_time(observed_rate, dead_model, tau):
    #Approximate conversion of measured background rate to true rate.
    rate = np.asarray(observed_rate, dtype=float)

    if dead_model == "none" or tau == 0.0:
        return rate
    if dead_model == "nonparalysable":
        denominator = 1.0 - rate * tau
        if np.any(denominator <= 0):
            raise ValueError("Background rate is outside inverse dead-time domain.")
        return rate / denominator

    raise ValueError(f"Unknown dead-time model: {dead_model}")

#-------- Model Definitions And Predictions --------

def parameter_names(family, attenuation, add_constant):
    names = []
    if add_constant:
        names.append("B_extra")
    names.append("log_A")
    if family in ("power", "geometry_power"):
        names.append("p")
    if attenuation:
        names.append("mu_per_m")
    return names

#Evaluate the selected source- distance family before background and
# detector dead- time effects are applied.
def source_rate(theta, family, attenuation, add_constant, distance):
    #Underlying source signal rate before background and dead- time effects.
    names = parameter_names(family, attenuation, add_constant)
    pars = dict(zip(names, theta))

    d = np.asarray(distance, dtype=float)
    A = np.exp(pars["log_A"])

    if family == "inverse":
        shape = (d / 0.1)**-2.0
    elif family == "power":
        shape = (d / 0.1)**(-pars["p"])
    elif family == "geometry":
        mu = pars["mu_per_m"] if attenuation else 0.0
        shape = geometry_kernel(d, mu=mu)
    elif family == "geometry_power":
        mu = pars["mu_per_m"] if attenuation else 0.0
        shape = geometry_kernel(d, mu=mu) * (d / 0.1)**(-(pars["p"] - 2.0))
    else:
        raise ValueError(f"Unknown model family: {family}")

    #For point- source families attenuation is applied along the axial distance.
    if attenuation and family in ("inverse", "power"):
        shape = shape * np.exp(-pars["mu_per_m"] * d)

    signal = A * shape
    if add_constant:
        signal = signal + pars["B_extra"]

    return signal

#Add the background and apply the chosen detector response to the signal.
def predict_observed_rate(theta, family, attenuation, add_constant,
                          distance, offset, dead_model, tau, bg_true):
    effective_distance = np.asarray(distance, dtype=float) + offset
    signal = source_rate(theta, family, attenuation, add_constant,
                         effective_distance)
    if np.any(~np.isfinite(signal)) or np.any(signal < 0.0):
        return np.full_like(effective_distance, np.nan, dtype=float)

    #Apply dead- time response to source plus true background.
    return apply_dead_time(signal + bg_true, dead_model, tau)

#-------- Initial Values And Bounds --------

def initial_and_bounds(family, attenuation, add_constant, offset, dead_model, tau):
    names = parameter_names(family, attenuation, add_constant)

    d0 = distances + offset
    bg_true = float(invert_dead_time(bg_rate_observed, dead_model, tau))
    first_signal = max(observed_rates[0] - bg_rate_observed, 1.0)

    if family in ("geometry", "geometry_power"):
        mu0 = 0.2 if attenuation else 0.0
        first_shape = float(geometry_kernel(np.array([d0[0]]), mu=mu0)[0])
        A0 = max(first_signal / max(first_shape, 1e-30), 1e-12)
    else:
        A0 = max(first_signal * (d0[0] / 0.1)**2, 1e-12)

    p0, lower, upper = [], [], []
    for name in names:
        if name == "B_extra":
            p0.append(0.0)
            lower.append(-np.inf)
            upper.append(np.inf)
        elif name == "log_A":
            p0.append(np.log(A0))
            lower.append(np.log(1e-12))
            upper.append(np.log(1e20))
        elif name == "p":
            p0.append(2.0)
            lower.append(POWER_BOUNDS[0])
            upper.append(POWER_BOUNDS[1])
        elif name == "mu_per_m":
            p0.append(0.2)
            lower.append(MU_BOUNDS[0])
            upper.append(MU_BOUNDS[1])

    return np.asarray(p0), (np.asarray(lower), np.asarray(upper)), bg_true

#-------- Covariance Matrix --------

#Assemble independent counting/ distance terms plus correlated nuisance
# contributions from shared background and (when enabled) offset uncertainty.
#Assemble the covariance matrix for one model configuration.
def covariance_matrix(theta, family, attenuation, add_constant,
                      offset, dead_model, tau, bg_true):
    #Build covariance from:
    # 1.independent Poisson counting variance;
    # 2.shared background-rate uncertainty (correlated across all distances);
    # 3.per-distance ruler uncertainty integrated by Gauss-Hermite quadrature;
    # 4.shared offset uncertainty (correlated across all distances).
    #The covariance depends on the model parameters, so it is recomputed during
    # fitting. The GLS objective uses the whitened residual vector.
    
    d0 = distances + offset
    dnodes = d0[:, None] + distance_shifts[None, :]

    # Predicted rates at distance quadrature nodes.
    node_rates = predict_observed_rate(
        theta, family, attenuation, add_constant, dnodes,
        0.0, dead_model, tau, bg_true
    )

    mean_rates = np.sum(node_rates * quad_w[None, :], axis=1)
    distance_variance = np.sum(
        quad_w[None, :] * (node_rates - mean_rates[:, None])**2,
        axis=1
    )

    cov = np.diag(counting_variance + distance_variance)

    #Shared background uncertainty propagated through the dead- time response.
    #The derivative is evaluated at the total source + background rate at
    # each distance, as in the covariance treatment.
    centre_signal = source_rate(
        theta, family, attenuation, add_constant, d0
    )
    total_true_rate = centre_signal + bg_true

    if dead_model == "none":
        derivative_bg = np.ones_like(d0)
    else:  #Non- paralysable dead- time correction
        derivative_bg = 1.0 / (1.0 + tau * total_true_rate)**2

    bg_vector = derivative_bg * bg_sigma_observed
    cov += np.outer(bg_vector, bg_vector)

    #A common offset uncertainty moves every distance together.
    if offset != 0.0:
        step = 1e-5
        plus = predict_observed_rate(
            theta, family, attenuation, add_constant,
            d0 + step, 0.0, dead_model, tau, bg_true
        )
        minus = predict_observed_rate(
            theta, family, attenuation, add_constant,
            d0 - step, 0.0, dead_model, tau, bg_true
        )
        derivative_offset = (plus - minus) / (2.0 * step)
        offset_vector = derivative_offset * SHARED_OFFSET_SIGMA_M
        cov += np.outer(offset_vector, offset_vector)

    #Small numerical diagonal term ensures stable Cholesky decomposition.
    cov += np.eye(len(distances)) * 1e-12
    return mean_rates, cov

#-------- Fit One Model --------

#Fit one point in the full model grid using covariance- whitened residuals.
#Fit one configuration using covariance- whitened residuals.
def fit_model(family, attenuation, add_constant, offset, dead_model, tau):
    p0, bounds, bg_true = initial_and_bounds(
        family, attenuation, add_constant, offset, dead_model, tau
    )

    def whitened_residuals(theta):
        predicted, cov = covariance_matrix(
            theta, family, attenuation, add_constant,
            offset, dead_model, tau, bg_true
        )
        try:
            chol = np.linalg.cholesky(cov)
            return solve_triangular(
                chol, observed_rates - predicted,
                lower=True, check_finite=False
            )
        except np.linalg.LinAlgError:
            return np.full(len(distances), 1e20)

    result = least_squares(
        whitened_residuals,
        p0,
        bounds=bounds,
        method="trf",
        x_scale="jac",
        max_nfev=5000
    )

    #A small number of difficult combinations can hit the evaluation ceiling.
    #Retry from the first solution rather than dropping that model from the grid.
    if not result.success and np.all(np.isfinite(result.x)):
        result = least_squares(
            whitened_residuals,
            result.x,
            bounds=bounds,
            method="trf",
            x_scale="jac",
            max_nfev=10000
        )

    if not result.success or np.any(~np.isfinite(result.x)):
        raise RuntimeError(result.message)

    final_residuals = whitened_residuals(result.x)
    chi_squared = float(final_residuals @ final_residuals)
    n_parameters = len(result.x)
    dof = len(distances) - n_parameters
    p_value = float(chi2.sf(chi_squared, dof)) if dof > 0 else np.nan
    reduced_chi_squared = chi_squared / dof if dof > 0 else np.nan

    names = parameter_names(family, attenuation, add_constant)
    parameter_values = dict(zip(names, result.x))

    return {
        "family": family,
        "attenuation": attenuation,
        "extra_constant": add_constant,
        "deadtime_model": dead_model,
        "tau_s": tau,
        "offset_m": offset,
        "chi2": chi_squared,
        "dof": dof,
        "reduced_chi2": reduced_chi_squared,
        "p_value": p_value,
        "success": result.success,
        "parameters": parameter_values,
    }

#-------- Run The Model Grid --------

#Worker wrapper used by the parallel model- grid evaluation.
def run_one_model(task):
    #Top- level worker so independent model fits can run in separate processes.
    family, attenuation, add_constant, offset, dead_model, tau = task
    return fit_model(
        family, attenuation, add_constant, offset, dead_model, tau
    )


def model_label(result):
    family_names = {
        "inverse": "Inverse-square",
        "power": "Free power law",
        "geometry": "Finite geometry",
        "geometry_power": "Geometry + power adjustment",
    }
    label = family_names[result["family"]]
    label += " | attenuation " + ("on" if result["attenuation"] else "off")
    label += " | extra constant " + ("on" if result["extra_constant"] else "off")
    if result["deadtime_model"] == "none":
        label += " | no dead time"
    else:
        label += f" | {result['deadtime_model']} {result['tau_s']*1e6:.0f} us"
    label += f" | offset {result['offset_m']*1000:.1f} mm"
    return label

#Evaluate the Cartesian product of model families and correction scenarios,
# then rank successful fits by their goodness- of- fit statistic.
def main():
    print("\nDISTANCE- LAW AND CORRECTION MODEL COMPARISON")
    print("=" * 100)
    print(f"Experimental distances: {len(distances)}")
    print(f"Distance range: {distances.min():.3f} to {distances.max():.3f} m")
    print(f"Total pooled counts: {int(counts.sum()):,}")
    print(f"Background rate: {bg_rate_observed:.6f} +/- {bg_sigma_observed:.6f} counts/s")
    print(f"Distance offsets tested: {[f'{x*1000:.1f} mm' for x in DISTANCE_OFFSETS_M]}")
    print(f"Dead-time options: off, or non-paralysable with tau = {DEAD_TIME_S*1e6:.2f} us")
    print("Expected model combinations: 4 x 2 x 2 x 2 x 2 = 64")
    print("Background and shared offset uncertainties are included as correlated terms.")
    print("Distance uncertainty is integrated using Gauss-Hermite quadrature.")
    print("=" * 100)

    #Two dead- time options: correction off, or non- paralysable
    # correction on with the fixed value tau = 2.5 microseconds.
    dead_cases = [
        ("none", 0.0),
        ("nonparalysable", DEAD_TIME_S)
    ]

    constant_options = [False, True] if INCLUDE_EXTRA_CONSTANT_MODELS else [False]

    tasks = [
        (family, attenuation, add_constant, offset, dead_model, tau)
        for family, attenuation, add_constant, offset, (dead_model, tau)
        in itertools.product(
            FAMILIES,
            (False, True),
            constant_options,
            DISTANCE_OFFSETS_M,
            dead_cases
        )
    ]
    total = len(tasks)
    assert total == 64, f"Expected exactly 64 model combinations, got {total}"

    #Independent fits are run in parallel; the cap is deliberately modest.
    max_workers = 1
    results = []
    failures = []

    print(f"Running {total} fits using up to {max_workers} worker processes.",
          flush=True)

    if max_workers == 1:
        for completed, task in enumerate(tasks, start=1):
            try:
                results.append(run_one_model(task))
            except Exception as exc:
                failures.append((completed, task, str(exc)))
            if completed % 10 == 0 or completed == total:
                print(f"Completed {completed}/{total} model fits.", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(run_one_model, task): task for task in tasks
            }
            for completed, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append((completed, task, str(exc)))
                if completed % 10 == 0 or completed == total:
                    print(f"Completed {completed}/{total} model fits.", flush=True)

    for completed, task, message in failures:
        family, attenuation, add_constant, offset, dead_model, tau = task
        print(f"WARNING: fit failed ({completed}/{total}): "
              f"{family}, attenuation={attenuation}, "
              f"constant={add_constant}, dead={dead_model}, "
              f"tau={tau}, offset={offset}: {message}")

    if not results:
        raise RuntimeError("No models fitted successfully.")

    # Print all successful fits, ranked by chi- squared only.
    results.sort(key=lambda item: item["chi2"])

    print(f"\nSUCCESSFUL FITS: {len(results)} / {total}")
    print("\nALL MODELS RANKED BY CHI-SQUARED")
    print("-" * 145)
    header = (
        f"{'Rank':>4} {'Family':<24} {'Atten.':>7} {'B extra':>8} "
        f"{'Dead-time':<16} {'tau (us)':>8} {'Offset (mm)':>11} "
        f"{'chi-squared':>13} {'dof':>5} {'reduced chi2':>14} {'p-value':>12}"
    )
    print(header)
    print("-" * 145)

    for rank, r in enumerate(results, start=1):
        print(
            f"{rank:4d} {r['family']:<24} "
            f"{str(r['attenuation']):>7} "
            f"{str(r['extra_constant']):>8} "
            f"{r['deadtime_model']:<16} "
            f"{r['tau_s']*1e6:8.1f} "
            f"{r['offset_m']*1000:11.1f} "
            f"{r['chi2']:13.3f} "
            f"{r['dof']:5d} "
            f"{r['reduced_chi2']:14.3f} "
            f"{r['p_value']:12.4g}"
        )

    print("\nBEST-FITTING MODEL")
    best = results[0]
    print(model_label(best))
    print(f"chi-squared       = {best['chi2']:.4f}")
    print(f"degrees of freedom= {best['dof']}")
    print(f"reduced chi2      = {best['reduced_chi2']:.4f}")
    print(f"p-value           = {best['p_value']:.6g}")
    print("Fitted parameters:")
    for name, value in best["parameters"].items():
        if name == "log_A":
            print(f"  A = {np.exp(value):.8g}")
        else:
            print(f"  {name} = {value:.8g}")

    #One plot only: reduced chi- squared for all successful model combinations.
    plot_results = sorted(results, key=lambda item: item["reduced_chi2"])
    labels = [model_label(item) for item in plot_results]
    values = np.array([item["reduced_chi2"] for item in plot_results])

    fig_height = max(8, 0.24 * len(plot_results))
    fig, ax = plt.subplots(figsize=(15, fig_height))
    y = np.arange(len(plot_results))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(r"Reduced chi-squared, $\chi^2_\nu$ (log scale)")
    ax.set_title("Reduced chi-squared: all fitted model combinations")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    plt.savefig("All Models Comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
