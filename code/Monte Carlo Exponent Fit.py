#Physical Monte Carlo study of geometric distance dependence

#Samples particle emission points across a finite circular source and
# corresponding points on a finite square detector. The geometric
# acceptance is estimated over a range of source-detector separations
# and fitted with a power law, P_detect(d) = C * d**(-n).

#Independent simulations are used to examine the variability in the
# fitted power- law exponent. The model includes geometric acceptance only;
# energy loss, scattering, air attenuation, detector efficiency, and
# dead time are not included.

#Requires: numpy, scipy, matplotlib.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#---------------------------- SETTINGS ---------------------------------

RNG_SEED = 20261121

N_SIMULATIONS = 100
N_TRAJECTORIES = 100_000

SOURCE_DIAMETER_M = 0.00975
DETECTOR_WIDTH_M = 0.010
DETECTOR_HEIGHT_M = 0.010

DISTANCES_M = np.linspace(0.05, 1.00, 24)

FIT_MIN_DISTANCE_M = float(DISTANCES_M.min())
FIT_MAX_DISTANCE_M = float(DISTANCES_M.max())

#---------------------- PHYSICAL MONTE CARLO ----------------------------

#Sampling radius as R*sqrt(U) gives a uniform distribution per unit area,
#rather than over- weighting the centre of the circular source.
#Generate source positions uniformly over the circular face.
def sample_uniform_disk(rng, n, radius):
    #Sample points uniformly by area over a circular source face.
    radius_sample = radius * np.sqrt(rng.random(n))
    angle = 2.0 * np.pi * rng.random(n)
    x = radius_sample * np.cos(angle)
    y = radius_sample * np.sin(angle)
    return x, y

#Generate detector positions uniformly over the rectangular active face.
def sample_uniform_rectangle(rng, n, width, height):
    #Sample points uniformly over the active detector face.
    x = rng.uniform(-width / 2.0, width / 2.0, n)
    y = rng.uniform(-height / 2.0, height / 2.0, n)
    return x, y

#Estimate finite- area acceptance by averaging the solid-angle contribution
# over random source and detector surface points.
def simulate_geometric_acceptance(distance_m, rng):
    
    #Estimate the isotropic-emission probability of hitting the detector.
    #Source and detector points are sampled uniformly. Each sampled ray is
    # weighted by its isotropic solid-angle contribution:
    #  dOmega = cos(theta) * dA / r**2
    #  cos(theta) = distance_m / r
    #Therefore:
    # P = A_detector/(4*pi) * mean(distance_m / r**3)
    
    if distance_m <= 0:
        raise ValueError("Source-to-detector distance must be positive.")

    source_radius = SOURCE_DIAMETER_M / 2.0
    detector_area = DETECTOR_WIDTH_M * DETECTOR_HEIGHT_M

    sx, sy = sample_uniform_disk(rng, N_TRAJECTORIES, source_radius)
    dx, dy = sample_uniform_rectangle(
        rng, N_TRAJECTORIES, DETECTOR_WIDTH_M, DETECTOR_HEIGHT_M
    )

    lateral_x = dx - sx
    lateral_y = dy - sy

    r_squared = (
        distance_m**2 + lateral_x**2 + lateral_y**2
    )
    r = np.sqrt(r_squared)

    trajectory_weights = distance_m / (r_squared * r)
    prefactor = detector_area / (4.0 * np.pi)

    probability = prefactor * float(np.mean(trajectory_weights))
    standard_error = (
        prefactor
        * float(np.std(trajectory_weights, ddof=1))
        / np.sqrt(N_TRAJECTORIES)
    )

    return probability, standard_error

#------------------------------ FIT -------------------------------------

def power_law(distance, amplitude, exponent):
    #Geometric acceptance power- law approximation.
    return amplitude * distance**(-exponent)


def fit_power_law(distances, probabilities, probability_errors):
    #Fit amplitude and exponent to one simulated acceptance curve.
    fit_mask = (
        (distances >= FIT_MIN_DISTANCE_M)
        & (distances <= FIT_MAX_DISTANCE_M)
        & (probability_errors > 0)
        & np.isfinite(probabilities)
        & np.isfinite(probability_errors)
    )

    if np.count_nonzero(fit_mask) < 3:
        raise RuntimeError("Not enough valid points to fit the power law.")

    #Weight the fit by the Monte Carlo standard error at each separation.
    fit_parameters, covariance = curve_fit(
        power_law,
        distances[fit_mask],
        probabilities[fit_mask],
        p0=(probabilities[fit_mask][0] * distances[fit_mask][0]**2, 2.0),
        sigma=probability_errors[fit_mask],
        absolute_sigma=True,
        bounds=([0.0, 0.0], [np.inf, 6.0]),
        maxfev=20000
    )

    return fit_parameters, covariance

#------------------------- PLOTTING -------------------------------------

def plot_acceptance_curve(mean_probabilities, mean_probability_errors,
                          mean_fit_parameters):
    #Plot the mean acceptance curve and its fitted power law.
    mean_amplitude, mean_exponent = mean_fit_parameters

    distance_fine = np.linspace(
        DISTANCES_M.min(), DISTANCES_M.max(), 400
    )
    fitted_curve = power_law(
        distance_fine, mean_amplitude, mean_exponent
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        DISTANCES_M,
        mean_probabilities,
        yerr=mean_probability_errors,
        fmt="o",
        capsize=3,
        label="Mean Monte Carlo geometric acceptance"
    )
    ax.plot(
        distance_fine,
        fitted_curve,
        label=f"Power-law fit: n = {mean_exponent:.6f}"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Source-to-detector separation (m)")
    ax.set_ylabel("Probability of a particle reaching the detector")
    ax.set_title("Finite-source, finite-detector particle Monte Carlo")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    plt.savefig("Exponent Fit.png")
    fig.tight_layout()

def plot_exponent_histogram(fitted_exponents):
    from matplotlib.ticker import ScalarFormatter

    fitted_exponents = np.asarray(fitted_exponents, dtype=float)
    n_min = float(np.min(fitted_exponents))
    n_max = float(np.max(fitted_exponents))
    mean_exponent = float(np.mean(fitted_exponents))
    spread = n_max - n_min

    fig, ax = plt.subplots(figsize=(9, 5.5))

    #Choose bins across the observed simulation results only.
    if spread > 0:
        padding = max(0.08 * spread, 1e-10)
        hist_range = (n_min - padding, n_max + padding)
        ax.hist(
            fitted_exponents,
            bins="auto",
            range=hist_range,
            edgecolor="black",
            linewidth=1.0,
            rwidth=0.9
        )
        ax.set_xlim(hist_range)
    else:
        #All simulations returned exactly the same floating- point value.
        padding = max(abs(mean_exponent) * 1e-7, 1e-7)
        ax.hist(
            fitted_exponents,
            bins=1,
            range=(mean_exponent - padding, mean_exponent + padding),
            edgecolor="black",
            linewidth=1.0,
            rwidth=0.9
        )
        ax.set_xlim(mean_exponent - padding, mean_exponent + padding)
        ax.text(
            0.5, 0.90,
            "All fitted exponents are identical",
            transform=ax.transAxes,
            ha="center",
            va="top"
        )

    ax.axvline(
        mean_exponent,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_exponent:.8f}"
    )

    visible_min, visible_max = ax.get_xlim()
    if visible_min <= 2.0 <= visible_max:
        ax.axvline(
            2.0,
            color="black",
            linestyle=":",
            linewidth=2,
            label="Inverse-square reference: n = 2"
        )
    else:
        ax.plot(
            [],
            [],
            color="black",
            linestyle=":",
            linewidth=2,
            label="Inverse-square reference: n = 2 (outside zoom)"
        )

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel("Fitted power-law exponent, n")
    ax.set_ylabel("Frequency (number of simulations)")
    ax.set_title(
        f"Distribution of fitted exponents from {len(fitted_exponents)} "
        "independent Monte Carlo simulations"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

#------------------------- MAIN SIMULATION ------------------------------

#Run independent geometry simulations, refit each acceptance curve, and
# use the spread of fitted exponents to quantify simulation variability.
def main():
    rng = np.random.default_rng(RNG_SEED)

    fitted_exponents = np.empty(N_SIMULATIONS)
    fitted_amplitudes = np.empty(N_SIMULATIONS)
    all_probabilities = np.empty((N_SIMULATIONS, len(DISTANCES_M)))

    print("PHYSICAL PARTICLE-TRAJECTORY MONTE CARLO")
    print(f"Independent simulations: {N_SIMULATIONS}")
    print(f"Trajectories per distance per simulation: {N_TRAJECTORIES:,}")
    print(
        "Total sampled source-detector pairs: "
        f"{N_SIMULATIONS * len(DISTANCES_M) * N_TRAJECTORIES:,}"
    )
    print(f"Source diameter: {SOURCE_DIAMETER_M * 1e3:.2f} mm")
    print(
        "Detector active area: "
        f"{DETECTOR_WIDTH_M * 1e3:.1f} x "
        f"{DETECTOR_HEIGHT_M * 1e3:.1f} mm"
    )
    print("\nRunning independent simulations...")

    for simulation in range(N_SIMULATIONS):
        probabilities = np.empty(len(DISTANCES_M))
        probability_errors = np.empty(len(DISTANCES_M))

        for i, distance in enumerate(DISTANCES_M):
            probabilities[i], probability_errors[i] = (
                simulate_geometric_acceptance(distance, rng)
            )

        fit_parameters, _ = fit_power_law(
            DISTANCES_M, probabilities, probability_errors
        )
        amplitude_fit, exponent_fit = fit_parameters

        all_probabilities[simulation, :] = probabilities
        fitted_amplitudes[simulation] = amplitude_fit
        fitted_exponents[simulation] = exponent_fit

        if (simulation + 1) % 10 == 0:
            print(
                f"Completed simulation {simulation + 1}/{N_SIMULATIONS} | "
                f"Fitted exponent n = {exponent_fit:.6f}"
            )

    #---------------------- EXPONENT STATISTICS -------------------------

    mean_exponent = float(np.mean(fitted_exponents))
    sd_exponent = float(np.std(fitted_exponents, ddof=1))
    median_exponent = float(np.median(fitted_exponents))
    q_low, q_high = np.quantile(fitted_exponents, [0.025, 0.975])

    print("\nPOWER-LAW EXPONENT MONTE CARLO RESULTS")
    print(f"Number of simulations: {N_SIMULATIONS}")
    print(f"Mean fitted exponent: {mean_exponent:.8f}")
    print(f"Median fitted exponent: {median_exponent:.8f}")
    print(f"Standard deviation: {sd_exponent:.8f}")
    print(f"Minimum fitted exponent: {np.min(fitted_exponents):.8f}")
    print(f"Maximum fitted exponent: {np.max(fitted_exponents):.8f}")
    print(
        f"Central 95% interval: [{q_low:.8f}, {q_high:.8f}]"
    )

    #---------------------- MEAN ACCEPTANCE CURVE -----------------------

    mean_probabilities = np.mean(all_probabilities, axis=0)
    #Standard error of the mean acceptance across independent simulations.
    mean_probability_errors = (
        np.std(all_probabilities, axis=0, ddof=1)
        / np.sqrt(N_SIMULATIONS)
    )

    mean_fit_parameters, _ = fit_power_law(
        DISTANCES_M,
        mean_probabilities,
        mean_probability_errors
    )

    print(
        "\nExponent fitted to the mean acceptance curve: "
        f"{mean_fit_parameters[1]:.8f}"
    )

    #Both plots are created after the simulations and exponent fits finish.
    plot_acceptance_curve(
        mean_probabilities,
        mean_probability_errors,
        mean_fit_parameters
    )
    plot_exponent_histogram(fitted_exponents)

    #A single show call displays both figures together in standard
    # interactive Matplotlib backends.
    plt.savefig("Exponent Histogram.png")
    plt.show()

    print("\nAll fitted exponent values:")
    for i, value in enumerate(fitted_exponents, start=1):
        print(f"Simulation {i:3d}: n = {value:.8f}")

if __name__ == "__main__":
    main()
