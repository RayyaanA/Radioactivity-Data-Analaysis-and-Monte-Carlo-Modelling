#Monte Carlo study of beta- electron transmission through air

#Samples an approximate combined Sr- 90/ Y- 90 beta spectrum and estimates
# the fraction of electrons whose CSDA range exceeds the air areal density
# for a set of path lengths. An effective exponential is fitted to the
# resulting transmission curve, and independent simulations are used to
# examine the variability of the fitted attenuation coefficient.

#The calculation uses an allowed-spectrum approximation and an empirical
# CSDA range relation. It assumes straight- line paths and does not model
# energy- loss straggling, multiple scattering, source encapsulation,
# backscatter, detector thresholds, or detector response.

#Requires: numpy, scipy, matplotlib.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Fixed seed ensures the sampled spectra can be reproduced.
SEED = 20261030
N_PARTICLES = 1_000_000
N_SIMULATIONS = 100

DISTANCES_M = np.linspace(0.0, 0.85, 18)
AIR_DENSITY_G_CM3 = 0.001205
SR90_ENDPOINT_MEV = 0.546
Y90_ENDPOINT_MEV = 2.28
SR90_FRACTION = 0.5

rng = np.random.default_rng(SEED)

#Allowed beta-spectrum shape, omitting the Fermi/ Coulomb correction:
# p(E) * W * (E0-E)^2, where W is total electron energy.
#Sample electron kinetic energies from the approximate allowed beta spectrum.
def sample_allowed_beta(endpoint_mev, n):
    me = 0.51099895
    grid = np.linspace(1e-5, endpoint_mev, 30000)
    total_energy = grid + me
    momentum = np.sqrt(grid * (grid + 2.0*me))
    pdf = momentum * total_energy * np.maximum(endpoint_mev-grid, 0.0)**2
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, grid)

#Mix the Sr- 90 and Y- 90 branches using the assumed branching fraction.
#Combine the two beta branches using the assumed equal mixture.
def sample_spectrum(n):
    sr = rng.random(n) < SR90_FRACTION
    energy = np.empty(n)
    n_sr = np.count_nonzero(sr)
    energy[sr] = sample_allowed_beta(SR90_ENDPOINT_MEV, n_sr)
    energy[~sr] = sample_allowed_beta(Y90_ENDPOINT_MEV, n-n_sr)
    return energy

#Convert kinetic energy to an approximate mass thickness (g/cm^2)
# that an electron can traverse under the CSDA approximation.
#Convert electron kinetic energy to an approximate CSDA mass range.
def csda_range_g_cm2(E_MeV):
    E = np.maximum(np.asarray(E_MeV, dtype=float), 1e-5)
    return 0.412 * E**(1.265 - 0.0954*np.log(E))

#Numerically invert the range- energy relation for a specified residual range.
def residual_energy_from_range(residual_range, endpoint_max=Y90_ENDPOINT_MEV):
    grid_E = np.geomspace(1e-5, endpoint_max, 30000)
    grid_R = csda_range_g_cm2(grid_E)
    return np.interp(residual_range, grid_R, grid_E, left=0.0, right=endpoint_max)

#Convert air path length from metres to centimetres before multiplying
# by density, giving the traversed air areal density in g/cm^2.
areal_density = AIR_DENSITY_G_CM3 * DISTANCES_M * 100.0

#Effective exponential used to summarise the simulated transmission curve.
def exp_model(d, mu):
    return np.exp(-mu*d)

#Store the fitted attenuation coefficient from every simulation.
mu_values = np.empty(N_SIMULATIONS)

#Keep the first simulation's transmission curve for the original plot.
first_transmission = None
first_transmission_unc = None
first_mu_eff = None
first_mu_unc = None

print("TOY MONTE CARLO: SR-90/Y-90 BETA TRANSMISSION THROUGH AIR")
print(f"Independent simulations: {N_SIMULATIONS}")
print(f"Particles per simulation: {N_PARTICLES:,}")
print(f"Total particle samples: {N_SIMULATIONS * N_PARTICLES:,}")

#Run the physical Monte Carlo simulations.
#Each repeat resamples the beta spectrum independently to estimate
#Monte Carlo variability in the transmission curve and fitted mu_eff.
for simulation in range(N_SIMULATIONS):

    energy0 = sample_spectrum(N_PARTICLES)
    range0 = csda_range_g_cm2(energy0)

    #An electron is counted as transmitted if its initial CSDA range
    # exceeds the air areal density along the path.
    transmission = np.array([
        np.mean(range0 > x) for x in areal_density
    ])

    #Binomial standard error for the transmitted fraction at each distance.
    transmission_unc = np.sqrt(
        np.maximum(
            transmission*(1-transmission),
            1/N_PARTICLES
        ) / N_PARTICLES
    )

    #Fit only distances where transmission is measurably below 1.
    #Exclude zero- distance and near- unity points, which carry little
    # information about the exponential fall- off.
    mask = (
        (DISTANCES_M >= 0.05)
        & (transmission > 0)
        & (transmission < 0.9999)
    )

    #Fit a one- parameter effective exponential; mu_eff summarises this
    # spectrum/ range model and is not a microscopic constant for all betas.
    popt, pcov = curve_fit(
        exp_model,
        DISTANCES_M[mask],
        transmission[mask],
        sigma=np.maximum(transmission_unc[mask], 1e-6),
        absolute_sigma=True,
        p0=[0.1],
        bounds=([0.0], [100.0])
    )

    mu_values[simulation] = popt[0]

    mu_unc = (
        np.sqrt(pcov[0, 0])
        if np.isfinite(pcov[0, 0])
        else np.nan
    )

    #Retain the first simulation for the original transmission plot.
    #Preserve one representative realisation for the transmission plot.
    if simulation == 0:
        first_transmission = transmission.copy()
        first_transmission_unc = transmission_unc.copy()
        first_mu_eff = popt[0]
        first_mu_unc = mu_unc

    if (simulation + 1) % 10 == 0:
        print(
            f"Completed simulation {simulation + 1}/{N_SIMULATIONS}"
        )


#Report the first simulation's fitted coefficient.
print("\nFIRST SIMULATION")
print(
    f"Fitted effective exponential coefficient: "
    f"mu = {first_mu_eff:.6g} +/- {first_mu_unc:.2g} m^-1"
)

#Original transmission plot, using the first simulation.
plt.figure(figsize=(9, 5.5))

plt.errorbar(
    DISTANCES_M,
    first_transmission,
    yerr=first_transmission_unc,
    fmt="o",
    capsize=3,
    label="Monte Carlo transmission"
)

d_plot = np.linspace(0, DISTANCES_M.max(), 250)

plt.plot(
    d_plot,
    exp_model(d_plot, first_mu_eff),
    label=f"Exponential fit: mu={first_mu_eff:.3g} m$^{{-1}}$"
)

plt.xlabel("Air path length (m)")
plt.ylabel("Fraction of beta electrons not stopped in air")
plt.title("Approximate beta-spectrum transmission through air")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    "beta_air_attenuation_monte_carlo.png",
    dpi=180,
    bbox_inches="tight"
)
plt.show()

#Histogram of fitted effective attenuation coefficients.
plt.figure(figsize=(9, 5.5))

plt.hist(
    mu_values,
    bins=20,
    edgecolor="black",
    linewidth=0.8,
    alpha=0.8
)

plt.axvline(
    np.mean(mu_values),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean mu = {np.mean(mu_values):.4f} m$^{{-1}}$"
)

plt.xlabel(r"Fitted effective attenuation coefficient, $\mu_{\mathrm{eff}}$ (m$^{-1}$)")
plt.ylabel("Number of simulations")
plt.title("Monte Carlo distribution of effective air-attenuation coefficients")
plt.grid(axis="y", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("air attenuation histogram.png")
plt.show()
