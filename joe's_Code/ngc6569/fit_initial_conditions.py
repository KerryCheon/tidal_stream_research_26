"""Fit the 1-Gyr-ago initial conditions (theta) so that forward-integrating
the cluster reproduces the observed present-day state.

theta = [log10(M), W0, log10(r_scale), x, y, z, vx, vy, vz]

    f(theta):  sample King(M, W0, r_scale) -> place at (x,y,z,vx,vy,vz)
               -> kdk_leapfrog forward for T_BACK -> bound particles at t=0
               -> theta_model = [log10(M_bound), x, y, z, vx, vy, vz]

    objective: L(theta) = 1/2 sum_i [(theta_model_i - theta_observed_i)/sigma_i]^2

Notes
-----
- W0 and r_scale are held fixed by default (see FIT_STRUCTURE below); the
  present-day snapshot alone weakly constrains them. Flip FIT_STRUCTURE on
  once phase-space + mass fitting is reliable.
- Sampling the King model is stochastic. THETA_SEED pins the RNG per-call so
  the objective is deterministic in theta (required for a gradient-based
  optimizer -- otherwise it's fitting sampling noise, not the physics).
- Use a small N_BODY_FIT while searching, then re-run the best-fit theta at
  full N to get the final answer / uncertainty.
"""
import numpy as np
from scipy.optimize import least_squares

import agama

from gc_initial_conditions import sample_king_model
from leap_frog import kdk_leapfrog
from bound_funcs import get_bound_particles

FIT_STRUCTURE = False   # set True to also fit W0, log10(r_scale)
THETA_SEED = 12345


def unpack_theta(theta, W0_fixed=None, r_scale_fixed=None):
    if FIT_STRUCTURE:
        logM, W0, log_r_scale, x, y, z, vx, vy, vz = theta
    else:
        logM, x, y, z, vx, vy, vz = theta
        W0, log_r_scale = W0_fixed, np.log10(r_scale_fixed)

    M = 10.0 ** logM
    r_scale = 10.0 ** log_r_scale
    pos_ic = np.array([x, y, z])
    vel_ic = np.array([vx, vy, vz])
    return M, W0, r_scale, pos_ic, vel_ic


def pack_theta(M, W0, r_scale, pos_ic, vel_ic):
    base = [np.log10(M)]
    if FIT_STRUCTURE:
        base += [W0, np.log10(r_scale)]
    base += list(pos_ic) + list(vel_ic)
    return np.array(base)


def forward_model(theta, pot, T_back_gyr, time_unit, Nbody, tau, eps,
                   W0_fixed=None, r_scale_fixed=None, seed=THETA_SEED):
    """f(theta): run the cluster forward from t=-T_back_gyr to t=0.

    Returns theta_model = [log10(M_bound), x, y, z, vx, vy, vz] at t=0.
    """
    M, W0, r_scale, pos_ic, vel_ic = unpack_theta(theta, W0_fixed, r_scale_fixed)

    # Fixed seed -> same theta always gives the same sampled cluster.
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        xv, mass, _, _ = sample_king_model(W0, r_scale, M, Nbody)
    finally:
        np.random.set_state(rng_state)

    pos_0 = xv[:, :3] + pos_ic
    vel_0 = xv[:, 3:] + vel_ic

    nt = int(round(T_back_gyr / time_unit / tau))
    # Only need the final snapshot -> downsample = whole run, last_snapshot=True.
    sim_data = kdk_leapfrog(pot, pos_0, vel_0, mass, nt, tau, agama.G, eps,
                             time_unit, downsample=max(nt, 1), last_snapshot=True)

    bound_data = get_bound_particles(sim_data, mass)
    final = bound_data[-1]

    return np.concatenate([[np.log10(final["total mass"])], final["pos"], final["vel"]])


def residuals(theta, theta_observed, sigma, pot, T_back_gyr, time_unit, Nbody, tau, eps,
              W0_fixed=None, r_scale_fixed=None):
    """Residual vector for scipy.optimize.least_squares.

    least_squares minimizes 1/2 * sum(residuals**2) internally, which is
    exactly L(theta) as defined in the notes -- no separate scalar wrapper
    needed unless you switch to a non-least-squares optimizer.
    """
    theta_model = forward_model(theta, pot, T_back_gyr, time_unit, Nbody, tau, eps,
                                 W0_fixed, r_scale_fixed)
    return (theta_model - theta_observed) / sigma


def fit_initial_conditions(theta_guess, theta_observed, sigma, pot, T_back_gyr,
                            time_unit, tau, eps, Nbody_fit=2000,
                            W0_fixed=None, r_scale_fixed=None, bounds=None):
    """Drive the fit. theta_guess is the current backward-orbit-integration
    guess (e.g. from gc_initial_conditions()); theta_observed/sigma come
    from step 2 above.
    """
    result = least_squares(
        residuals, theta_guess,
        args=(theta_observed, sigma, pot, T_back_gyr, time_unit, Nbody_fit, tau, eps,
              W0_fixed, r_scale_fixed),
        bounds=bounds if bounds is not None else (-np.inf, np.inf),
        method="trf" if bounds is not None else "lm",
        diff_step=1e-2,   # finite-diff step; the model is noisy, don't go smaller
    )
    return result
