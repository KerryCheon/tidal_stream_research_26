import numpy as np
import agama
from scipy.optimize import brentq


# --- from NGC6569_Dev3_original ---

def shift_to_gc(xv, center_pos, center_vel):
    """Shift cluster-centric phase space to galactocentric frame."""
    return xv[:, :3] + center_pos, xv[:, 3:] + center_vel


def trajectories(bound_data, sim_data):
    """Extract time-series of bound cluster properties from simulation output."""
    n = len(bound_data)
    if len(sim_data) != n:
        raise ValueError(f"Length mismatch: bound_data {n}, sim_data {len(sim_data)}")
    time_array  = np.zeros(n)
    mass_traj   = np.zeros(n)
    energy_traj = np.zeros(n)
    r_traj      = np.zeros((n, 3))
    d_traj      = np.zeros(n)
    v_traj      = np.zeros((n, 3))
    for i in range(n):
        mass_traj[i]   = bound_data[i]["total mass"]
        energy_traj[i] = bound_data[i]["total energy"]
        r_traj[i, :]   = bound_data[i]["pos"]
        v_traj[i, :]   = bound_data[i]["vel"]
        d_traj[i]      = np.linalg.norm(r_traj[i, :])
        time_array[i]  = sim_data[i]["time"]
    return {
        "mass":     mass_traj,
        "energy":   energy_traj,
        "time":     time_array,
        "pos":      r_traj,
        "vel":      v_traj,
        "distance": d_traj,
    }


# --- generic utilities ---

def sample_king(W, scale_radius, mass, N):
    """Sample N particles from a King W0=W profile."""
    pot = agama.Potential(type="king", W0=W, scaleRadius=scale_radius, mass=mass)
    df  = agama.DistributionFunction(type="quasispherical", potential=pot)
    return agama.GalaxyModel(pot, df).sample(N)


def king_trunc_ratio(W, rgrid=None):
    """Return dimensionless truncation radius r_t / r_scale for King W=W0."""
    if rgrid is None:
        rgrid = np.logspace(-4, 3, 4000)
    p = agama.Potential(type="king", W0=W, scaleRadius=1.0, mass=1.0)
    pts = np.column_stack([rgrid, np.zeros_like(rgrid), np.zeros_like(rgrid)])
    rho = p.density(pts)
    return rgrid[rho > rho.max() * 1e-10].max()


def tidal_radius(R, M, pot):
    """Jacobi tidal radius at galactocentric distance R for satellite mass M."""
    def _Mg(r):
        a_R = pot.force([r, 0.0, 0.0])[0]
        return r * r * (-a_R) / agama.G
    dR  = 1e-3 * R
    dln = (np.log(_Mg(R + dR)) - np.log(_Mg(R - dR))) / \
          (np.log(R + dR)      - np.log(R - dR))
    return R * (M / (_Mg(R) * (3.0 - dln))) ** (1.0 / 3.0)


def apocenter_ic(pot, R_apo, R_peri, time_unit, tmax_gyr=3.0, n=4000):
    """Find the in-plane IC at apocenter that produces periapocenter R_peri.

    Returns (center_pos, center_vel) as galactocentric Cartesian arrays.
    """
    def _Mg(r):
        a_R = pot.force([r, 0.0, 0.0])[0]
        return r * r * (-a_R) / agama.G

    def _orbit_min_r(v_t):
        _, traj = agama.orbit(potential=pot, ic=[R_apo, 0, 0, 0, v_t, 0],
                              time=tmax_gyr / time_unit, trajsize=n)
        return np.sqrt((traj[:, :3] ** 2).sum(axis=1)).min()

    v_circ = np.sqrt(_Mg(R_apo) * agama.G / R_apo)
    v_t = brentq(lambda v: _orbit_min_r(v) - R_peri, 0.1 * v_circ, v_circ)
    return np.array([R_apo, 0.0, 0.0]), np.array([0.0, v_t, 0.0])
