"""Cluster setup and analysis utilities used by the NGC 6569 notebooks."""
import numpy as np
import agama

try:
    from .gc_initial_conditions import sample_king_model
except ImportError:  # Support running this file directly from helpers/.
    from gc_initial_conditions import sample_king_model


def shift_to_gc(xv, pos_0, vel_0, verbose=False):
    """Shift cluster-frame phase-space coordinates into the Galactocentric frame."""
    xv = np.asarray(xv)
    pos_0 = np.asarray(pos_0)
    vel_0 = np.asarray(vel_0)

    pos = xv[:, :3] + pos_0
    vel = xv[:, 3:] + vel_0

    if verbose:
        print("checks")
        print(*np.mean(pos, axis=0))
        print(pos_0)
        print(*np.mean(vel, axis=0))
        print(vel_0)

    return pos, vel


def trajectories(bound_data, sim_data):
    """Extract bound-cluster mass, energy, position, velocity, and distance histories."""
    n = len(bound_data)
    if len(sim_data) != n:
        raise ValueError(f"Length mismatch: bound_data has {n} entries, sim_data has {len(sim_data)}")

    time_array = np.zeros(n)
    mass_traj = np.zeros(n)
    energy_traj = np.zeros(n)
    r_traj = np.zeros((n, 3))
    d_traj = np.zeros(n)
    v_traj = np.zeros((n, 3))

    for i in range(n):
        mass_traj[i] = bound_data[i]["total mass"]
        energy_traj[i] = bound_data[i]["total energy"]
        r_traj[i, :] = bound_data[i]["pos"]
        v_traj[i, :] = bound_data[i]["vel"]
        d_traj[i] = np.linalg.norm(r_traj[i, :])
        time_array[i] = sim_data[i]["time"]

    return {
        "mass": mass_traj,
        "energy": energy_traj,
        "time": time_array,
        "pos": r_traj,
        "vel": v_traj,
        "distance": d_traj,
    }


def sample_king(W0_value, r_scale, mass, Nbody):
    """Return King-model particle phase-space samples and particle masses."""
    xv, particle_mass, _, _ = sample_king_model(W0_value, r_scale, mass, Nbody)
    return xv, particle_mass


def king_trunc_ratio(W0, rgrid=None):
    """Return the approximate King-model truncation radius in scale-radius units."""
    if rgrid is None:
        rgrid = np.logspace(-4, 3, 4000)

    pot_king = agama.Potential(type="king", W0=W0, scaleRadius=1.0, mass=1.0)
    pts = np.column_stack((rgrid, np.zeros_like(rgrid), np.zeros_like(rgrid)))
    rho = pot_king.density(pts)
    return rgrid[rho > rho.max() * 1e-10].max()


def apocenter_ic(potential, R_apo, R_peri, time_unit=None, inclination_deg=0.0):
    """Return position and velocity for an apocenter launch point."""
    del time_unit  # Kept for compatibility with older notebook calls.

    e = (R_apo - R_peri) / (R_apo + R_peri)
    if e < 1e-8:
        a_R = potential.force([R_apo, 0.0, 0.0])[0]
        V_a = np.sqrt(-a_R * R_apo)
    else:
        phi_a = potential.potential([R_apo, 0.0, 0.0])
        phi_p = potential.potential([R_peri, 0.0, 0.0])
        V_a = (1.0 - e) * np.sqrt((phi_a - phi_p) / (2.0 * e))

    i = np.radians(inclination_deg)
    pos = np.array([R_apo, 0.0, 0.0])
    vel = np.array([0.0, V_a * np.cos(i), V_a * np.sin(i)])
    return pos, vel


def _radial_second_derivative(potential, pos):
    """Project Agama's upper-triangular Hessian onto the radial direction."""
    pos = np.asarray(pos, dtype=float)
    r = np.linalg.norm(pos)
    if r == 0:
        raise ValueError("radial derivative is undefined at R=0")

    der = np.asarray(potential.eval(pos, der=True), dtype=float)
    dxx, dyy, dzz, dxy, dyz, dzx = der[:6]
    x, y, z = pos
    return -(x * x * dxx + y * y * dyy + z * z * dzz + 2 * x * y * dxy + 2 * y * z * dyz + 2 * z * x * dzx) / r**2


def tidal_radius(R, mass, potential):
    """Estimate the circular-orbit Jacobi radius at galactocentric radius R."""
    pos = np.array([R, 0.0, 0.0], dtype=float)
    force = np.asarray(potential.force(pos), dtype=float)
    omega = np.sqrt(-force[0] / R)
    d2Phi_dr2 = _radial_second_derivative(potential, pos)
    return (agama.G * mass / (omega**2 - d2Phi_dr2)) ** (1.0 / 3.0)
