import numpy as np
from center_of_mass import get_center_of_mass


def _cluster_properties(pos, vel, e, mass, use):
    """
    Compute mass-weighted cluster properties for a boolean subset of particles.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
    vel : ndarray, shape (N, 3)
    e : ndarray, shape (N,)
        Energy per unit mass of each particle.
    mass : ndarray, shape (N,)
    use : ndarray of bool, shape (N,)
        True for particles included in the cluster (bound).

    Returns
    -------
    m_tot : float
    e_tot : float
    pos_cm : ndarray, shape (3,)
    vel_cm : ndarray, shape (3,)
    """
    m_tot = np.sum(mass[use])
    e_tot = np.sum(e[use] * mass[use])
    pos_cm = get_center_of_mass(pos[use, :], mass[use])
    vel_cm = get_center_of_mass(vel[use, :], mass[use])
    return m_tot, e_tot, pos_cm, vel_cm


def get_bound_by_energy(sim_data, r_search=10, tol=0.1 / 1000, n_search=10):
    """
    Track bound particles throughout a simulation by iteratively finding the center of mass
    of gravitationally bound particles, using an energy criteria, i.e., energy < 0 for
    bound particles.

    Parameters
    ----------
    sim_data : dict
        Simulation output with keys 'time', 'pos', 'vel', 'phi', 'mass', as returned by
        kdk_leapfrog / kdk_leapfrog_TD.
    r_search : float, optional
        Search radius for finding particles (default: 10)
    tol : float, optional
        Convergence tolerance for center of mass (default: 0.0001)
    n_search : int, optional
        Maximum number of search iterations (default: 10)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'readme': str, brief description of the method and fields in this dictionary
        - 'time': ndarray, shape (n_snap,), physical time of each snapshot
        - 'pos': ndarray, shape (n_snap, 3), mass-weighted center of mass of bound particles
        - 'vel': ndarray, shape (n_snap, 3), mass-weighted center of mass velocity of bound particles
        - 'mass': ndarray, shape (n_snap,), total mass of bound particles
        - 'energy': ndarray, shape (n_snap,), total energy of bound particles
        - 'status': ndarray, shape (n_snap, N), 0 if particle is bound, 1 if unbound
        - 'converged': ndarray of bool, shape (n_snap,), True if the center-of-mass search
          converged within n_search iterations at that timestep (timestep 0 is always True,
          since all particles are assumed bound at t=0)
    """
    readme = (
        "Bound particles identified by energy criterion (e = 0.5*v_rel^2 + phi < 0, "
        "v_rel measured relative to the iteratively refined bound-particle center of "
        "mass). At each timestep, pos/vel/mass/energy give mass-weighted properties of "
        "the bound subset; status[i] is 0/1 (bound/unbound) per particle; converged[i] "
        "flags whether the center-of-mass search converged within n_search iterations."
    )
    times = sim_data["time"]
    pos_all = sim_data["pos"]
    vel_all = sim_data["vel"]
    phi_all = sim_data["phi"]
    mass = sim_data["mass"]

    n = len(times)
    Nbody = pos_all.shape[1]

    # Preallocate output arrays
    cluster_time = np.empty(n)
    cluster_pos = np.empty((n, 3))
    cluster_vel = np.empty((n, 3))
    cluster_mass = np.empty(n)
    cluster_energy = np.empty(n)
    status = np.empty((n, Nbody), dtype=int)
    converged = np.empty(n, dtype=bool)

    # Initialize with first timestep (all particles assumed bound)
    pos_0 = pos_all[0]
    vel_0 = vel_all[0]
    phi = phi_all[0]
    t1 = times[0]

    r_cm = get_center_of_mass(pos_0, mass)
    v_cm = get_center_of_mass(vel_0, mass)

    v = vel_0 - v_cm
    v = np.linalg.norm(v, axis=1)
    e = 0.5 * v**2 + phi

    use = np.ones(Nbody, dtype=bool)
    m_tot, e_tot, pos_cm, vel_cm = _cluster_properties(pos_0, vel_0, e, mass, use)

    cluster_time[0] = t1
    cluster_pos[0] = pos_cm
    cluster_vel[0] = vel_cm
    cluster_mass[0] = m_tot
    cluster_energy[0] = e_tot
    status[0] = np.where(use, 0, 1)
    converged[0] = True

    # Process remaining timesteps
    for i in range(1, n):

        pos = pos_all[i]
        vel = vel_all[i]
        phi = phi_all[i]
        t2 = times[i]

        # find candidate position by Euler update
        dt = t2 - t1
        r = r_cm + v_cm * dt

        # Iteratively refine center of mass position
        for j in range(n_search):

            dr = pos - r
            dr_mag = np.linalg.norm(dr, axis=1)
            # select all particles within the search region
            use = dr_mag < r_search

            # Check if any particles are in search region
            if not np.any(use):
                print(f"Warning: No particles found in search region at timestep {i}")
                # Use all particles as fallback
                use = np.ones(Nbody, dtype=bool)

            # mean velocity of particles within search region
            v_cm = get_center_of_mass(vel[use, :], mass[use])

            # subtract velocity and compute magnitude
            v = vel - v_cm
            v = np.linalg.norm(v, axis=1)

            # compute energy per mass
            e = 0.5 * v**2 + phi
            # bound particles
            bound_mask = e < 0

            # Use mass-weighted center of mass for bound particles
            r_new = get_center_of_mass(pos[bound_mask, :], mass[bound_mask])

            if np.allclose(r_new, r, atol=tol):
                m_tot, e_tot, pos_cm, vel_cm = _cluster_properties(pos, vel, e, mass, bound_mask)
                r_cm = r_new
                did_converge = True
                break
            elif j == n_search - 1:
                m_tot, e_tot, pos_cm, vel_cm = _cluster_properties(pos, vel, e, mass, bound_mask)
                r_cm = r_new
                did_converge = False
            else:
                r = r_new

        cluster_time[i] = t2
        cluster_pos[i] = pos_cm
        cluster_vel[i] = vel_cm
        cluster_mass[i] = m_tot
        cluster_energy[i] = e_tot
        status[i] = np.where(bound_mask, 0, 1)
        converged[i] = did_converge

        t1 = t2

    return {
        "readme": readme,
        "time": cluster_time,
        "pos": cluster_pos,
        "vel": cluster_vel,
        "mass": cluster_mass,
        "energy": cluster_energy,
        "status": status,
        "converged": converged,
    }