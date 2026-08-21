import numpy as np


def get_cluster_frame(r_cm, v_cm, pos, vel):
    """
    Compute unit vectors for the cluster frame at a single timestep and
    transform positions/velocities of all particles into that frame.

    Parameters
    ----------
    r_cm : array, shape (3,)
        Position vector from galactic center to cluster center of mass.
    v_cm : array, shape (3,)
        Velocity vector of cluster center of mass.
    pos : array, shape (N, 3)
        Positions of N particles (galactocentric frame).
    vel : array, shape (N, 3)
        Velocities of N particles (galactocentric frame).

    Returns
    -------
    x_hat, y_hat, z_hat : array, shape (3,)
        Unit vectors of the cluster frame.
    pos_cluster_frame : array, shape (N, 3)
        Positions of all particles in the cluster frame.
    vel_cluster_frame : array, shape (N, 3)
        Velocities of all particles in the cluster frame.
    omega_vec : array, shape (3,)
        Orbital angular velocity vector of the cluster, omega = L / |r_cm|^2.
    """
    # x-unit vector points toward galactic center
    x_hat = -r_cm / np.linalg.norm(r_cm)

    # z-unit vector is perpendicular to orbital plane (r x v direction)
    L_vec = np.cross(r_cm, v_cm)
    z_hat = L_vec / np.linalg.norm(L_vec)

    # y-unit vector completes right-handed system (z x x)
    y_hat = np.cross(z_hat, x_hat)

    # angular velocity vector: omega = L / |r|^2
    omega_vec = L_vec / np.dot(r_cm, r_cm)

    # transform positions and velocities to cluster frame
    pos_cluster_frame = np.column_stack(
        [pos @ x_hat, pos @ y_hat, pos @ z_hat]
    )
    vel_cluster_frame = np.column_stack(
        [vel @ x_hat, vel @ y_hat, vel @ z_hat]
    )

    return x_hat, y_hat, z_hat, pos_cluster_frame, vel_cluster_frame, omega_vec


def get_cluster_data(bound_data, sim_data):
    """
    Compute cluster-frame quantities at every snapshot of a simulation.

    Parameters
    ----------
    bound_data : dict
        Output of get_bound_by_energy, with keys 'time', 'pos', 'vel'
        (bound-particle center of mass position/velocity, shape (n_snap, 3)).
    sim_data : dict
        Output of kdk_leapfrog / kdk_leapfrog_TD, with keys 'pos', 'vel'
        (all-particle positions/velocities, shape (n_snap, N, 3)).

    Returns
    -------
    dict
        'time'   : array, shape (n_snap,) - time of each snapshot (from bound_data)
        'omega'  : array, shape (n_snap, 3) - cluster orbital angular velocity vs time
        'x_hat'  : array, shape (n_snap, 3) - cluster-frame x unit vector vs time
        'y_hat'  : array, shape (n_snap, 3) - cluster-frame y unit vector vs time
        'z_hat'  : array, shape (n_snap, 3) - cluster-frame z unit vector vs time
        'pos'    : array, shape (n_snap, N, 3) - particle positions in cluster frame
        'vel'    : array, shape (n_snap, N, 3) - particle velocities in cluster frame
    """
    n = bound_data["time"].shape[0]
    if sim_data["pos"].shape[0] != n:
        raise ValueError(
            f"Length mismatch: bound_data has {n} snapshots, "
            f"sim_data has {sim_data['pos'].shape[0]}"
        )

    Nbody = sim_data["pos"].shape[1]

    omega = np.empty((n, 3))
    x_hat = np.empty((n, 3))
    y_hat = np.empty((n, 3))
    z_hat = np.empty((n, 3))
    pos_cluster_frame = np.empty((n, Nbody, 3))
    vel_cluster_frame = np.empty((n, Nbody, 3))

    for i in range(n):
        r_cm = bound_data["pos"][i]
        v_cm = bound_data["vel"][i]
        pos = sim_data["pos"][i]
        vel = sim_data["vel"][i]

        (x_hat[i], y_hat[i], z_hat[i],
         pos_cluster_frame[i], vel_cluster_frame[i], omega[i]) = get_cluster_frame(
            r_cm, v_cm, pos, vel
        )

    return {
        "time": bound_data["time"],
        "omega": omega,
        "x_hat": x_hat,
        "y_hat": y_hat,
        "z_hat": z_hat,
        "pos": pos_cluster_frame,
        "vel": vel_cluster_frame,
    }
