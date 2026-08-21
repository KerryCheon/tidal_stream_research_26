import numpy as np


def get_tidal_data(bound_data, cluster_data, pot_ext):
    """
    Compute the tidal tensor and effective (centrifugally-corrected) tidal tensor
    at the cluster center of mass for every snapshot, along with their eigenvalues
    and eigenvectors.

    The tidal tensor is the Hessian of the external potential,
    T_ij = d^2(phi)/dx_i dx_j, evaluated at the cluster center of mass. Its
    eigenvalues give the stretching/compressing rate along the principal axes
    of tidal deformation (positive eigenvalue = stretching direction).

    The effective tidal tensor adds the centrifugal correction for the cluster's
    non-inertial, co-rotating frame:

        T_eff = T + Omega^2 * (I - z_hat z_hat^T)

    where Omega is the magnitude of the cluster's orbital angular velocity and
    z_hat is the unit vector along the orbital angular momentum (i.e. the
    rotation axis). This correction only acts in the plane perpendicular to the
    rotation axis, since motion along the rotation axis is unaffected by the
    centrifugal force (e.g. Renaud & Gieles 2013).

    Parameters
    ----------
    bound_data : dict
        Output of get_bound_by_energy, with keys 'time', 'pos' (bound-particle
        center-of-mass position, shape (n_snap, 3)).
    cluster_data : dict
        Output of get_cluster_data(bound_data, sim_data), with keys 'time',
        'omega' (cluster orbital angular velocity vector, shape (n_snap, 3)).
    pot_ext : object
        External potential object with an eval(pos, der=True) method that
        returns the six independent (upper-triangular) components of the
        potential's Hessian at a single position, in the order given by
        np.triu_indices(3): [xx, xy, xz, yy, yz, zz].

    Returns
    -------
    dict
        'time'                     : array, shape (n_snap,) - time of each snapshot
        'tensor'                   : array, shape (n_snap, 3, 3) - tidal tensor
        'eigenvalues'               : array, shape (n_snap, 3) - tidal tensor eigenvalues,
                                       ascending order (from np.linalg.eigh)
        'eigenvectors'              : array, shape (n_snap, 3, 3) - tidal tensor eigenvectors,
                                       columns correspond to eigenvalues above
        'max_eigenvalue'            : array, shape (n_snap,) - largest tidal tensor eigenvalue
        'effective_tensor'          : array, shape (n_snap, 3, 3) - effective (centrifugally
                                       corrected) tidal tensor
        'effective_eigenvalues'     : array, shape (n_snap, 3) - effective tidal tensor
                                       eigenvalues, ascending order
        'effective_eigenvectors'    : array, shape (n_snap, 3, 3) - effective tidal tensor
                                       eigenvectors, columns correspond to eigenvalues above
        'max_effective_eigenvalue'  : array, shape (n_snap,) - largest effective tidal
                                       tensor eigenvalue
    """
    pos = bound_data["pos"]
    time = bound_data["time"]
    omega_vec = cluster_data["omega"]

    n = pos.shape[0]
    if cluster_data["time"].shape[0] != n:
        raise ValueError(
            f"Length mismatch: bound_data has {n} snapshots, "
            f"cluster_data has {cluster_data['time'].shape[0]}"
        )

    triu_indices = np.triu_indices(3)
    identity = np.eye(3)

    tensor = np.zeros((n, 3, 3))
    eigenvalues = np.zeros((n, 3))
    eigenvectors = np.zeros((n, 3, 3))
    max_eigenvalue = np.zeros(n)

    effective_tensor = np.zeros((n, 3, 3))
    effective_eigenvalues = np.zeros((n, 3))
    effective_eigenvectors = np.zeros((n, 3, 3))
    max_effective_eigenvalue = np.zeros(n)

    for i in range(n):
        hess = pot_ext.eval(pos[i, :], der=True)

        H = np.zeros((3, 3))
        H[triu_indices] = hess
        H = H + H.T - np.diag(np.diag(H))

        tensor[i] = H
        vals, vecs = np.linalg.eigh(H)
        eigenvalues[i] = vals
        eigenvectors[i] = vecs
        max_eigenvalue[i] = vals[-1]

        Omega = np.linalg.norm(omega_vec[i])
        if Omega > 0:
            z_hat = omega_vec[i] / Omega
        else:
            z_hat = np.array([0.0, 0.0, 1.0])

        centrifugal = Omega**2 * (identity - np.outer(z_hat, z_hat))
        H_eff = H + centrifugal

        effective_tensor[i] = H_eff
        eff_vals, eff_vecs = np.linalg.eigh(H_eff)
        effective_eigenvalues[i] = eff_vals
        effective_eigenvectors[i] = eff_vecs
        max_effective_eigenvalue[i] = eff_vals[-1]

    return {
        "time": time,
        "tensor": tensor,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "max_eigenvalue": max_eigenvalue,
        "effective_tensor": effective_tensor,
        "effective_eigenvalues": effective_eigenvalues,
        "effective_eigenvectors": effective_eigenvectors,
        "max_effective_eigenvalue": max_effective_eigenvalue,
    }
