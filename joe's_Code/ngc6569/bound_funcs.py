import numpy as np


def get_bound_particles(sim_data, mass, max_iter=20, tol=1e-8, min_bound=1):
    """Compute bound-particle summaries for each saved simulation snapshot.

    Particles are classified as bound when their satellite-frame energy is
    negative, using the self-gravity potential stored in each snapshot as
    ``phi`` and kinetic energy relative to the iterated bound-particle COM.
    """
    mass = np.asarray(mass)
    if mass.ndim != 1:
        raise ValueError("mass must be a one-dimensional array")

    bound_data = []
    previous_bound = None

    for snapshot in sim_data:
        pos = np.asarray(snapshot["pos"])
        vel = np.asarray(snapshot["vel"])
        phi = np.asarray(snapshot["phi"])

        if pos.shape != vel.shape or pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("snapshot 'pos' and 'vel' must both have shape (N, 3)")
        if len(pos) != len(mass) or len(phi) != len(mass):
            raise ValueError("snapshot arrays and mass must have the same length")

        candidate = previous_bound.copy() if previous_bound is not None else np.ones(len(mass), dtype=bool)
        if np.count_nonzero(candidate) <= min_bound:
            candidate = np.ones(len(mass), dtype=bool)

        for _ in range(max_iter):
            candidate_mass = mass[candidate]
            total_mass = np.sum(candidate_mass)
            if total_mass <= 0:
                break

            pos_cm = np.average(pos[candidate], axis=0, weights=candidate_mass)
            vel_cm = np.average(vel[candidate], axis=0, weights=candidate_mass)

            rel_vel = vel - vel_cm
            energy = phi + 0.5 * np.sum(rel_vel**2, axis=1)
            bound = energy < 0

            if np.count_nonzero(bound) <= min_bound:
                break
            if np.array_equal(bound, candidate):
                candidate = bound
                break

            old_pos_cm = pos_cm
            candidate = bound
            new_mass = np.sum(mass[candidate])
            if new_mass <= 0:
                break
            new_pos_cm = np.average(pos[candidate], axis=0, weights=mass[candidate])
            if np.linalg.norm(new_pos_cm - old_pos_cm) < tol:
                break

        bound = candidate
        bound_mass = mass[bound]
        total_mass = np.sum(bound_mass)

        if total_mass > 0:
            pos_cm = np.average(pos[bound], axis=0, weights=bound_mass)
            vel_cm = np.average(vel[bound], axis=0, weights=bound_mass)
        else:
            pos_cm = np.full(3, np.nan)
            vel_cm = np.full(3, np.nan)

        rel_pos = pos - pos_cm
        rel_vel = vel - vel_cm
        particle_energy = phi + 0.5 * np.sum(rel_vel**2, axis=1)
        total_energy = np.sum(mass[bound] * particle_energy[bound])
        r_bound = np.max(np.linalg.norm(rel_pos[bound], axis=1)) if np.any(bound) else 0.0

        bound_data.append({
            "bound": bound,
            "mask": bound,
            "total mass": total_mass,
            "total energy": total_energy,
            "pos": pos_cm,
            "vel": vel_cm,
            "r_bound": r_bound,
            "energy": particle_energy,
        })
        previous_bound = bound

    return bound_data
