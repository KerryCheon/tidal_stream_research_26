import numpy as np


def _jacobi_radius(pos_cm, vel_cm, total_mass, pot_ext, G, t_native):
    """Jacobi (tidal) radius at the cluster COM, same formula as the Jacobi-radius
    cell in NGC6569_Dev3_original.ipynb. Returns np.inf where the tidal term is
    degenerate (e.g. near pericenter, Omega**2 - d2Phi_dr2 <= 0), so callers can
    treat it as "no spatial constraint" rather than take a complex/negative root.
    """
    r_cm = np.linalg.norm(pos_cm)
    L_cm = np.cross(pos_cm, vel_cm)
    Omega = np.linalg.norm(L_cm) / r_cm**2

    x, y, z = pos_cm
    der = pot_ext.eval(pos_cm, t=t_native, der=True)
    d2Phi_dr2 = -(x**2 * der[0] + y**2 * der[1] + z**2 * der[2]
                  + 2*x*y * der[3] + 2*y*z * der[4] + 2*z*x * der[5]) / r_cm**2

    denom = Omega**2 - d2Phi_dr2
    if denom <= 0:
        return np.inf
    return (G * total_mass / denom) ** (1.0 / 3)


def _estimate_dwell_time(first_snapshot, mass, time_unit):
    """Rough internal crossing time of the initial cluster (half-mass radius
    over 3D velocity dispersion, both about the mass-weighted COM), converted
    to the same physical time unit as ``snapshot["time"]``. Used as the
    default dwell time before a particle is permanently written off as
    stripped, so a single-snapshot fluctuation (well under one crossing time)
    can't trigger a permanent exclusion, but a shock a star never recovers
    from still does within a crossing time or so.
    """
    pos0 = np.asarray(first_snapshot["pos"])
    vel0 = np.asarray(first_snapshot["vel"])
    pos_cm = np.average(pos0, axis=0, weights=mass)
    vel_cm = np.average(vel0, axis=0, weights=mass)
    r_half = np.median(np.linalg.norm(pos0 - pos_cm, axis=1))
    rel_vel = vel0 - vel_cm
    sigma_v = np.sqrt(np.mean(np.sum(rel_vel**2, axis=1)))
    if sigma_v <= 0:
        return 0.0
    return (r_half / sigma_v) * time_unit  # native (kpc / (km/s)) -> physical


def get_bound_particles(sim_data, mass, pot_ext, G, time_unit, max_iter=20, tol=1e-8, min_bound=1,
                         dwell_time=None, start_time=0.0):
    """Compute bound-particle summaries for each saved simulation snapshot.

    At each snapshot, a particle is classified as *instantaneously* bound in
    two phases:

    1. The original self-energy iteration (unchanged): converge on the set of
       particles with negative energy relative to the cluster's own self-gravity
       potential (``phi``, from pyfalcon) and the iterated bound-particle COM.
       This is purely a numerical device for locating a self-consistent COM —
       its warm-start is seeded from the *previous snapshot's instantaneous*
       classification (not the cumulative one below), and is allowed to
       fluctuate snapshot to snapshot.
    2. A single Jacobi-radius (tidal) trim: using the *converged* self-bound mass
       and COM from phase 1, compute the instantaneous tidal radius of the host
       potential ``pot_ext`` and drop any self-bound particle sitting beyond it.

    The self-energy-only criterion alone cannot detect tidal stripping: a star
    can be negative-energy relative to the cluster's own self-potential in
    isolation while already sitting well outside the tidal boundary, actively
    being pulled away by the host. Phase 2 accounts for that.

    Phase 2 is deliberately a single evaluation, not folded into the phase-1
    iteration: feeding the tidal radius back through more iterations makes it
    depend on the shrinking candidate's own mass (r_jacobi ~ total_mass**(1/3)),
    which is an unstable positive-feedback loop — a few particles falling
    outside r_jacobi shrinks total_mass, which shrinks r_jacobi, which excludes
    more particles, and so on toward a spurious near-zero-mass fixed point
    within a single snapshot. Computing it once from the already-converged
    phase-1 state avoids that feedback entirely.

    The *instantaneous* bound/unbound split from phases 1-2 is, on its own,
    still not a valid measure of cumulative mass loss: the Jacobi radius
    itself pulses with orbital phase (it shrinks approaching pericenter and
    grows again receding from it), so a star can be legitimately outside the
    instantaneous tidal radius at the sharpest point of a passage and back
    inside it a few snapshots later without ever having actually escaped.
    Real tidal stripping is irreversible, so what's reported (``total mass``,
    ``total energy``, ``bound``, ...) is a *cumulative* bound set built with
    hysteresis: a particle only gets permanently written off once it has
    failed the instantaneous test continuously for at least ``dwell_time``
    (physical time, same units as ``snapshot["time"]``) — long enough that
    it's a real, sustained shock rather than one snapshot's fluctuation at
    the sharpest instant of a passage. Passing the instantaneous test again
    resets that particle's fail-streak to zero. Once written off, a particle
    never re-enters the cumulative set, so the reported mass can only shrink
    over the run, never regrow.

    Parameters
    ----------
    pot_ext : agama.Potential
        Host potential the cluster orbits in (pass the same potential used to
        integrate the orbit, e.g. the rotating Hunter24 potential, not a static
        stand-in, so the tidal radius reflects the field actually acting on it).
    G : float
        Gravitational constant in the same unit system as ``pot_ext``/``mass``.
    time_unit : float
        Conversion factor from ``snapshot["time"]`` (physical units, e.g. Gyr)
        back to the native time unit ``pot_ext.eval(..., t=...)`` expects.
    dwell_time : float, optional
        Minimum continuous time (same units as ``snapshot["time"]``) a
        particle must keep failing the instantaneous bound test before it's
        permanently counted as stripped. Defaults to an estimate of the
        initial cluster's own crossing time (half-mass radius / velocity
        dispersion), computed from the first snapshot.
    start_time : float, optional
        Absolute simulation start time in the external potential's native
        units. Snapshot times are treated as elapsed times from this epoch.
    """
    mass = np.asarray(mass)
    if mass.ndim != 1:
        raise ValueError("mass must be a one-dimensional array")

    bound_data = []
    previous_bound = None
    cumulative_bound = np.ones(len(mass), dtype=bool)
    fail_duration = np.zeros(len(mass))
    prev_time = None
    if dwell_time is None and len(sim_data) > 0:
        dwell_time = _estimate_dwell_time(sim_data[0], mass, time_unit)

    for snapshot in sim_data:
        pos = np.asarray(snapshot["pos"])
        vel = np.asarray(snapshot["vel"])
        phi = np.asarray(snapshot["phi"])
        t_native = start_time + snapshot["time"] / time_unit

        if pos.shape != vel.shape or pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("snapshot 'pos' and 'vel' must both have shape (N, 3)")
        if len(pos) != len(mass) or len(phi) != len(mass):
            raise ValueError("snapshot arrays and mass must have the same length")

        candidate = previous_bound.copy() if previous_bound is not None else np.ones(len(mass), dtype=bool)
        if np.count_nonzero(candidate) <= min_bound:
            candidate = np.ones(len(mass), dtype=bool)

        # Phase 1: original self-energy-only iteration, unchanged.
        for _ in range(max_iter):
            candidate_mass = mass[candidate]
            total_mass = np.sum(candidate_mass)
            if total_mass <= 0:
                break

            pos_cm = np.average(pos[candidate], axis=0, weights=candidate_mass)
            vel_cm = np.average(vel[candidate], axis=0, weights=candidate_mass)

            rel_vel = vel - vel_cm
            energy = phi + 0.5 * np.sum(rel_vel**2, axis=1)
            self_bound = energy < 0

            if np.count_nonzero(self_bound) <= min_bound:
                break
            if np.array_equal(self_bound, candidate):
                candidate = self_bound
                break

            old_pos_cm = pos_cm
            candidate = self_bound
            new_mass = np.sum(mass[candidate])
            if new_mass <= 0:
                break
            new_pos_cm = np.average(pos[candidate], axis=0, weights=mass[candidate])
            if np.linalg.norm(new_pos_cm - old_pos_cm) < tol:
                break

        self_bound_final = candidate
        self_bound_mass = mass[self_bound_final]
        self_bound_total_mass = np.sum(self_bound_mass)

        # Phase 2: single Jacobi-radius trim on top of the converged self-bound set.
        if self_bound_total_mass > 0:
            pos_cm = np.average(pos[self_bound_final], axis=0, weights=self_bound_mass)
            vel_cm = np.average(vel[self_bound_final], axis=0, weights=self_bound_mass)
            r_jacobi = _jacobi_radius(pos_cm, vel_cm, self_bound_total_mass, pot_ext, G, t_native)
            within_tidal_radius = np.linalg.norm(pos - pos_cm, axis=1) <= r_jacobi
        else:
            r_jacobi = np.inf
            within_tidal_radius = np.ones(len(mass), dtype=bool)

        instantaneous_bound = self_bound_final & within_tidal_radius

        t_phys = snapshot["time"]
        dt = 0.0 if prev_time is None else (t_phys - prev_time)
        prev_time = t_phys

        # Track how long each particle has continuously failed the
        # instantaneous test; reset to 0 the moment it passes again.
        fail_duration = np.where(instantaneous_bound, 0.0, fail_duration + dt)
        newly_stripped = fail_duration >= dwell_time

        # Cumulative (monotonically non-increasing) bound set: what actually
        # gets reported. A particle is only permanently written off once its
        # fail-streak has lasted at least dwell_time.
        cumulative_bound = cumulative_bound & ~newly_stripped
        bound = cumulative_bound
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
        specific_kinetic = 0.5 * np.sum(rel_vel**2, axis=1)
        particle_energy = phi + specific_kinetic

        # ``phi`` is the self-potential at each particle.  Its contribution to
        # the system energy needs a factor of 1/2 because every particle pair
        # appears twice in sum(m_i * phi_i).  The full specific potential is
        # still appropriate above for the per-particle boundness test.
        kinetic_energy = np.sum(mass[bound] * specific_kinetic[bound])
        potential_energy = 0.5 * np.sum(mass[bound] * phi[bound])
        total_energy = kinetic_energy + potential_energy
        r_bound = np.max(np.linalg.norm(rel_pos[bound], axis=1)) if np.any(bound) else 0.0

        bound_data.append({
            "bound": bound,
            "mask": bound,
            "total mass": total_mass,
            "total energy": total_energy,
            "pos": pos_cm,
            "vel": vel_cm,
            "r_bound": r_bound,
            "r_jacobi": r_jacobi,
            "energy": particle_energy,
            "kinetic energy": kinetic_energy,
            "potential energy": potential_energy,
        })
        # Phase-1's warm-start next time uses the instantaneous classification
        # (a numerical convenience for finding a stable COM), not the
        # cumulative one — the cumulative set only ever shrinks and would
        # starve phase 1's search space over a long run.
        previous_bound = instantaneous_bound

    return bound_data
