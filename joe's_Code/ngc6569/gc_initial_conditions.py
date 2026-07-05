"""Initial position & velocity of a GC from Chen-paper orbit parameters.

Method (from 6/30/26 notes): conserve specific angular momentum and energy
between apocenter and pericenter.

    L = R_a V_a = R_p V_p           ->   V_p = (R_a / R_p) V_a
    1/2 V_a^2 + Phi(R_a) = 1/2 V_p^2 + Phi(R_p)
    e = (R_a - R_p) / (R_a + R_p)

Solving for the apocenter speed:

    V_a = (1 - e) * sqrt( (Phi(R_a) - Phi(R_p)) / (2 e) )

The cluster is placed at apocenter on the x-axis; at a turning point the
velocity is purely tangential, so

    position = (R_apo, 0, 0)
    velocity = (0, V_a cos i, V_a sin i)      # i = 0 in-plane, 90 polar

This assumes a spherical Phi(r) (evaluated along the x-axis) -- the same
approximation the paper makes for the tidal radius (valid far from the disk).

CLI:
    python gc_initial_conditions.py R_apo R_peri [inclination_deg]
e.g. python gc_initial_conditions.py 40 10        ->  x y z vx vy vz
"""
import os
import numpy as np
import astropy.units as u
import agama

agama.setUnits(length=1 * u.kpc, velocity=1 * u.km / u.s, mass=1 * u.Msun)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_INI = os.path.join(_HERE, "milkyway", "MWPotential2014.ini")


def gc_initial_conditions(R_apo, R_peri, inclination_deg=0.0, potential=None):
    """Return [x, y, z, vx, vy, vz] (kpc, km/s) for a GC at apocenter.

    Parameters
    ----------
    R_apo, R_peri : float
        Apo- and pericentric galactocentric distances [kpc] (R_apo >= R_peri).
    inclination_deg : float
        0 = in-plane (disk) orbit, 90 = polar. Tilts the velocity vector.
    potential : agama.Potential, optional
        Host potential. Defaults to MWPotential2014 (milkyway/MWPotential2014.ini).
    """
    pot = potential if potential is not None else agama.Potential(_DEFAULT_INI)

    e = (R_apo - R_peri) / (R_apo + R_peri)

    if e < 1e-8:                                   # circular orbit
        a_R = pot.force([R_apo, 0.0, 0.0])[0]      # radial accel (negative)
        V_a = np.sqrt(-a_R * R_apo)
    else:
        phi_a = pot.potential([R_apo, 0.0, 0.0])
        phi_p = pot.potential([R_peri, 0.0, 0.0])
        V_a = (1.0 - e) * np.sqrt((phi_a - phi_p) / (2.0 * e))

    i = np.radians(inclination_deg)
    pos = np.array([R_apo, 0.0, 0.0])
    vel = np.array([0.0, V_a * np.cos(i), V_a * np.sin(i)])
    return np.concatenate([pos, vel])


def sample_king_model(W0_value, r_scale, mass, Nbody):
    """Sample a King-model cluster using Agama's call structure.

    Parameters
    ----------
    W0_value : float
        Dimensionless King model central potential.
    r_scale : float
        King model scale/core radius [kpc].
    mass : float
        Total cluster mass [Msun].
    Nbody : int
        Number of particles to sample.
    """
    pot_sat = agama.Potential(type="king", W0=W0_value, scaleRadius=r_scale, mass=mass)
    df_sat = agama.DistributionFunction(type="quasispherical", potential=pot_sat)
    xv, particle_mass = agama.GalaxyModel(pot_sat, df_sat).sample(Nbody)
    return xv, particle_mass, pot_sat, df_sat


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        sys.exit("usage: python gc_initial_conditions.py R_apo R_peri [inclination_deg]")
    R_apo = float(sys.argv[1])
    R_peri = float(sys.argv[2])
    incl = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    ic = gc_initial_conditions(R_apo, R_peri, incl)
    e = (R_apo - R_peri) / (R_apo + R_peri)
    print(f"# R_apo={R_apo} kpc, R_peri={R_peri} kpc, e={e:.3f}, inclination={incl} deg")
    print(f"# x y z vx vy vz   [kpc, km/s]")
    print(" ".join(f"{v:.6f}" for v in ic))
