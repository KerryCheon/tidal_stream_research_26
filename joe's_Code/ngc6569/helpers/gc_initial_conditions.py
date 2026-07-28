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

# Tell Agama what physical units this script will use everywhere.
agama.setUnits(length=1 * u.kpc, velocity=1 * u.km / u.s, mass=1 * u.Msun)

# Find this helper folder, then step back to the NGC 6569 project folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_NGC6569_DIR = os.path.dirname(_HERE)
# Build the path to the default Milky Way potential file.
_DEFAULT_INI = os.path.join(_NGC6569_DIR, "milkyway", "MWPotential2014.ini")


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
    # Use the supplied potential if one was passed in; otherwise load the default MW model.
    pot = potential if potential is not None else agama.Potential(_DEFAULT_INI)

    # Convert apo/pericenter radii into orbital eccentricity.
    e = (R_apo - R_peri) / (R_apo + R_peri)

    # If eccentricity is basically zero, treat the orbit as circular.
    if e < 1e-8:
        # Ask the potential for the force at x=R_apo, then take the x/radial component.
        a_R = pot.force([R_apo, 0.0, 0.0])[0]
        # Circular speed satisfies V^2 / R = inward acceleration, so V = sqrt(-a_R * R).
        V_a = np.sqrt(-a_R * R_apo)
    else:
        # Get the gravitational potential energy per unit mass at apocenter.
        phi_a = pot.potential([R_apo, 0.0, 0.0])
        # Get the gravitational potential energy per unit mass at pericenter.
        phi_p = pot.potential([R_peri, 0.0, 0.0])
        # Solve conservation of energy + angular momentum for apocenter speed.
        V_a = (1.0 - e) * np.sqrt((phi_a - phi_p) / (2.0 * e))

    # Convert the user-provided inclination angle from degrees to radians.
    i = np.radians(inclination_deg)
    # Place the cluster at apocenter on the positive x-axis.
    pos = np.array([R_apo, 0.0, 0.0])
    # Put all velocity in the tangential y/z direction, tilted by inclination.
    vel = np.array([0.0, V_a * np.cos(i), V_a * np.sin(i)])
    # Return one flat vector: x, y, z, vx, vy, vz.
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
    # Create the self-gravitating King potential for the cluster.
    pot_sat = agama.Potential(type="king", W0=W0_value, scaleRadius=r_scale, mass=mass)
    # Create the distribution function that describes particle phase-space density.
    df_sat = agama.DistributionFunction(type="quasispherical", potential=pot_sat)
    # Draw Nbody particles; xv has positions+velocities and particle_mass stores masses.
    xv, particle_mass = agama.GalaxyModel(pot_sat, df_sat).sample(Nbody)
    # Return the sampled particles plus the model objects in case caller wants to reuse them.
    return xv, particle_mass, pot_sat, df_sat
