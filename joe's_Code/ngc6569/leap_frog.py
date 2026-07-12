import numpy as np
import pyfalcon

from snapshot_store import SnapshotBinaryReader, SnapshotBinaryWriter

def kdk_leapfrog(pot_ext, pos_0, vel_0, mass, nt, tau, G, eps, time_unit, downsample,
                 last_snapshot=True):
    """
    Kick-Drift-Kick leapfrog integrator.
    
    Parameters
    ----------
    pot_ext : object
        External potential object with a force(positions) method that returns
        accelerations due to external forces.
    pos_0 : array_like, shape (N, 3)
        Initial positions of N particles in 3D space.
    vel_0 : array_like, shape (N, 3)
        Initial velocities of N particles in 3D space.
    mass : array_like, shape (N,)
        Masses of N particles.
    nt : int
        Number of integration timesteps to perform.
    tau : float
        Integration timestep in simulation units.
    G : float
        Gravitational constant in simulation units.
    eps : float
        Softening parameter for gravitational force calculation.
    time_unit : float
        Conversion factor from simulation time units to physical units.
    downsample : int
        Save snapshot data every `downsample` timesteps. Must be >= 1.
    last_snapshot : bool, optional
        If True, ensure the final timestep is saved even if not on a 
        downsample boundary. Default is True.
    
    Returns
    -------
    list of dict
        List of simulation snapshots, where each snapshot is a dictionary with keys:
        - 'time': float, physical time in units specified by time_unit
        - 'pos': ndarray, shape (N, 3), particle positions
        - 'vel': ndarray, shape (N, 3), particle velocities  
        - 'phi': ndarray, shape (N,), gravitational potential at each particle
    
    Notes
    -----
    The leapfrog algorithm follows this sequence:
    1. Kick: v_{i+1/2} = v_i + a_i * dt/2
    2. Drift: x_{i+1} = x_i + v_{i+1/2} * dt
    3. Force: compute a_{i+1} from x_{i+1}
    4. Kick: v_{i+1} = v_{i+1/2} + a_{i+1} * dt/2
    
    This method is symplectic and time-reversible.
    
    """
    # Input validation
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    if nt < 0:
        raise ValueError("nt must be non-negative")

    # position and velocity arrays 
    pos = np.copy(pos_0)
    vel = np.copy(vel_0)
    
    # Ensure arrays are proper numpy arrays with correct shapes
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel)
    mass = np.atleast_1d(mass)
    
    # Initialize simulation data storage
    sim_data = []
    t = 0.0
    
    # Compute initial acceleration and potential
    acc, phi = pyfalcon.gravity(pos, G * mass, eps)
    acc = acc + pot_ext.force(pos)
    
    # Main integration loop
    for i in range(nt + 1):
        # Save snapshot if on downsample boundary
        if i % downsample == 0:
            snapshot = _create_snapshot(t, pos, vel, phi, time_unit)
            sim_data.append(snapshot)
        
        # Check if we've reached the final timestep
        if i == nt:
            # Save final snapshot if requested and not already saved
            if last_snapshot and (i % downsample != 0):
                snapshot = _create_snapshot(t, pos, vel, phi, time_unit)
                sim_data.append(snapshot)
            break
        
        # Leapfrog integration step
        # Kick: update velocities by half timestep
        vel = vel + acc * tau / 2.0
        
        # Drift: update positions by full timestep
        pos = pos + vel * tau
        
        # Compute new forces at updated positions
        acc, phi = pyfalcon.gravity(pos, G * mass, eps)
        acc = acc + pot_ext.force(pos)
        
        # Kick: complete velocity update
        vel = vel + acc * tau / 2.0
        
        # Advance time
        t += tau
    
    return sim_data


def kdk_leapfrog_to_disk(pot_ext, pos_0, vel_0, mass, nt, tau, G, eps, time_unit, downsample,
                          cache_path, last_snapshot=False):
    """
    Kick-Drift-Kick leapfrog integrator that streams snapshots to a binary file.

    Identical physics to kdk_leapfrog, but each snapshot is written straight to
    `cache_path` via SnapshotBinaryWriter as it's produced instead of being
    accumulated in a Python list. Peak memory is therefore one snapshot, not
    the whole run, and a snapshot already flushed to disk survives a kernel
    crash even though later snapshots would be lost.

    Parameters
    ----------
    (same as kdk_leapfrog)
    cache_path : str or Path
        Destination binary file. A JSON sidecar is written alongside it.

    Returns
    -------
    SnapshotBinaryReader
        Lazy, list-like view over the snapshots just written.
    """
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    if nt < 0:
        raise ValueError("nt must be non-negative")

    pos = np.atleast_2d(np.copy(pos_0))
    vel = np.atleast_2d(np.copy(vel_0))
    mass = np.atleast_1d(mass)

    t = 0.0
    acc, phi = pyfalcon.gravity(pos, G * mass, eps)
    acc = acc + pot_ext.force(pos)

    writer = SnapshotBinaryWriter(cache_path, n_particles=len(mass))
    try:
        for i in range(nt + 1):
            if i % downsample == 0:
                writer.append(t * time_unit, pos, vel, phi)

            if i == nt:
                if last_snapshot and (i % downsample != 0):
                    writer.append(t * time_unit, pos, vel, phi)
                break

            vel = vel + acc * tau / 2.0
            pos = pos + vel * tau
            acc, phi = pyfalcon.gravity(pos, G * mass, eps)
            acc = acc + pot_ext.force(pos)
            vel = vel + acc * tau / 2.0
            t += tau
    finally:
        writer.close()

    return SnapshotBinaryReader(cache_path)


def kdk_leapfrog_TD(pot_ext, pos_0, vel_0, mass, nt, tau, G, eps, time_unit, downsample,
                 last_snapshot=True):
    """
    Kick-Drift-Kick leapfrog integrator for time dependent potential 
    
    Parameters
    ----------
    pot_ext : object
        External potential object with a force(positions) method that returns
        accelerations due to external forces.
    pos_0 : array_like, shape (N, 3)
        Initial positions of N particles in 3D space.
    vel_0 : array_like, shape (N, 3)
        Initial velocities of N particles in 3D space.
    mass : array_like, shape (N,)
        Masses of N particles.
    nt : int
        Number of integration timesteps to perform.
    tau : float
        Integration timestep in simulation units.
    G : float
        Gravitational constant in simulation units.
    eps : float
        Softening parameter for gravitational force calculation.
    time_unit : float
        Conversion factor from simulation time units to physical units.
    downsample : int
        Save snapshot data every `downsample` timesteps. Must be >= 1.
    last_snapshot : bool, optional
        If True, ensure the final timestep is saved even if not on a 
        downsample boundary. Default is True.
    
    Returns
    -------
    list of dict
        List of simulation snapshots, where each snapshot is a dictionary with keys:
        - 'time': float, physical time in units specified by time_unit
        - 'pos': ndarray, shape (N, 3), particle positions
        - 'vel': ndarray, shape (N, 3), particle velocities  
        - 'phi': ndarray, shape (N,), gravitational potential at each particle
    
    Notes
    -----
    The leapfrog algorithm follows this sequence:
    1. Kick: v_{i+1/2} = v_i + a_i * dt/2
    2. Drift: x_{i+1} = x_i + v_{i+1/2} * dt
    3. Force: compute a_{i+1} from x_{i+1}
    4. Kick: v_{i+1} = v_{i+1/2} + a_{i+1} * dt/2
    
    This method is symplectic and time-reversible.
    
    """
    # Input validation
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    if nt < 0:
        raise ValueError("nt must be non-negative")

    # position and velocity arrays 
    pos = np.copy(pos_0)
    vel = np.copy(vel_0)
    
    # Ensure arrays are proper numpy arrays with correct shapes
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel)
    mass = np.atleast_1d(mass)
    
    # Initialize simulation data storage
    sim_data = []
    t = 0.0
    
    
    # Compute initial acceleration and potential
    acc, phi = pyfalcon.gravity(pos, G * mass, eps)
    acc = acc + pot_ext.force(pos, t=t)
    
    # Main integration loop
    for i in range(nt + 1):
        # Save snapshot if on downsample boundary
        if i % downsample == 0:
            snapshot = _create_snapshot(t, pos, vel, phi, time_unit)
            sim_data.append(snapshot)
        
        # Check if we've reached the final timestep
        if i == nt:
            # Save final snapshot if requested and not already saved
            if last_snapshot and (i % downsample != 0):
                snapshot = _create_snapshot(t, pos, vel, phi, time_unit)
                sim_data.append(snapshot)
            break
        
        # Leapfrog integration step
        # Kick: update velocities by half timestep
        vel = vel + acc * tau / 2.0
        
        # Drift: update positions by full timestep
        pos = pos + vel * tau
        
        # Compute new forces at updated positions
        acc, phi = pyfalcon.gravity(pos, G * mass, eps)
        acc = acc + pot_ext.force(pos, t=t)
        
        # Kick: complete velocity update
        vel = vel + acc * tau / 2.0
        
        # Advance time
        t += tau
    
    return sim_data


def _create_snapshot(time, pos, vel, phi, time_unit):
    """
    Create a snapshot dictionary with deep copies of the current state.
    
    Parameters
    ----------
    time : float
        Current simulation time
    pos : ndarray
        Current particle positions
    vel : ndarray  
        Current particle velocities
    phi : ndarray
        Current gravitational potential
    time_unit : float
        Time unit conversion factor
        
    Returns
    -------
    dict
        Snapshot dictionary with copied arrays
    """
    return {
        "time": time * time_unit,
        "pos": np.copy(pos),
        "vel": np.copy(vel), 
        "phi": np.copy(phi)
    }