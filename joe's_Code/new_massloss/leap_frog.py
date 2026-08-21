import numpy as np
import pyfalcon

def _num_snapshots(nt, downsample, last_snapshot):
    n = nt // downsample + 1
    if last_snapshot and (nt % downsample != 0):
        n += 1
    return n


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
    dict
        Dictionary of simulation output with keys:
        - 'time': ndarray, shape (n_snap,), physical time in units specified by time_unit
        - 'pos': ndarray, shape (n_snap, N, 3), particle positions
        - 'vel': ndarray, shape (n_snap, N, 3), particle velocities  
        - 'phi': ndarray, shape (n_snap, N), gravitational potential at each particle
        - 'mass': ndarray, shape (N,), particle masses
        - 'tau': float, integration timestep in simulation units
        - 'eps': float, softening length used for gravity calculation
        - 'N': int, number of particles
        - 'time_span': float, total simulated time in physical units (nt * tau * time_unit)
    
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

    N = pos.shape[0]

    # Preallocate output arrays
    n_snap = _num_snapshots(nt, downsample, last_snapshot)
    times = np.empty(n_snap)
    pos_arr = np.empty((n_snap, N, 3))
    vel_arr = np.empty((n_snap, N, 3))
    phi_arr = np.empty((n_snap, N))
    j = 0

    t = 0.0
    
    # Compute initial acceleration and potential
    acc, phi = pyfalcon.gravity(pos, G * mass, eps)
    acc = acc + pot_ext.force(pos)
    
    # Main integration loop
    for i in range(nt + 1):
        # Save snapshot if on downsample boundary
        if i % downsample == 0:
            times[j] = t * time_unit
            pos_arr[j] = pos
            vel_arr[j] = vel
            phi_arr[j] = phi
            j += 1
        
        # Check if we've reached the final timestep
        if i == nt:
            # Save final snapshot if requested and not already saved
            if last_snapshot and (i % downsample != 0):
                times[j] = t * time_unit
                pos_arr[j] = pos
                vel_arr[j] = vel
                phi_arr[j] = phi
                j += 1
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
    
    return {
        "time": times,
        "pos": pos_arr,
        "vel": vel_arr,
        "phi": phi_arr,
        "mass": mass,
        "tau": tau,
        "eps": eps,
        "N": N,
        "time_span": nt * tau * time_unit,
    }

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
    dict
        Dictionary of simulation output with keys:
        - 'time': ndarray, shape (n_snap,), physical time in units specified by time_unit
        - 'pos': ndarray, shape (n_snap, N, 3), particle positions
        - 'vel': ndarray, shape (n_snap, N, 3), particle velocities  
        - 'phi': ndarray, shape (n_snap, N), gravitational potential at each particle
        - 'mass': ndarray, shape (N,), particle masses
        - 'tau': float, integration timestep in simulation units
        - 'eps': float, softening length used for gravity calculation
        - 'N': int, number of particles
        - 'time_span': float, total simulated time in physical units (nt * tau * time_unit)
    
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

    N = pos.shape[0]

    # Preallocate output arrays
    n_snap = _num_snapshots(nt, downsample, last_snapshot)
    times = np.empty(n_snap)
    pos_arr = np.empty((n_snap, N, 3))
    vel_arr = np.empty((n_snap, N, 3))
    phi_arr = np.empty((n_snap, N))
    j = 0

    t = 0.0
    
    
    # Compute initial acceleration and potential
    acc, phi = pyfalcon.gravity(pos, G * mass, eps)
    acc = acc + pot_ext.force(pos, t=t)
    
    # Main integration loop
    for i in range(nt + 1):
        # Save snapshot if on downsample boundary
        if i % downsample == 0:
            times[j] = t * time_unit
            pos_arr[j] = pos
            vel_arr[j] = vel
            phi_arr[j] = phi
            j += 1
        
        # Check if we've reached the final timestep
        if i == nt:
            # Save final snapshot if requested and not already saved
            if last_snapshot and (i % downsample != 0):
                times[j] = t * time_unit
                pos_arr[j] = pos
                vel_arr[j] = vel
                phi_arr[j] = phi
                j += 1
            break
        
        # Leapfrog integration step
        # Kick: update velocities by half timestep
        vel = vel + acc * tau / 2.0
        
        # Drift: update positions by full timestep
        pos = pos + vel * tau

        # update time for new potential calculation 
        t += tau
        
        # Compute new forces at updated positions
        acc, phi = pyfalcon.gravity(pos, G * mass, eps)
        acc = acc + pot_ext.force(pos, t=t)
        
        # Kick: complete velocity update
        vel = vel + acc * tau / 2.0
        
    
    return {
        "time": times,
        "pos": pos_arr,
        "vel": vel_arr,
        "phi": phi_arr,
        "mass": mass,
        "tau": tau,
        "eps": eps,
        "N": N,
        "time_span": nt * tau * time_unit,
    }