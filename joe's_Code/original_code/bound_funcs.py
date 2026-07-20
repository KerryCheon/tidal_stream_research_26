import numpy as np
from center_of_mass import get_center_of_mass


def add_bound_data(pos, vel, e, r_cm, mass, use, search_success): 
    """
    Process bound particle data and create dictionary with aggregate properties.
    
    Parameters:
    -----------
    pos : array-like
        Position of each particle
    vel : array-like
        Velocity of each particle
    e : array-like
        Energy per unit mass of each particle
    r_cm : array-like
        Center of mass position
    mass : array-like
        Mass of each particle
    use : array-like of bool
        Boolean mask indicating which particles are bound
    search_success : bool
        Flag indicating if the search converged successfully
        
    Returns:
    --------
    dict : Dictionary containing bound particle properties
    """
    
    # Calculate total mass and energy of bound particles
    m_tot = np.sum(mass[use])
    e_tot = np.sum(e[use] * mass[use])  # Total energy = sum(energy_per_mass * mass)
    
    data = {} 
    
    # Add mass and energy of bound particles to dictionary
    data["total mass"] = m_tot
    data["total energy"] = e_tot 
    # Flag to indicate that search was successful 
    data["search flag"] = search_success
    
    # Extract bound particle positions and velocities
    bound_pos = pos[use, :]
    bound_vel = vel[use, :]
    bound_mass = mass[use]
    
    # Position and velocity using mass-weighted center of mass
    data["pos"] = get_center_of_mass(bound_pos, bound_mass)
    data["vel"] = get_center_of_mass(bound_vel, bound_mass)
    
    # Calculate maximum distance from center of mass
    pos_relative = bound_pos - r_cm 
    r_max = np.max(np.linalg.norm(pos_relative, axis=1)) 
    data["r_bound"] = r_max
    data["bound_id"] = use 
    data["unbound_id"] = [not item for item in use]  
    
    return data

def get_bound_particles(sim_data, mass, r_search=10, tol=0.1/1000, n_search=10):
    """
    Track bound particles throughout a simulation by iteratively finding the center of mass
    of gravitationally bound particles, using an energy criteria, i.e., energy < 0 for 
    for bound particles. 
    
    Parameters:
    -----------
    sim_data : list of dict
        Simulation data where each entry contains 'pos', 'vel', 'phi', 'time'
    mass : array, mass of each particle 
    r_search : float, optional
        Search radius for finding particles (default: 10)
    tol : float, optional  
        Convergence tolerance for center of mass (default: 0.0001)
    n_search : int, optional
        Maximum number of search iterations (default: 10)
        
    Returns:
    --------
    list : bound_data - list of dictionaries with bound particle data
    """
    
    n = len(sim_data)
    Nbody = len(sim_data[0]["pos"])
    
    # Initialize with first timestep
    data = sim_data[0]
    pos_0 = data["pos"]
    vel_0 = data["vel"]
    t1 = data["time"] 
    phi = data["phi"] 
    
    # position and velocity center of mass
    r_cm = get_center_of_mass(pos_0, mass)
    v_cm = get_center_of_mass(vel_0, mass)
    
    # subtract velocity 
    v = vel_0 - v_cm
    v = np.linalg.norm(v, axis=1)
        
    # compute energy per mass 
    e = 0.5 * v**2 + phi
    
    # list to collect data 
    bound_data = []
    # update data dictionary 
    b_data = add_bound_data(pos_0, vel_0, e, r_cm, mass, [True]*Nbody, True)
    bound_data.append(b_data) 
    
    # Process remaining timesteps
    for i in range(1, n): 
        
        data = sim_data[i]
        pos = data["pos"]
        vel = data["vel"] 
        phi = data["phi"] 
        t2 = data["time"] 
        
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
                b_data = add_bound_data(pos, vel, e, r_new, mass, bound_mask, True)
                bound_data.append(b_data)
                r_cm = r_new
                break 
            elif j == n_search - 1: 
                b_data = add_bound_data(pos, vel, e, r_new, mass, bound_mask, False)
                bound_data.append(b_data)
                r_cm = r_new 
            else: 
                r = r_new
                
        t1 = t2 
    
    return bound_data