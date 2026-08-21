import numpy as np

def get_center_of_mass(x,  mass):
    """
    Compute position and velocity center of mass for particles.
    
    Parameters:
    -----------
    x: array-like, shape (N, 3)
        Position or velocity  vectors of N particles
    
    mass : array-like, shape (N,)
        Masses of N particles
        
    Returns:
    --------
    tuple : x_cm
        x_cm : array, shape (3,)
            Position or velocity  center of mass
   
    """
    
    # Check if all masses are equal (within numerical tolerance)
    if np.allclose(mass, mass[0], rtol=1e-10):
        # All masses equal - use simple mean
        x_cm = np.mean(x, axis=0)
       
    else:
        # Different masses - use weighted average
        total_mass = np.sum(mass)
        x_cm = np.sum(mass[:, np.newaxis] * x, axis=0) / total_mass
       
    
    return x_cm