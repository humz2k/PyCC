import sys
import os

fpath = os.path.join(os.path.dirname(__file__))
sys.path.append(fpath)

import direct_sum_double
import direct_sum_single
import gpu_single
import gpu_half
import numpy as np

def evaluate(particle_df, steps = 0, eps = 0, G = 1, dt = 1, precision="f8", accelerate=False, gpu_precision = "highp",dtatol=0):
    """Does an nbody n**2 simulation on an initial distribution of particles.

    A function that takes in a DataFrame with columns ["x","y","z","vx","vy","vz","mass"], and does a nbody n**2 simulation on it.

    Parameters
    ----------
    particle_df : pandas.DataFrame
        The DataFrame of the initial distribution of particles.
    steps : int
        The number of timesteps to evaluate. Step 0 evaluates the accelerations/GPEs at the initial positions, so steps = 1 will do two evaluations. If unspecified, steps defaults to 0.
    eps : float
        The plummer softening value to use for the simulation. If unspecified, this defaults to 0.
    G : float
        The G constant to use for the simulation. If unspecified, this defaults to 1.
    dt : float
        The timestep to use for the simulation. If unspecified, this defaults to 1.
    precision : str
        The floating point precision used for the simulation. Options: "f8", "f4", "f2". If unspecified, this defaults to "f8".
    accelerate : bool
        Whether to use GPU acceleration. Only available for "f4" and "f2" precision. "f2" precision required accelerate to be true. If unspecified, this defaults to False.
    gpu_precision : str
        If accelerate is True, this is used to specify the precision of floating point operations on the GPU. Options: "highp", "mediump", "lowp". If unspecified, this defaults to "highp".

    Returns
    -------
    pandas.DataFrame
        The DataFrame of the resulting simulation. Has columns ["step","id","x","y","z","vx","vy","vz","ax","ay","az","gpe"].

    """
    particles = particle_df.loc[:,["x","y","z"]].to_numpy()
    velocities = particle_df.loc[:,["vx","vy","vz"]].to_numpy()
    masses = particle_df.loc[:,"mass"].to_numpy()

    if accelerate:
        if precision == "f4":
            return gpu_single.evaluate(particles,velocities,masses,steps,eps,G,dt,gpu_precision=gpu_precision,dtatol=dtatol)
        if precision == "f2":
            return gpu_half.evaluate(particles,velocities,masses,steps,eps,G,dt,gpu_precision=gpu_precision,dtatol=dtatol)
    else:
        if precision == "f8":
            return direct_sum_double.evaluate(particles,velocities,masses,steps,eps,G,dt)
        elif precision == "f4":
            return direct_sum_single.evaluate(particles,velocities,masses,steps,eps,G,dt)

def find_timestep(df,start_timestep,atol,**kwargs):
    """Finds a timestep to use for a simulation with an error smaller than atol.

    A function that takes in a DataFrame with columns ["x","y","z","vx","vy","vz","mass"], and determines a timestep to use such that the error is less than atol.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame of the initial distribution of particles.
    start_timestep : float
        The initial timestep to begin the search.
    atol : float
        The absolute tolerance value for the error of start_timestep.
    
    Other Parameters
    ----------------
    eps : float
        The plummer softening value to use for the simulation. If unspecified, this defaults to 0.
    G : float
        The G constant to use for the simulation. If unspecified, this defaults to 1.
    precision : str
        The floating point precision used for the simulation. Options: "f8", "f4", "f2". If unspecified, this defaults to "f8".
    accelerate : bool
        Whether to use GPU acceleration. Only available for "f4" and "f2" precision. "f2" precision required accelerate to be true. If unspecified, this defaults to False.
    gpu_precision : str
        If accelerate is True, this is used to specify the precision of floating point operations on the GPU. Options: "highp", "mediump", "lowp". If unspecified, this defaults to "highp".


    Returns
    -------
    tuple
        A tuple of (the output timestep, the absolute error of the output timestep)

    """
    outdf,stats = evaluate(df,dt=start_timestep,steps=1,**kwargs)
    out_pos = outdf[outdf["step"] == 1].loc[:,["x","y","z"]].to_numpy()
    
    testdf,stats = evaluate(df,dt=start_timestep/2,steps=2,**kwargs)
    test_pos = testdf[testdf["step"] == 2].loc[:,["x","y","z"]].to_numpy()

    diff = np.max(np.abs(out_pos - test_pos))

    if atol > diff:
        return start_timestep,diff
    else:
        return find_timestep(df,start_timestep/2,atol,**kwargs)
