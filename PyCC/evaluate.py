import sys
import os

fpath = os.path.join(os.path.dirname(__file__))
sys.path.append(fpath)

import direct_sum_double
import direct_sum_single
import gpu_single
import gpu_half
import numpy as np

def evaluate(particle_df, steps = 0, eps = 0, G = 1, dt = 1, precision="f8", accelerate=False, gpu_precision = "highp"):
    particles = particle_df.loc[:,["x","y","z"]].to_numpy()
    velocities = particle_df.loc[:,["vx","vy","vz"]].to_numpy()
    masses = particle_df.loc[:,"mass"].to_numpy()

    if accelerate:
        if precision == "f4":
            return gpu_single.evaluate(particles,velocities,masses,steps,eps,G,dt,gpu_precision=gpu_precision)
        if precision == "f2":
            return gpu_half.evaluate(particles,velocities,masses,steps,eps,G,dt,gpu_precision=gpu_precision)
    else:
        if precision == "f8":
            return direct_sum_double.evaluate(particles,velocities,masses,steps,eps,G,dt)
        elif precision == "f4":
            return direct_sum_single.evaluate(particles,velocities,masses,steps,eps,G,dt)

def find_timestep(df,start_timestep,atol,**kwargs):
    outdf,stats = evaluate(df,dt=start_timestep,steps=1,**kwargs)
    out_pos = outdf[outdf["step"] == 1].loc[:,["x","y","z"]].to_numpy()
    
    testdf,stats = evaluate(df,dt=start_timestep/2,steps=2,**kwargs)
    test_pos = testdf[testdf["step"] == 2].loc[:,["x","y","z"]].to_numpy()

    diff = np.max(np.abs(out_pos - test_pos))

    if atol > diff:
        return start_timestep,diff
    else:
        return find_timestep(df,start_timestep/2,atol,**kwargs)
