import sys
import os

fpath = os.path.join(os.path.dirname(__file__))
sys.path.append(fpath)

import direct_sum_double
import direct_sum_single
import gpu_single
import gpu_half

def evaluate(particle_df, steps = 0, eps = 0, G = 1, dt = 1, precision="f8", accelerate=False):
    particles = particle_df.loc[:,["x","y","z"]].to_numpy()
    velocities = particle_df.loc[:,["vx","vy","vz"]].to_numpy()
    masses = particle_df.loc[:,"mass"].to_numpy()

    if accelerate:
        if precision == "f4":
            return gpu_single.evaluate(particles,velocities,masses,steps,eps,G,dt)
        if precision == "f2":
            return gpu_half.evaluate(particles,velocities,masses,steps,eps,G,dt)
    else:
        if precision == "f8":
            return direct_sum_double.evaluate(particles,velocities,masses,steps,eps,G,dt)
        elif precision == "f4":
            return direct_sum_single.evaluate(particles,velocities,masses,steps,eps,G,dt)

