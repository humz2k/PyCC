import sys
import os

fpath = os.path.join(os.path.dirname(__file__))
sys.path.append(fpath)

import direct_sum_double
import direct_sum_single

def evaluate(particle_df, steps = 0, eps = 0, G = 1, dt = 1,precision="double"):
    particles = particle_df.loc[:,["x","y","z"]].to_numpy()
    velocities = particle_df.loc[:,["vx","vy","vz"]].to_numpy()
    masses = particle_df.loc[:,"mass"].to_numpy()

    if precision == "double":
        return direct_sum_double.evaluate(particles,velocities,masses,steps,eps,G,dt)
    elif precision == "single":
        return direct_sum_single.evaluate(particles,velocities,masses,steps,eps,G,dt)

