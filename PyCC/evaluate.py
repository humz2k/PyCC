from scipy import constants
import sys
import os

fpath = os.path.join(os.path.dirname(__file__))
sys.path.append(fpath)

import direct_sum_double
import direct_sum_single

def evaluate(particle_df,evaluate_at=None, steps = 0, eps = 0, G = constants.G, dt = 1000,precision="double"):
    particles = particle_df.loc[:,["x","y","z"]].to_numpy()
    velocities = particle_df.loc[:,["vx","vy","vz"]].to_numpy()
    masses = particle_df.loc[:,"mass"].to_numpy()

    eval_pos = None
    if type(evaluate_at) != type(None):
        eval_pos = evaluate_at.loc[:,["x","y","z"]].to_numpy()

    if precision == "double":
        return direct_sum_double.evaluate(particles,velocities,masses,eval_pos,steps,eps,G,dt)
    elif precision == "single":
        return direct_sum_single.evaluate(particles,velocities,masses,eval_pos,steps,eps,G,dt)

