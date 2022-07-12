import treecode
import direct_sum
import numpy as np
import time

import sys
# setting path
sys.path.append('..')
import PyCC.PyCC as PyCC

box = np.array([[-100,100],[-100,100],[-100,100]],dtype=float)

points_df = PyCC.Distributions.Uniform(r=100,n=1000,p=10)
points = points_df.loc[:,["x","y","z"]].to_numpy()
vels = points_df.loc[:,["vx","vy","vz"]].to_numpy()
masses = points_df.loc[:,"mass"].to_numpy()

first = time.perf_counter()
acc,phi = treecode.evaluate(points,vels,masses)
second = time.perf_counter()

print(second-first)

first = time.perf_counter()
part_df, stats_fast = direct_sum.evaluate(points,vels,masses, steps=0,dt = 50)
second = time.perf_counter()
true_acc = part_df.loc[:,["ax","ay","az"]].to_numpy()
true_phi = part_df.loc[:,"phi"].to_numpy()

print(second-first)

print(np.mean(np.abs(true_acc - acc)))
print(np.mean(np.abs(true_phi - phi)))