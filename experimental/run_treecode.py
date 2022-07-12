import treecode
import direct_sum
import numpy as np
import time
import matplotlib.pyplot as plt

import sys
# setting path
sys.path.append('..')
import PyCC.PyCC as PyCC

box = np.array([[-100,100],[-100,100],[-100,100]],dtype=float)

tree_time = []
sum_time = []
ns = list(range(10,10000,1000))
for n in ns:
    print(n)
    points_df = PyCC.Distributions.Uniform(r=100,n=n,p=10)
    points = points_df.loc[:,["x","y","z"]].to_numpy()
    vels = points_df.loc[:,["vx","vy","vz"]].to_numpy()
    masses = points_df.loc[:,"mass"].to_numpy()

    first = time.perf_counter()
    acc,phi,truncs = treecode.evaluate(points,vels,masses,theta=10)
    second = time.perf_counter()

    print(truncs)

    tree_time.append(second-first)

    #first = time.perf_counter()
    #part_df, stats_fast = direct_sum.evaluate(points,vels,masses, steps=0,dt = 50)
    #second = time.perf_counter()
    #true_acc = part_df.loc[:,["ax","ay","az"]].to_numpy()
    #true_phi = part_df.loc[:,"phi"].to_numpy()

    #sum_time.append(second-first)

    #print(np.mean(np.abs(true_acc - acc)))

plt.plot(ns,tree_time,label="tree")
#plt.plot(ns,sum_time,label="sum")
plt.legend()
plt.show()
