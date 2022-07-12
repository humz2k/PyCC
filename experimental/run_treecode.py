import treecode
import direct_sum
import numpy as np
import time
import matplotlib.pyplot as plt

import sys
# setting path
sys.path.append('..')
import PyCCOld.PyCC as PyCC

tree_time = []
tree_time_n2 = []
sum_time = []
ns = list(range(100,5000,1000))
errors = []
for n in ns:
    points_df = PyCC.Distributions.Uniform(r=100,n=n,p=10)
    points = points_df.loc[:,["x","y","z"]].to_numpy()
    vels = points_df.loc[:,["vx","vy","vz"]].to_numpy()
    masses = points_df.loc[:,"mass"].to_numpy()

    first = time.perf_counter()
    acc,phi,truncs = treecode.evaluate(points,vels,masses,theta=1)
    second = time.perf_counter()

    tree_time.append(second-first)

    #first = time.perf_counter()
    #acc2,phi2,truncs = treecode.evaluate(points,vels,masses,theta=0)
    #second = time.perf_counter()

    #tree_time_n2.append(second-first)

    first = time.perf_counter()
    part_df, stats_fast = direct_sum.evaluate(points,vels,masses, steps=0,dt = 50)
    second = time.perf_counter()
    true_acc = part_df.loc[:,["ax","ay","az"]].to_numpy()
    true_phi = part_df.loc[:,"phi"].to_numpy()

    sum_time.append(second-first)

    print("A")
    errors.append(np.mean(np.abs(true_phi - phi)/np.abs(true_phi)))
    #print(np.mean(np.abs(true_acc - acc2)))

plt.plot(ns,tree_time,label="tree")
#plt.plot(ns,tree_time_n2,label="treen2")
plt.plot(ns,sum_time,label="sum")
plt.legend()
plt.show()

plt.scatter(ns,errors)
plt.legend()
plt.show()
