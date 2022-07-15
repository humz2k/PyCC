import direct_sum
import numpy as np
import pandas as pd
import scipy.spatial as spatial
import astropy.constants as const
import time

import sys
# setting path
sys.path.append('..')
import PyCCOld.PyCC as PyCC

df = PyCC.Distributions.Uniform(10,1000,10)
pos = df.loc[:,["x","y","z"]].to_numpy()
masses = df.loc[:,["mass"]].to_numpy().flatten()
vels = np.zeros_like(pos,dtype=float)

first = time.perf_counter()
part_df, stats_fast = direct_sum.evaluate(pos,vels,masses, steps=0,dt = 50)
second = time.perf_counter()
print(second-first,stats_fast)

'''
first = time.perf_counter()
out_fast = np.asarray(direct_sum.phi_acc(pos,masses,pos,const.G.value,10))
acc_fast = out_fast[:,[0,1,2]]
phi_fast = out_fast[:,3]
second = time.perf_counter()
out_fast = np.asarray(out_fast)
print(second-first)

first = time.perf_counter()
acc,phi = DirectSum.acc_func(pos,pos,masses,10)
second = time.perf_counter()
print(second-first)

#print(np.mean(np.abs(acc-acc_fast)))
print(np.mean(np.abs(phi-phi_fast)))
'''
