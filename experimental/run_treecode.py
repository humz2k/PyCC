import treecode
import numpy as np
import time

import sys
# setting path
sys.path.append('..')
import PyCC.PyCC as PyCC

box = np.array([[-100,100],[-100,100],[-100,100]],dtype=float)

points_df = PyCC.Distributions.Uniform(r=100,n=10000,p=10)
points = points_df.loc[:,["x","y","z"]].to_numpy()
masses = points_df.loc[:,"mass"].to_numpy()

first = time.perf_counter()
boxes = treecode.divide_box(box)
points,masses = treecode.separate_particles(boxes,points,masses)
second = time.perf_counter()

treecode.build_tree()

print(second-first)