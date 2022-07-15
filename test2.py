import PyCC
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import time

df = PyCC.Distributions.Uniform(1,100,1)

ray = PyCC.ray(np.array([1,0,0]),2,25)

out,stats = PyCC.evaluate(df,steps=0,G=constants.G,precision="double")
print(stats)
pos_double = out.loc[:,["x","y","z"]].to_numpy()
phi_double = out.loc[:,"phi"].to_numpy()

out2,stats = PyCC.evaluate(df,steps=0,G=constants.G,precision="single")
print(stats)
pos_single = out2.loc[:,["x","y","z"]].to_numpy()
phi_single = out2.loc[:,"phi"].to_numpy()

masses = df.loc[:,"mass"].to_numpy()

pos_temp = np.reshape(out.loc[:,["x","y","z"]].to_numpy(),(1,100,3))
pos2_temp = np.reshape(out2.loc[:,["x","y","z"]].to_numpy(),(1,100,3))

analytic = PyCC.Analytic.Uniform(1,1,ray,G=constants.G)

measured_double = PyCC.analysis.measure_phi_double(pos_temp,masses,np.array([0],dtype=np.int32),ray.loc[:,["x","y","z"]].to_numpy(),G=constants.G,eps=0)
double_phi = np.asarray(measured_double)[0][:,3]

measured_single = PyCC.analysis.measure_phi_single(pos_temp.astype('f4'),masses.astype('f4'),np.array([0],dtype=np.int32),ray.loc[:,["x","y","z"]].to_numpy().astype("f4"),G=constants.G,eps=0)
single_phi = np.asarray(measured_single)[0][:,3]

xs = PyCC.points2radius(ray)
plt.scatter(xs,double_phi-single_phi,label="double")
plt.show()