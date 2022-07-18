import PyCC
import numpy as np
import matplotlib.pyplot as plt
import PyCC.gpu_analysis as gpu_analysis

n = 1000
G = 1
r = 1
p = 10

uniform = PyCC.Distributions.Uniform(n,r,p)

ray = PyCC.ray(np.array([1,0,0]),2*r,25)

nsteps = 0

outf4gpu,stats = PyCC.evaluate(uniform,steps=nsteps,dt=1,G=G,precision="f8",accelerate=False)

phi = outf4gpu.loc[:,["phi"]].to_numpy().flatten()

particles = outf4gpu.loc[:,["x","y","z"]].to_numpy().reshape(nsteps+1,n,3)
masses = uniform.loc[:,"mass"].to_numpy()
pos = ray.loc[:,["x","y","z"]].to_numpy()

out,stats = gpu_analysis.evaluate(particles,masses,[0],pos,eps=0,G=G)

analytics = PyCC.Analytic.Uniform(r,p,ray,G)
summed = out.loc[:,"phi"].to_numpy()

plt.plot(PyCC.points2radius(ray),analytics)
plt.scatter(PyCC.points2radius(ray),summed)
#plt.scatter(PyCC.points2radius(outf4gpu),phi)
plt.show()

