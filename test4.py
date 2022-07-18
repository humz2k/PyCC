import PyCC
import numpy as np
import matplotlib.pyplot as plt
import PyCC.gpu_analysis as gpu_analysis

#plummer = PyCC.Distributions.Plummer(1000,1,1,1)
n = 1000
G = 1
rs = 1
c = 1
ps = 10
Rvir = rs * c

df = PyCC.Distributions.NFW(rs,ps,c,n)

ray = PyCC.ray(np.array([1,0,0]),Rvir,25)

nsteps = 0

outf4gpu,stats = PyCC.evaluate(df,steps=nsteps,dt=1,G=G,precision="f8",accelerate=False)

phi = outf4gpu.loc[:,["phi"]].to_numpy().flatten()

particles = outf4gpu.loc[:,["x","y","z"]].to_numpy().reshape(nsteps+1,n,3)
masses = df.loc[:,"mass"].to_numpy()
pos = ray.loc[:,["x","y","z"]].to_numpy()

out,stats = gpu_analysis.evaluate(particles,masses,[0],pos)

analytics = PyCC.Analytic.NFW(rs,ps,ray,G)
summed = out.loc[:,"phi"].to_numpy()

#plt.plot(PyCC.points2radius(ray)/Rvir,analytics)
plt.scatter(PyCC.points2radius(ray)/Rvir,summed-analytics)
#plt.scatter(PyCC.points2radius(df)/Rvir,phi/masses[0])
plt.xlabel(r"$\frac{r}{R_{vir}}$")
plt.ylabel(r"$\phi$")
plt.show()

