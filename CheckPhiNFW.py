import PyCC
import numpy as np
import matplotlib.pyplot as plt
import PyCC.gpu_analysis as gpu_analysis

#plummer = PyCC.Distributions.Plummer(1000,1,1,1)
n = 100
G = 1
rs = 1
c = 10
ps = 10
Rvir = rs * c

df = PyCC.Distributions.NFW(rs,ps,c,200,n)

ray = PyCC.ray(np.array([1,0,0]),Rvir*10,50)

outf8,stats = PyCC.evaluate(df,steps=0,dt=1,G=G,precision="f8",accelerate=False)

particles = outf8.loc[:,["x","y","z"]].to_numpy().reshape(1,n,3)
masses = df.loc[:,"mass"].to_numpy()
pos = ray.loc[:,["x","y","z"]].to_numpy()

particle_mass = df.loc[:,"mass"][0]

particle_rs = PyCC.points2radius(outf8)
particle_phi = outf8.loc[:,"gpe"].to_numpy()/particle_mass

out,stats = gpu_analysis.evaluate(particles,masses,[0],pos)

analytics = PyCC.Analytic.NFW(rs,ps,ray,G)
summed = out.loc[:,"phi"].to_numpy()

plt.plot(PyCC.points2radius(ray)/Rvir,analytics,color="red",label="Analytic")
plt.scatter(PyCC.points2radius(ray)/Rvir,summed,label=r"$\phi along ray$")
#plt.scatter(particle_rs/Rvir,particle_phi)
plt.xlabel(r"$\frac{r}{R_{vir}}$")
plt.ylabel(r"$\phi$")
plt.show()

