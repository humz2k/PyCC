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

particle_mass = uniform.loc[:,"mass"][0]

particle_gpes = outf4gpu.loc[:,"gpe"].to_numpy()
particle_phis = particle_gpes/particle_mass

particle_rs = PyCC.points2radius(outf4gpu)

particles = outf4gpu.loc[:,["x","y","z"]].to_numpy().reshape(nsteps+1,n,3)
masses = uniform.loc[:,"mass"].to_numpy()
pos = ray.loc[:,["x","y","z"]].to_numpy()

out,stats = gpu_analysis.evaluate(particles,masses,[0],pos,eps=0,G=G)

analytics = PyCC.Analytic.Uniform(r,p,ray,G)
summed = out.loc[:,"phi"].to_numpy()

xs = PyCC.points2radius(ray)/r

plt.title("Uniform Distribution\n" + r"$n=" + str(n) + r"$")
plt.plot(xs,analytics,color="red",zorder=2,alpha=0.5,label="Analytic")
plt.scatter(xs,summed,label=r"$\phi$ along ray",alpha=0.8,zorder=1)
plt.scatter(particle_rs,particle_phis,label=r"$\phi$ at particle",s=1,alpha=0.5,zorder=0)
plt.legend()
plt.xlabel(r"$\frac{r}{R}$")
plt.ylabel(r"$\phi$")
plt.show()

