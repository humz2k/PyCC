import PyCC
import numpy as np
import matplotlib.pyplot as plt

n = 1000000
G = 1
rs = 1
c = 1
ps = 1
Rvir = c*rs
nsteps = 1

df = PyCC.Distributions.NFW(n,rs,ps,c,100,1)
particle_mass = df.loc[:,"mass"][0]

outdf,stats = PyCC.evaluate(df,nsteps,0,1,1/256,precision="f4",accelerate=True)
print(stats)

out = PyCC.outdf2numpy(outdf)

vels = out["vel"]
speeds = np.linalg.norm(vels,axis=2)
kes = (0.5) * particle_mass * (speeds**2)
ke_tot = np.sum(kes,axis=1)

gpes = out["gpe"]
gpe_tot = np.sum(gpes,axis=1).flatten()/2

tot = ke_tot + gpe_tot

plt.plot(tot)
plt.show()