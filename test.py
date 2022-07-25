import PyCC
import numpy as np
import matplotlib.pyplot as plt

#df = PyCC.Distributions.Plummer(100)
#out,stats = PyCC.evaluate.evaluate(df,steps=2,dt=1/64,precision="f4",accelerate=True,dtatol=1e-5)
n_particles = 1000
dtatol = 1e-5
start_dt = 1/64

nsteps = 100

sim = PyCC.Sim(PyCC.Halo.PLUMMER,n_particles)

sim.evaluate(precision="f2",accelerate=True,steps=nsteps,dt=start_dt,dtatol=dtatol)
sim.evaluate(precision="f4",accelerate=True,steps=nsteps,dt=start_dt,dtatol=dtatol)

out = sim.energies(precision="f2",dtatol=dtatol)
out3 = sim.energies(precision="f4")

dts = sim.find_runs(precision="f2",dtatol=dtatol)[0]["df"].loc[:,"dt"][::n_particles].to_numpy()
dts3 = sim.find_runs(precision="f4",dtatol=dtatol)[0]["df"].loc[:,"dt"][::n_particles].to_numpy()

print(dts)
print(dts3)

'''
xs = []
xs3 = []
for i in range(nsteps+1):
    xs.append(np.sum(dts[:i]))
    xs3.append(np.sum(dts3[:i]))
'''

plt.plot(out["total"] - out["total"][0],label="timecontrol f2")
plt.plot(out3["total"] - out3["total"][0],label="timecontrol f4")
plt.ticklabel_format(useOffset=False)
plt.legend()
plt.tight_layout()
plt.show()

