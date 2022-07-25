import PyCC
import numpy as np
import matplotlib.pyplot as plt

#df = PyCC.Distributions.Plummer(100)
#out,stats = PyCC.evaluate.evaluate(df,steps=2,dt=1/64,precision="f4",accelerate=True,dtatol=1e-5)
n_particles = 10000
dtatol = 1e-5
start_dt = 1/64

nsteps = 100

sim = PyCC.Sim(PyCC.Halo.PLUMMER,n_particles,G=1)

print(sim.evaluate(downsample=1,precision="f4",accelerate=True,steps=nsteps,dt=start_dt,dtatol=dtatol,G=1)["stats"])
print(sim.evaluate(downsample=2,precision="f4",accelerate=True,steps=nsteps,dt=start_dt,dtatol=dtatol,G=1)["stats"])
print(sim.evaluate(downsample=3,precision="f4",accelerate=True,steps=nsteps,dt=start_dt,dtatol=dtatol,G=1)["stats"])
print(sim.evaluate(downsample=4,precision="f4",accelerate=True,steps=nsteps,dt=start_dt,dtatol=dtatol,G=1)["stats"])
print(sim.evaluate(downsample=5,precision="f4",accelerate=True,steps=nsteps,dt=start_dt,dtatol=dtatol,G=1)["stats"])

out = sim.energies(downsample=1)
out2 = sim.energies(downsample=2)
out3 = sim.energies(downsample=3)
out4 = sim.energies(downsample=4)
out5 = sim.energies(downsample=5)

dts = sim.find_runs(downsample=1)[0]["df"].loc[:,"dt"][::n_particles].to_numpy()
print(dts)
dts2 = sim.find_runs(downsample=2)[0]["df"].loc[:,"dt"][::n_particles].to_numpy()
print(dts2)
dts3 = sim.find_runs(downsample=3)[0]["df"].loc[:,"dt"][::n_particles].to_numpy()
print(dts3)

plt.plot(out["total"]/out["total"][0],label="n=" + str(n_particles))
plt.plot(out2["total"]/out2["total"][0],label="n=" + str(int(n_particles/2)))
plt.plot(out3["total"]/out3["total"][0],label="n=" + str(int(n_particles/3)))
plt.plot(out4["total"]/out4["total"][0],label="n=" + str(int(n_particles/4)))
plt.plot(out5["total"]/out5["total"][0],label="n=" + str(int(n_particles/5)))

plt.ticklabel_format(useOffset=False)
plt.legend()
plt.tight_layout()
plt.show()

