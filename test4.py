import PyCC
import numpy as np
import matplotlib.pyplot as plt

plummer = PyCC.Distributions.Plummer(100,1,1,1)
#plummer = PyCC.Distributions.Uniform(1,100,1)

nsteps = 100

outf8,stats = PyCC.evaluate(plummer,steps=nsteps,dt=1/64,precision="f4",accelerate=False)
outf4,stats = PyCC.evaluate(plummer,steps=nsteps,dt=1/64,precision="f4",accelerate=True)

ke = []
phis = []
kef4 = []
phisf4 = []
for step in range(nsteps+1):
    vels = outf8[outf8["step"] == step].loc[:,["vx","vy","vz"]].to_numpy()
    speeds = np.linalg.norm(vels,axis=1)
    energies = (1/2) * plummer.loc[:,"mass"].to_numpy()[0] * (speeds**2)
    energy = np.sum(energies)
    ke.append(energy)
    phis.append(np.sum(outf8[outf8["step"] == step].loc[:,"phi"].to_numpy())/2)

    vels = outf4[outf4["step"] == step].loc[:,["vx","vy","vz"]].to_numpy()
    speeds = np.linalg.norm(vels,axis=1)
    energies = (1/2) * plummer.loc[:,"mass"].to_numpy()[0] * (speeds**2)
    energy = np.sum(energies)
    kef4.append(energy)
    phisf4.append(np.sum(outf4[outf4["step"] == step].loc[:,"phi"].to_numpy())/2)

ke = np.array(ke)
phis = np.array(phis)
tot = ke+phis

kef4 = np.array(kef4)
phisf4 = np.array(phisf4)
totf4 = kef4+phisf4

plt.plot(tot,label="f8",alpha=0.5)
plt.plot(totf4,label="f4",alpha=0.5)
plt.legend()
plt.show()