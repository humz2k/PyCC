import evaluate
import utilities
from enum import Enum
import json
import numpy as np

class Halo(Enum):
    UNIFORM = 0
    PLUMMER = 1
    NFW = 2

class Sim:
    def __init__(self,halo,n,**kwargs):

        HaloTypes = {Halo.UNIFORM:{"sampler":utilities.Distributions.Uniform,"analytic":utilities.Analytic.Uniform},
                    Halo.PLUMMER:{"sampler":utilities.Distributions.Plummer,"analytic":utilities.Analytic.Plummer},
                    Halo.NFW:{"sampler":utilities.Distributions.NFW,"analytic":utilities.Analytic.NFW}}

        sampler = HaloTypes[halo]["sampler"]
        analytic = HaloTypes[halo]["analytic"]
        self.kwargs = kwargs
        self.df = sampler(n,**kwargs)
        self.kwargs["n"] = n
        self.out = {}
    
    def evaluate(self,downsample=1,**kwargs):
        eval_df = self.df
        if downsample > 1:
            eval_df = utilities.downsample(self.df,downsample)
        key = {"downsample":downsample}
        key.update(kwargs)
        key = json.dumps(key,sort_keys=True)
        df,stats = evaluate.evaluate(eval_df,**kwargs)
        self.out[key] = {"df":df,"init":eval_df,"stats":stats}
        return self.out[key]
    
    def find_runs(self,**kwargs):
        key = json.dumps(kwargs,sort_keys=True)
        out = []
        for i in self.out.keys():
            all_keys = key[1:-1].split(",")
            inside = True
            for temp in all_keys:
                if not temp in i:
                    inside = False
                    break
            if inside:
                out.append(self.out[i])
        return tuple(out)

    def energies(self,run_number=0,**kwargs):
        run = self.find_runs(**kwargs)[run_number]

        out = utilities.outdf2numpy(run["df"])

        particle_mass = run["init"].iloc[0]["mass"]

        vels = out["vel"]
        speeds = np.linalg.norm(vels,axis=2)
        kes = (0.5) * particle_mass * (speeds**2)
        ke_tot = np.sum(kes,axis=1)

        gpes = out["gpe"]
        gpe_tot = np.sum(gpes,axis=1).flatten()/2

        tot = ke_tot + gpe_tot

        if len(tot) == 1:
            gpe_tot = gpe_tot[0]
            ke_tot = ke_tot[0]
            tot = tot[0]

        return {"gpe":gpe_tot,"ke":ke_tot,"total":tot}


'''
import matplotlib.pyplot as plt

sim = Sim(halo=Halo.PLUMMER,n=10000)

downsamples = [1,2,3,4]
for downsample in downsamples:
    sim.evaluate(steps=10,precision="f4",accelerate=True,downsample=downsample,G=1,dt=1/64)
#sim.evaluate(steps=10,precision="f4",accelerate=True,downsample=2,G=1,dt=1/64)

for downsample in downsamples:
    plt.plot(sim.energies(downsample=downsample)["total"],label="n="+str(int(sim.kwargs["n"] * 1/downsample)))
plt.legend()
plt.show()
'''