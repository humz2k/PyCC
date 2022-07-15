import PyCC
import numpy as np


df = PyCC.Distributions.Plummer(1000,1,1,1)

out,stats = PyCC.evaluate(df,steps=0,eps=0,G=1)
print(stats)

print(np.sum(out.loc[:,"phi"].to_numpy())/2)

def e_kin(outs,df):
    steps = np.unique(outs.loc[:,"step"].to_numpy())
    energies = np.zeros((len(steps)),dtype=float)
    for step in steps:
        energies[step] = np.sum(0.5 * df.loc[:,"mass"].to_numpy() * np.linalg.norm(out[out["step"] == step].loc[:,["vx","vy","vz"]].to_numpy(),axis=1)**2)
    return energies

print(e_kin(out,df))