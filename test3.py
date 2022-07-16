import PyCC
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import time

df = PyCC.Distributions.Uniform(100,1000000,1)

ray = PyCC.ray(np.array([1,0,0]),2,25)

#out,stats = PyCC.evaluate(df,steps=5,G=constants.G,precision="f8",accelerate=False)

#print(stats["eval_time"])

out,stats = PyCC.evaluate(df,steps=0,G=constants.G,precision="f4",accelerate=True)

print(stats["eval_time"])

'''

nogpu = out.loc[:,["x","y","z"]].to_numpy().astype("f4")
yesgpu = out2.loc[:,["x","y","z"]].to_numpy().astype("f4")

for i in range(len(nogpu)):
    print(nogpu[i])
    print(yesgpu[i])
    print("")

nogpu = out.loc[:,["x","y","z","vx","vy","vz"]].to_numpy().astype("f4")
yesgpu = out2.loc[:,["x","y","z","vx","vy","vz"]].to_numpy().astype("f4")
diff = nogpu - yesgpu
print(np.sum(diff,axis=0))
'''