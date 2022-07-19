import PyCC
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import time

df = PyCC.Distributions.Uniform(100,10,1)

ray = PyCC.ray(np.array([1,0,0]),2,25)

out2,stats = PyCC.evaluate(df,steps=10,G=1,precision="f4",accelerate=True)
out,stats = PyCC.evaluate(df,steps=10,G=1,precision="f4")

print(out2)
print(out)

acc2 = (out2.loc[:,["ax","ay","az"]].to_numpy())
acc = (out.loc[:,["ax","ay","az"]].to_numpy())
print(acc2-acc)

acc2 = (out2.loc[:,["x","y","z"]].to_numpy())
acc = (out.loc[:,["x","y","z"]].to_numpy())
print(acc2-acc)

acc2 = (out2.loc[:,["gpe"]].to_numpy())
acc = (out.loc[:,["gpe"]].to_numpy())
print(acc2-acc)