import PyCC
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import time

df = PyCC.Distributions.Uniform(1,5,1)

ray = PyCC.ray(np.array([1,0,0]),2,25)

out2,stats = PyCC.evaluate(df,steps=3,G=1,precision="f4",accelerate=True)
out,stats = PyCC.evaluate(df,steps=3,G=1,precision="f4")

acc2 = (out2[out2["step"]==0].loc[:,["ax","ay","az"]].to_numpy())
acc = (out[out["step"]==0].loc[:,["ax","ay","az"]].to_numpy())

print(acc2-acc)