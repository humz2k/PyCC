import PyCC
import numpy as np
import matplotlib.pyplot as plt

dt = 1/64

df = PyCC.Distributions.Plummer(100,1,1,1)

ray = PyCC.ray(np.array([1,0,0]),2,25)

outdf_gpu_low,gpu_low_stats = PyCC.evaluate(df,steps=10,G=1,dt=dt,precision="f4",accelerate=True,gpu_precision="lowp")
outdf_gpu_high,gpu_high_stats = PyCC.evaluate(df,steps=10,G=1,dt=dt,precision="f4",accelerate=True,gpu_precision="highp")

print(gpu_low_stats)
print(gpu_high_stats)

out_gpu_low = PyCC.outdf2numpy(outdf_gpu_low)
out_gpu_high = PyCC.outdf2numpy(outdf_gpu_high)

acc_diff = out_gpu_high["acc"] - out_gpu_low["acc"]
acc_diff = np.sum(acc_diff,axis=2)

plt.plot(acc_diff,color="blue",alpha=0.5)
plt.xlabel("Timestep")
plt.show()