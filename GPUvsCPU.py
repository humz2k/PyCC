import PyCC
import numpy as np
import matplotlib.pyplot as plt

dt = 1/64
nsteps = 100

df = PyCC.Distributions.Plummer(100,1,1,1)

ray = PyCC.ray(np.array([1,0,0]),2,25)

outdf_gpu,gpu_stats = PyCC.evaluate(df,steps=nsteps,G=1,dt=dt,precision="f4",accelerate=True,gpu_precision="lowp")
outdf_cpu,cpu_stats = PyCC.evaluate(df,steps=nsteps,G=1,dt=dt,precision="f4")

print(gpu_stats)
print(cpu_stats)

out_gpu = PyCC.outdf2numpy(outdf_gpu)
out_cpu = PyCC.outdf2numpy(outdf_cpu)

acc_diff = out_cpu["acc"] - out_gpu["acc"]
acc_diff = np.sum(acc_diff,axis=2)

plt.plot(acc_diff,color="blue",alpha=0.5)
plt.xlabel("Timestep")
plt.ylabel(r"$\vec{a}_{cpuf4} - \vec{a}_{gpuf4}$")
plt.tight_layout()
plt.show()