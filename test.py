import PyCC
import numpy as np
import matplotlib.pyplot as plt

r = 10000
p = 0.1
ray_steps = 25

x = list(PyCC.ray_rs(r*2,ray_steps))
xs = []
ys = []

rays = []
for vector in PyCC.random_vectors(50):
    rays.append(PyCC.ray(vector,r*2,ray_steps))

for i in range(2):
    df = PyCC.Distributions.Uniform(r=r,n=1000,p=p)
    ray_analytics = PyCC.Analytic.Uniform(r=r,p=p,positions=rays[0])

    for ray in rays:
        out,ray_out,stats = PyCC.evaluate(df,evaluate_at=ray,steps=0,precision="double")
        ray_phis = ray_out.loc[:,"phi"].to_numpy()
        xs += x
        ys += list(ray_phis - ray_analytics)

plt.hist2d(xs,ys,bins=50)
plt.show()