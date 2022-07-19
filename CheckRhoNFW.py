import PyCC
import numpy as np
import matplotlib.pyplot as plt

n = 100000
G = 1
rs = 1
c = 1
ps = 10
Rvir = c*rs

def rho(r,p0,Rs):
    return p0/((r/Rs) * ((1+(r/Rs))**2))

df = PyCC.Distributions.NFW(n,rs,ps,c,100)

mass = df.loc[:,"mass"].to_numpy()[0]

radii = PyCC.points2radius(df)

ys,xs = np.histogram(radii,bins=100)
vols = (4/3) * np.pi * xs[1:]**3 - (4/3) * np.pi * xs[:-1]**3
diff = xs[1] - xs[0]
xs = (diff)/2 + xs[:-1]

rhos = [rho(r,ps,rs) for r in xs]

plt.plot(xs/Rvir,(ys*mass)/vols)
plt.plot(xs/Rvir,rhos)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r"$\log[\frac{r}{R_s}]$")
plt.ylabel(r"$\log[\rho]$")
plt.show()
