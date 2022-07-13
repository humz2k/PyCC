import numpy as np
import pandas as pd
from scipy import spatial
from scipy import constants
from scipy.special import lambertw

class Distributions(object):
    @staticmethod
    def Uniform(r,n,p,file=None):
        phi = np.random.uniform(low=0,high=2*np.pi,size=n)
        theta = np.arccos(np.random.uniform(low=-1,high=1,size=n))
        particle_r = r * ((np.random.uniform(low=0,high=1,size=n))**(1/3))
        x = particle_r * np.sin(theta) * np.cos(phi)
        y = particle_r * np.sin(theta) * np.sin(phi)
        z = particle_r * np.cos(theta)
        vol = (4/3) * np.pi * (r ** 3)
        particle_mass = (p * vol)/n
        particles = np.column_stack([x,y,z])
        velocities = np.zeros_like(particles,dtype=float)
        masses = pd.DataFrame(np.full((1,n),particle_mass).T,columns=["mass"])
        particles = pd.DataFrame(particles,columns=["x","y","z"])
        velocities = pd.DataFrame(velocities,columns=["vx","vy","vz"])
        df = pd.concat((particles,velocities,masses),axis=1)
        if file != None:
            df.to_csv(file,index=False)
        return df

    @staticmethod
    def NFW(Rvir,c,p0,n,file=None):
        Rs = Rvir/c
        def mass(r,Rs=Rs,p0=p0):
            return 4 * np.pi * p0 * (Rs**3) * (np.log((Rs+r)/Rs) + (Rs/(Rs + r)) - 1)
        def cdf(r):
            return (np.log((Rs+r)/Rs) + (Rs/(Rs + r)) - 1)/(np.log((Rs+Rvir)/Rs) + (Rs/(Rs + Rvir)) - 1)
        maxMass = mass(Rvir,Rs,p0)
        def inverse_cdf(p):
            y = p*(np.log((Rs+Rvir)/Rs) + (Rs/(Rs + Rvir)) - 1)
            W = lambertw((-1)/(np.exp(y+1)))
            return float((-Rs/(W)) - Rs)
        radiuses = np.zeros((n))
        input_p = np.random.uniform(0,1,n)
        for i in range(n):
            radiuses[i] = inverse_cdf(input_p[i])
        phi = np.random.uniform(low=0,high=2*np.pi,size=n)
        theta = np.arccos(np.random.uniform(low=-1,high=1,size=n))
        x = radiuses * np.sin(theta) * np.cos(phi)
        y = radiuses * np.sin(theta) * np.sin(phi)
        z = radiuses * np.cos(theta)
        particle_mass = maxMass/n
        particles = np.column_stack([x,y,z])
        velocities = np.zeros_like(particles,dtype=float)
        masses = pd.DataFrame(np.full((1,n),particle_mass).T,columns=["mass"])
        particles = pd.DataFrame(particles,columns=["x","y","z"])
        velocities = pd.DataFrame(velocities,columns=["vx","vy","vz"])
        df = pd.concat((particles,velocities,masses),axis=1)
        if file != None:
            df.to_csv(file,index=False)
        return df

class Analytic(object):
    @staticmethod
    def Uniform(r,p,positions):
        positions = positions.loc[:,["x","y","z"]].to_numpy()
        def phi(r,p,pos):
            pos_r = spatial.distance.cdist(np.array([[0,0,0]]),np.reshape(pos,(1,)+pos.shape)).flatten()[0]
            relative = pos_r/r
            if relative == 1:
                return (-4/3) * np.pi * constants.G * p * (r ** 2)
            elif relative < 1:
                return (-2) * np.pi * constants.G * p * ((r ** 2) - ((1/3) * ((pos_r)**2)))
            else:
                return (-4/3) * np.pi * constants.G * p * ((r ** 3)/(pos_r))
        out = np.zeros((len(positions)),dtype=float)
        for idx,pos in enumerate(positions):
            out[idx] = phi(r,p,pos)
        return out

    @staticmethod
    def NFW(Rvir,c,p0,positions):
        positions = positions.loc[:,["x","y","z"]].to_numpy()
        Rs = Rvir/c
        def phi(Rs,p0,pos):
            pos_r = spatial.distance.cdist(np.array([[0,0,0]]),np.reshape(pos,(1,)+pos.shape)).flatten()[0]
            return ((-4 * np.pi * constants.G * p0 * (Rs**3))/pos_r) * np.log(1+(pos_r/Rs))
        out = np.zeros((len(positions)),dtype=float)
        for idx,pos in enumerate(positions):
            out[idx] = phi(Rs,p0,pos)
        return out

def angles2vectors(alphas,betas):
    x = np.cos(alphas) * np.cos(betas)
    z = np.sin(alphas) * np.cos(betas)
    y = np.sin(betas)
    return np.column_stack([x,y,z])

def randangles(size=10):
    return np.random.uniform(0,2*np.pi,size=size),np.random.uniform(0,2*np.pi,size=size)

def random_vectors(size=1):
    return angles2vectors(*randangles(size))

def ray(vector,length,nsteps,file=None):
    vector = np.reshape(vector/np.linalg.norm(vector),(1,) + vector.shape)
    rs = np.reshape(np.linspace(0,length,nsteps),(1,nsteps)).T
    points = rs * vector
    df = pd.DataFrame(points,columns=["x","y","z"])
    if file != None:
        df.to_csv(file,index=False)
    return df

def ray_rs(length,nsteps):
    return np.linspace(0,length,nsteps)

def points2radius(points):
    points = points.loc[:,["x","y","z"]].to_numpy()
    return spatial.distance.cdist(np.array([[0,0,0]]),points).flatten()
