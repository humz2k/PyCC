import numpy as np
import pandas as pd
from scipy import spatial
from scipy import special

class Distributions(object):
    @staticmethod
    def Uniform(n,r,p,file=None):
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
    def Plummer(n,a=1,M=1,G=1,file=None):
        phi = np.random.uniform(low=0,high=2*np.pi,size=n)
        theta = np.arccos(np.random.uniform(low=-1,high=1,size=n))
        particle_r = a / np.sqrt(((np.random.uniform(low=0,high=1,size=n)**(-2./3.))) - 1)
        x_pos = particle_r * np.sin(theta) * np.cos(phi)
        y_pos = particle_r * np.sin(theta) * np.sin(phi)
        z_pos = particle_r * np.cos(theta)
        particle_mass = (M)/n
        particles = np.column_stack([x_pos,y_pos,z_pos])

        x = np.zeros((n),dtype=float)
        y = np.zeros((n),dtype=float)
        
        idx = 0
        while idx < n:
            temp_x = np.random.uniform(low=0,high=1,size=1)[0]
            temp_y = np.random.uniform(low=0,high=0.1,size=1)[0]
            if temp_y <= temp_x*temp_x*((1.0 - temp_x**2)**3.5):
                x[idx] = temp_x
                y[idx] = temp_y
                idx += 1

        vel = x * np.sqrt(2.0) * np.sqrt((G * M)/(np.sqrt(a**2 + particle_r**2)))
        phi = np.random.uniform(low=0,high=2*np.pi,size=n)
        theta = np.arccos(np.random.uniform(low=-1,high=1,size=n))

        x_vel = vel * np.sin(theta) * np.cos(phi)
        y_vel = vel * np.sin(theta) * np.sin(phi)
        z_vel = vel * np.cos(theta)

        velocities = np.column_stack([x_vel,y_vel,z_vel])
        masses = pd.DataFrame(np.full((1,n),particle_mass).T,columns=["mass"])
        particles = pd.DataFrame(particles,columns=["x","y","z"])
        velocities = pd.DataFrame(velocities,columns=["vx","vy","vz"])
        df = pd.concat((particles,velocities,masses),axis=1)
        if file != None:
            df.to_csv(file,index=False)
        return df

    @staticmethod
    def NFW(Rs,p0,c,n,file=None):
        def mu(x):
            return np.log(1.0 + x) - x / (1.0 + x)

        def qnfw(p, c, logp=False):
            if (logp):
                p = np.exp(p)
            p[p>1] = 1
            p[p<=0] = 0
            p *= mu(c)
            return (-(1.0/np.real(special.lambertw(-np.exp(-p-1))))-1)/c

        def rnfw(n, c):
            return qnfw(np.random.rand(int(n)), c=c)

        Rvir = c*Rs

        #maxMass = 4*np.pi*p0*(Rs**3)*mu(c)
        maxMass = 4*np.pi*p0*(Rs**3)*(np.log(1+Rvir/Rs) - (Rvir/Rs)/(1+(Rvir/Rs)))
        radiuses = rnfw(n,c) * Rvir

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
    def Uniform(r,p,positions,G):
        positions = positions.loc[:,["x","y","z"]].to_numpy()
        def phi(r,p,pos):
            pos_r = spatial.distance.cdist(np.array([[0,0,0]]),np.reshape(pos,(1,)+pos.shape)).flatten()[0]
            relative = pos_r/r
            if relative == 1:
                return (-4/3) * np.pi * G * p * (r ** 2)
            elif relative < 1:
                return (-2) * np.pi * G * p * ((r ** 2) - ((1/3) * ((pos_r)**2)))
            else:
                return (-4/3) * np.pi * G * p * ((r ** 3)/(pos_r))
        out = np.zeros((len(positions)),dtype=float)
        for idx,pos in enumerate(positions):
            out[idx] = phi(r,p,pos)
        return out

    @staticmethod
    def NFW(Rs,p0,positions,G):
        positions = positions.loc[:,["x","y","z"]].to_numpy()
        def phi(Rs,pos):
            pos_r = spatial.distance.cdist(np.array([[0,0,0]]),np.reshape(pos,(1,)+pos.shape)).flatten()[0]
            if pos_r == 0:
                return -4 * np.pi * G * p0 * (Rs**2)
            return (-4 * np.pi * G * p0 * (Rs**2)) * np.log(1+(pos_r/Rs))/(pos_r/Rs)
        out = np.zeros((len(positions)),dtype=float)
        for idx,pos in enumerate(positions):
            out[idx] = phi(Rs,pos)
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
