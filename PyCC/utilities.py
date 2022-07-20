import numpy as np
import pandas as pd
from scipy import spatial
from scipy import special
import warnings
warnings.filterwarnings("ignore")

class Distributions(object):
    @staticmethod
    def Uniform(n,r=1,p=1,G=1,file=None):
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
        df.insert(0,"id",range(len(df)))
        if file != None:
            df.to_csv(file,index=False)
        return df

    @staticmethod
    def NFW(n,Rs=1,p0=1,c=1,a=100,G=1,file=None):
        
        def mu(x):
            return np.log(1.0 + x) - x / (1.0 + x)

        def qnfw(p, c, logp=False):
            if (logp):
                p = np.exp(p)
            p[p>1] = 1
            p[p<=0] = 0
            p *= mu(c)
            return (-(1.0/np.real(special.lambertw(-np.exp(-p-1))))-1)/c

        def rnfw(n,c,a):
            return qnfw(np.random.rand(int(n)), c=c * a)
        
        def vcirc(r,c,Rs):
            x = r/Rs
            return  np.sqrt((1/x) * (np.log(1+c*x) - (c*x)/(1+c*x))/(np.log(1+c)-c/(1+c)))
                    
        Rvir = c*Rs
        aRvir = a * Rvir
        
        maxMass = 4 * np.pi * p0 * (Rs**3) * (np.log(1+a*c) - ((a*c)/(1+a*c)))
        virialMass = 4 * np.pi * p0 * (Rs**3) * (np.log(1+c) - (c/(1+c)))

        radiuses = rnfw(n,c,a) * aRvir

        phi = np.random.uniform(low=0,high=2*np.pi,size=n)
        theta = np.arccos(np.random.uniform(low=-1,high=1,size=n))
        x = radiuses * np.sin(theta) * np.cos(phi)
        y = radiuses * np.sin(theta) * np.sin(phi)
        z = radiuses * np.cos(theta)

        Vvir = np.sqrt((G*virialMass)/Rvir)

        vel = np.zeros_like(radiuses)
        for idx,r in enumerate(radiuses):
            vel[idx] = vcirc(r,c,Rs) * Vvir
        
        phi = np.random.uniform(low=0,high=2*np.pi,size=n)
        theta = np.arccos(np.random.uniform(low=-1,high=1,size=n))

        x_vel = vel * np.sin(theta) * np.cos(phi)
        y_vel = vel * np.sin(theta) * np.sin(phi)
        z_vel = vel * np.cos(theta)

        particle_mass = maxMass/n
        particles = np.column_stack([x,y,z])
        velocities = np.column_stack([x_vel,y_vel,z_vel])
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
    
    @staticmethod
    def Plummer(a,M,positions,G):
        positions = positions.loc[:,["x","y","z"]].to_numpy()
        def phi(a,M,G,pos):
            pos_r = spatial.distance.cdist(np.array([[0,0,0]]),np.reshape(pos,(1,)+pos.shape)).flatten()[0]
            return (-1) * G * M * (1/np.sqrt((pos_r**2) + (a**2)))
        out = np.zeros((len(positions)),dtype=float)
        for idx,pos in enumerate(positions):
            out[idx] = phi(a,M,G,pos)
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

def outdf2numpy(df):
    steps = np.unique(df.loc[:,"step"].to_numpy())
    nsteps = steps.shape[0]
    ids = np.unique(df.loc[:,"id"].to_numpy())
    nparticles = ids.shape[0]
    pos = df.loc[:,["x","y","z"]].to_numpy()
    pos = pos.reshape(nsteps,nparticles,3)
    vel = df.loc[:,["vx","vy","vz"]].to_numpy()
    vel = vel.reshape(nsteps,nparticles,3)
    acc = df.loc[:,["ax","ay","az"]].to_numpy()
    acc = acc.reshape(nsteps,nparticles,3)
    gpe = df.loc[:,["gpe"]].to_numpy()
    gpe = gpe.reshape(nsteps,nparticles,1)
    return {"pos":pos,"vel":vel,"acc":acc,"gpe":gpe}

def downsample(df,amount):
    assert amount >= 1
    amount = 1/amount
    ids = np.unique(df.loc[:,"id"].to_numpy())
    particle_mass = df.loc[:,"mass"][0]
    total_mass = particle_mass * ids.shape[0]
    new_n = int(ids.shape[0] * amount)
    new_mass = (total_mass/new_n) 
    choices = np.random.choice(ids,new_n,replace=False)
    new_df = df[df["id"].isin(choices)]
    new_df.loc[:,"id"] = range(len(new_df))
    new_df.loc[:,"mass"] = [new_mass for i in range(len(new_df))]
    return new_df