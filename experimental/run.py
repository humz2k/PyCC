import direct_sum
import numpy as np
import pandas as pd
import scipy.spatial as spatial
import astropy.constants as const
import time

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

class DirectSum(object):
    @staticmethod
    def dists(pos,particles):
        return spatial.distance.cdist(particles,np.reshape(pos,(1,)+pos.shape))

    @staticmethod
    def acc_and_phi(pos,particles,masses,eps=0):
        dists = DirectSum.dists(pos,particles).flatten()
        masses = masses[dists != 0]
        parts = particles[dists != 0]
        dists = dists[dists != 0]
        if eps == 0:
            potentials = (-1) * const.G.value * (masses)/dists
            muls = (const.G.value * ((masses) / (dists**3)))
            accelerations = (parts - pos) * np.reshape(muls,(1,) + muls.shape).T
        else:
            potentials = (-1) * const.G.value * (masses)/((dists**2+eps**2)**(1/2))
            muls = (const.G.value * masses / (((dists**2+eps**2)**(1/2))**3))
            accelerations = (parts - pos) * np.reshape(muls,(1,) + muls.shape).T
        return np.sum(accelerations,axis=0),np.sum(potentials)

    @staticmethod
    def acc_func(positions,particles,masses,eps=0):
        acc = np.zeros((positions.shape[0],3),dtype=float)
        phi = np.zeros(positions.shape[0],dtype=float)
        for idx,pos in enumerate(positions):
            temp_acc,temp_phi = DirectSum.acc_and_phi(pos,particles,masses,eps)
            phi[idx] = temp_phi
            acc[idx] = temp_acc
        return acc,phi

df = Distributions.Uniform(5,3,100)
pos = df.loc[:,["x","y","z"]].to_numpy()
masses = df.loc[:,["mass"]].to_numpy().flatten()
vels = np.zeros_like(pos,dtype=float)
eval_pos = np.array([[0,0,0],[0,0,5]],dtype=float)

first = time.perf_counter()
part_out, eval_pos_out = direct_sum.evaluate(pos,vels,masses,steps=1)
second = time.perf_counter()
print(second-first)

'''
first = time.perf_counter()
out_fast = np.asarray(direct_sum.phi_acc(pos,masses,pos,const.G.value,10))
acc_fast = out_fast[:,[0,1,2]]
phi_fast = out_fast[:,3]
second = time.perf_counter()
out_fast = np.asarray(out_fast)
print(second-first)

first = time.perf_counter()
acc,phi = DirectSum.acc_func(pos,pos,masses,10)
second = time.perf_counter()
print(second-first)

#print(np.mean(np.abs(acc-acc_fast)))
print(np.mean(np.abs(phi-phi_fast)))
'''
