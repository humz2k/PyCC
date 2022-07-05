import numpy as np
import pandas as pd
from scipy import spatial
from scipy import constants
from scipy.special import lambertw
import time
import os,sys

fpath = os.path.join(os.path.dirname(__file__))
sys.path.append(fpath)

import treecode
import directsum_gpu

#different schemes for nbody simulations
schemes = {}
schemes["euler"] = "adk"
schemes["kick-drift"] = "akd"
schemes["drift-kick"] = "dak"
schemes["leapfrog"] = "dakd"

def evaluate(save=False, #save output to file
            file=None, #name of file to read from
            outfile=None, #name of output file
            df=None, #dataframe to read from instead of file
            evaluate_at = None, #dataframe to evaluate points at. If this is none, it evaluates at the points themselves
            accelerate=False, #whether to use GPU accelerations
            algo = "directsum", #directsum vs treecode
            eval_type="both", #just phi or both phi and acceleration
            scheme=schemes["leapfrog"], #scheme to use
            dt=1000, #delta time
            steps=1, #number of steps
            **kwargs): #other keyword arguments

    if type(df) == type(None): #if df isnt specified, read from file
        a = pd.read_csv(file)
    else:
        a = df
    
    #get positions/masses/velocities from the df
    particles = a.loc[:,["x","y","z"]].to_numpy(dtype=float)
    masses = a.loc[:,"mass"].to_numpy(dtype=float)
    velocities = a.loc[:,["vx","vy","vz"]].to_numpy(dtype=float)

    #if evaluate is none, then set the evaluate points to the particles
    if type(evaluate_at) == type(None):
        evaluate_at = particles
    else:
        evaluate_at = evaluate_at.loc[:,["x","y","z"]].to_numpy(dtype=float)

    first = time.perf_counter() #time execution

    stats = {} #save stats here

    if eval_type == "phi": #if we are just getting the phis

        ids = np.reshape(np.arange(len(evaluate_at)),(1,len(evaluate_at))).T

        if algo == "directsum":
            if accelerate:
                accs,phis,stats = directsum_gpu.evaluate(particles,masses,evaluate_at,**kwargs)
            else:
                accs,phis = DirectSum.acc_func(evaluate_at,particles,masses,**kwargs)
        
        elif algo == "treecode":
            tree = treecode.Tree(particles,masses)
            build_time = tree.build_tree()
            accs,phis,stats = tree.evaluate(evaluate_at,**kwargs)
            stats["tree_build_time"] = build_time
        
        positions = evaluate_at
        vels = velocities

    else:

        ids = np.reshape(np.arange(len(particles)),(1,len(particles))).T

        scheme = scheme.lower()
        drift_t = 1/scheme.count("d")
        kick_t = 1/scheme.count("k")

        positions = np.zeros((steps+1,)+particles.shape,dtype=float)
        vels = np.zeros((steps+1,)+particles.shape,dtype=float)
        accs = np.zeros((steps+1,)+particles.shape,dtype=float)
        phis = np.zeros((steps+1,len(particles)),dtype=float)

        vels[0] = velocities
        positions[0] = particles

        truncations = 0
        directs = 0

        for step in range(steps + 1):
            for action in scheme:
                if action == "a":
                    if algo == "directsum":
                        if accelerate:
                            acc,temp_phi,stats = directsum_gpu.evaluate(particles,masses,particles,**kwargs)
                        else:
                            acc,temp_phi = DirectSum.acc_func(particles,particles,masses,**kwargs)
                    elif algo == "treecode":
                        tree = treecode.Tree(particles,masses)
                        build_time = tree.build_tree()
                        acc,temp_phi,stats = tree.evaluate(particles,**kwargs)
                        stats["tree_build_time"] = build_time
                        truncations += stats["truncations"]
                        directs += stats["directs"]
                if action == "k":
                    velocities = velocities + acc*dt*kick_t
                if action == "d":
                    particles = particles + velocities*dt*drift_t
            phis[step] = temp_phi
            accs[step] = acc
            if step + 1 < len(positions):
                positions[step+1] = particles
                vels[step+1] = velocities
        
        if algo == "treecode":
            stats["truncations"] = truncations
            stats["directs"] = directs

        ids = np.array([ids for i in range(steps+1)])
        ids = np.reshape(ids,(ids.shape[0]*ids.shape[1],ids.shape[2]))

        positions = np.reshape(positions,(positions.shape[0]*positions.shape[1],positions.shape[2]))
        vels = np.reshape(vels,(vels.shape[0]*vels.shape[1],vels.shape[2]))
        accs = np.reshape(accs,(accs.shape[0]*accs.shape[1],accs.shape[2]))
        phis = np.reshape(phis,(phis.shape[0]*phis.shape[1]))

    second = time.perf_counter() #end timer
    eval_time = second-first #calculate execution time
    stats.update({"eval_time":eval_time}) #add this to stats

    #save this to a pd dataframe
    phis = pd.DataFrame(np.reshape(phis,(1,)+phis.shape).T,columns=["phi"])
    positions = pd.DataFrame(positions,columns=["x","y","z"])
    ids = pd.DataFrame(ids,columns=["id"],dtype=int)

    if eval_type == "phi":
        accs = pd.DataFrame(accs,columns=["ax","ay","az"])
        out = pd.concat((ids,positions,accs,phis),axis=1)
    else:
        accs = pd.DataFrame(accs,columns=["ax","ay","az"])
        vels = pd.DataFrame(vels,columns=["vx","vy","vz"])
        out = pd.concat((ids,positions,accs,vels,phis),axis=1)
    
    if save:
        if file == None:
            outfile = file.split(".")[0]+"_out"+".csv"
        out.to_csv(outfile,index=False)
    
    return out,stats

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
            potentials = (-1) * constants.G * (masses)/dists
            muls = (constants.G * ((masses) / (dists**3)))
            accelerations = (parts - pos) * np.reshape(muls,(1,) + muls.shape).T
        else:
            potentials = (-1) * constants.G * (masses)/((dists**2+eps**2)**(1/2))
            muls = (constants.G * masses / (((dists**2+eps**2)**(1/2))**3))
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

def points2radius(points):
    points = points.loc[:,["x","y","z"]].to_numpy()
    return spatial.distance.cdist(np.array([[0,0,0]]),points).flatten()
