import numpy as np
import struct
import moderngl
import time
from scipy import constants
from math import ceil
import pandas as pd

ctx = moderngl.create_context(standalone=True)

prog = ctx.program(

    vertex_shader="""
    #version 330

    out vec3 acc;
    out float phi;

    in vec3 part;
    in float part_mass;

    in vec3 eval_part;
    in float eval_part_mass;

    in float eps;
    in float G;

    float d;
    float acc_mul;

    void main() {
        d = distance(part,eval_part);
        if (d == 0){
            acc[0] = 0.;
            acc[1] = 0.;
            acc[2] = 0.;
            phi = 0.;
        } else{
            if (eps != 0){
                d = sqrt(pow(d,2) + pow(eps,2));
            }
            phi = (-1) * G * part_mass * eval_part_mass / pow(d,2);
            acc_mul = G * part_mass / pow(d,3);
            acc[0] = (part[0] - eval_part[0]) * acc_mul;
            acc[1] = (part[1] - eval_part[1]) * acc_mul;
            acc[2] = (part[2] - eval_part[2]) * acc_mul;
         }
    }

    """,

    varyings=["phi","acc"],

)



def do_batch(part_buffer,
            mass_buffer,
            eval_part_buffer,
            eval_part_mass_buffer,
            eps_buffer,
            G_buffer,
            outbuffer,
            n_particles,
            n_evals):

    #outbuffer = ctx.buffer(reserve=n_particles*n_evals*4*4)

    vao = ctx.vertex_array(prog, [(part_buffer, "3f", "part"),
                                    (mass_buffer, "1f", "part_mass"), 
                                    (eval_part_buffer, "3f /i", "eval_part"), 
                                    (eval_part_mass_buffer, "3f /i", "eval_part_mass"), 
                                    (eps_buffer, "1f /r", "eps"), 
                                    (G_buffer, "1f /r", "G")])

    vao.transform(outbuffer,instances=n_evals)

    out = np.ndarray((n_particles*n_evals*4),"f4",outbuffer.read())
    x = np.sum(np.reshape(out[1::4],(n_evals,n_particles)),axis=1)
    y = np.sum(np.reshape(out[2::4],(n_evals,n_particles)),axis=1)
    z = np.sum(np.reshape(out[3::4],(n_evals,n_particles)),axis=1)
    phis = np.sum(np.reshape(out[::4],(n_evals,n_particles)),axis=1)
    acc = np.column_stack((x,y,z))

    vao.release()

    return acc,phis

def phi_acc(particles,masses,eps=0,G=1):

    fbo = ctx.simple_framebuffer((1,1))
    fbo.use()

    max_output_size = int((ctx.info["GL_MAX_TEXTURE_BUFFER_SIZE"]*2))/4
    n_particles = particles.shape[0]
    max_input = int(max_output_size/n_particles)

    particles_f4 = particles.astype("f4")
    masses_f4 = masses.astype("f4")
    
    out_acc = np.zeros_like(particles,dtype=np.float32)
    out_phi = np.zeros_like(masses,dtype=np.float32)

    part_buffer = ctx.buffer(particles_f4.tobytes())
    part_mass_buffer = ctx.buffer(masses_f4.tobytes())
    eps_buffer = ctx.buffer(np.array([[eps]]).astype("f4").tobytes())
    G_buffer = ctx.buffer(np.array([[G]]).astype("f4").tobytes())
    
    n_batches = ceil(n_particles/max_input)
    
    if n_batches == 1:
        max_input = n_particles
    
    eval_part_buffer = ctx.buffer(reserve=max_input * 3 * 4)
    eval_mass_buffer = ctx.buffer(reserve=max_input * 4)

    for i in range(n_batches):
        start = i * max_input
        end = start + max_input

        print(start,end)

        if end > n_particles:
            print("YEET")

        #if n_evals - start < max_input:
        #    end = n_evals
        #else:
        #    end = start + max_input
        #acc,phi,batch_time = do_batch(part_buffer,mass_buffer,eps_buffer,G_buffer,n_particles,evaluate_at[start:end])
        #out_phi[start:end] = phi
        #out_acc[start:end] = acc
        #start = end
        #times[i] = batch_time
    
    fbo.release()
    part_buffer.release()
    part_mass_buffer.release()
    eps_buffer.release()
    G_buffer.release()
    eval_part_buffer.release()
    eval_mass_buffer.release()

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

df = Uniform(10,50000,10)
pos = df.loc[:,["x","y","z"]].to_numpy()
masses = df.loc[:,["mass"]].to_numpy()

phi_acc(pos,masses)