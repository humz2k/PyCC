import cython
cimport numpy as np
import numpy as np
import struct
import moderngl
import time
from scipy import constants
from libc.math cimport ceil
import pandas as pd

ctx = moderngl.create_context(standalone=True)

prog = ctx.program(

    vertex_shader="""
    #version 330

    out vec3 acc;
    out float phi;

    in vec4 part;

    in vec4 eval_part;

    in float eps;
    in float G;

    float d;
    float acc_mul;

    void main() {
        d = sqrt(pow(part[0] - eval_part[0],2) + pow(part[1] - eval_part[1],2) + pow(part[2] - eval_part[2],2) + eps);
        if (d == 0){
            phi = 0.;
            acc[0] = 0.;
            acc[1] = 0.;
            acc[2] = 0.;
        } else{
            phi = (-1) * G * (part[3] * eval_part[3]) / pow(d,2);
            acc_mul = G * part[3] / pow(d,3);
            acc[0] = (part[0] - eval_part[0]) * acc_mul;
            acc[1] = (part[1] - eval_part[1]) * acc_mul;
            acc[2] = (part[2] - eval_part[2]) * acc_mul;
         }
    }

    """,

    varyings=["phi","acc"],

)

def phi_acc(particles,masses,eps,G,n_particles,max_input,n_batches,part_buffer,eps_buffer,G_buffer,
            outbuffer_big,outbuffer_small,
            eval_part_buffer_big,eval_part_buffer_small,
            vao_big,vao_small):

    combined_part_mass = np.concatenate((particles,masses),axis=1).astype("f4").tobytes()
    
    out_acc = np.zeros_like(particles,dtype=np.float32)
    out_phi = np.zeros_like(masses,dtype=np.float32)

    part_buffer.write(combined_part_mass)

    outbuffer = outbuffer_big
    eval_part_buffer = eval_part_buffer_big

    vao = vao_big

    for i in range(n_batches):

        input_size = max_input

        start = i * max_input

        if (start + input_size) > n_particles:
            input_size = n_particles - start

            eval_part_buffer = eval_part_buffer_small
            outbuffer = outbuffer_small

            vao = vao_small
        
        start = start
        end = start + (input_size)
        eval_part_buffer.write(combined_part_mass[start * 4 * 4:start * 4 * 4 + (input_size) * 4 * 4])

        vao.transform(outbuffer,instances=input_size)

        out = np.ndarray((n_particles*input_size*4),"f4",outbuffer.read())

        x = np.sum(np.reshape(out[1::4],(input_size,n_particles)),axis=1)
        y = np.sum(np.reshape(out[2::4],(input_size,n_particles)),axis=1)
        z = np.sum(np.reshape(out[3::4],(input_size,n_particles)),axis=1)
        phis = np.sum(np.reshape(out[::4],(input_size,n_particles)),axis=1)
        acc = np.column_stack((x,y,z))
        
        out_acc[start:end] = acc
        out_phi[start:end] = phis.reshape((phis.shape[0],1))

    return out_acc,out_phi

def evaluate(particles, velocities, masses, steps = 0, eps = 0, G = 1,dt = 1):
    fbo = ctx.simple_framebuffer((1,1))
    fbo.use()

    max_output_size = int((ctx.info["GL_MAX_TEXTURE_BUFFER_SIZE"]*2))/4
    n_particles = particles.shape[0]
    max_input = int(max_output_size/n_particles)

    n_batches = ceil(n_particles/max_input)

    if n_batches == 1:
        max_input = n_particles

    part_buffer = ctx.buffer(reserve=(n_particles * 4 * 4))
    eps_buffer = ctx.buffer(np.array([[eps**2]]).astype("f4").tobytes())
    G_buffer = ctx.buffer(np.array([[G]]).astype("f4").tobytes())

    outbuffer_big = ctx.buffer(reserve=n_particles*max_input*4*4)
    eval_part_buffer_big = ctx.buffer(reserve=max_input * 4 * 4)

    vao_big = ctx.vertex_array(prog, [(part_buffer, "4f", "part"),(eval_part_buffer_big, "4f /i", "eval_part"), (eps_buffer, "1f /r", "eps"), (G_buffer, "1f /r", "G")])

    outbuffer_small = None
    eval_part_buffer_small = None
    vao_small = None
    if n_batches != 1:
        if n_batches * max_input != n_particles:
            size = n_particles - ((n_batches - 1) * max_input)
            outbuffer_small = ctx.buffer(reserve=(n_particles*size*4*4))
            eval_part_buffer_small = ctx.buffer(reserve=size * 4 * 4)
            vao_small = ctx.vertex_array(prog, [(part_buffer, "4f", "part"),(eval_part_buffer_small, "4f /i", "eval_part"), (eps_buffer, "1f /r", "eps"), (G_buffer, "1f /r", "G")])

    acc,phi = phi_acc(particles,masses,eps,G,n_particles,max_input,n_batches,
                        part_buffer,
                        eps_buffer,
                        G_buffer,
                        outbuffer_big,
                        outbuffer_small,
                        eval_part_buffer_big,
                        eval_part_buffer_small,
                        vao_big,
                        vao_small)

    fbo.release()
    part_buffer.release()
    eps_buffer.release()
    G_buffer.release()
    outbuffer_big.release()
    eval_part_buffer_big.release()
    vao_big.release()
    if outbuffer_small != None:
        eval_part_buffer_small.release()
        outbuffer_small.release()
        vao_small.release()

    return acc,phi


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

df = Uniform(10,20000,1)
pos = df.loc[:,["x","y","z"]].to_numpy()
masses = df.loc[:,["mass"]].to_numpy()
vels = df.loc[:,["vx","vy","vz"]].to_numpy()

acc,phi = evaluate(pos,vels,masses,G=constants.G)