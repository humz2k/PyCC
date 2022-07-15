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

    varyings=["acc","phi"],

)

def phi_acc(particles,masses,eps,G,n_particles,max_input,n_batches,part_buffer,eps_buffer,G_buffer,
            outbuffer_big,outbuffer_small,
            eval_part_buffer_big,eval_part_buffer_small,
            vao_big,vao_small):

    gpu_time = 0

    combined_part_mass = np.concatenate((particles,masses),axis=1).astype("f4").tobytes()
    
    out = np.zeros((n_particles,4),dtype=np.float32)

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

        first = time.perf_counter()

        vao.transform(outbuffer,instances=input_size)
        temp = outbuffer.read()

        second = time.perf_counter()

        gpu_time += second-first

        temp = np.ndarray((n_particles*input_size*4),"f4",outbuffer.read())

        sum_time_start = time.perf_counter()
        x = np.sum(np.reshape(temp[1::4],(input_size,n_particles)),axis=1)
        y = np.sum(np.reshape(temp[2::4],(input_size,n_particles)),axis=1)
        z = np.sum(np.reshape(temp[3::4],(input_size,n_particles)),axis=1)
        phis = np.sum(np.reshape(temp[::4],(input_size,n_particles)),axis=1)
        out[start:end] = np.column_stack((x,y,z,phis))

        sum_end = time.perf_counter()

    return out,gpu_time,sum_end - sum_time_start

def evaluate(particles, velocities, masses, steps = 0, eps = 0, G = 1,dt = 1):

    stats = {}
    start = time.perf_counter()

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

    save_phi_acc = np.zeros((steps+1,n_particles,4),dtype=np.float32)
    save_vel = np.zeros((steps+1,n_particles,3),dtype=np.float32)
    save_pos = np.zeros_like(save_vel)

    masses = masses.reshape(masses.shape + (1,))

    current_pos = np.copy(particles)
    current_vel = np.copy(velocities)

    start_eval = time.perf_counter()

    save_pos[0] = current_pos
    save_vel[0] = current_vel
    gpu_time = 0
    phi_acc_time = 0
    total_sum_time = 0

    phi_acc_time_first = time.perf_counter()
    out,temp_time,sum_time = phi_acc(current_pos,masses,eps,G,n_particles,max_input,n_batches,
                part_buffer,
                eps_buffer,
                G_buffer,
                outbuffer_big,
                outbuffer_small,
                eval_part_buffer_big,
                eval_part_buffer_small,
                vao_big,
                vao_small)
    phi_acc_time_second = time.perf_counter()
    phi_acc_time += phi_acc_time_second-phi_acc_time_first
    total_sum_time += sum_time
    
    gpu_time += temp_time
    
    save_phi_acc[0] = out
    current_acc = out[:,[0,1,2]]

    for step in range(steps):
        current_pos += current_vel/2
        current_vel += current_acc
        current_pos += current_vel/2

        save_pos[step+1] = current_pos
        save_vel[step+1] = current_vel

        phi_acc_time_first = time.perf_counter()
        out,temp_time,sum_time = phi_acc(current_pos,masses,eps,G,n_particles,max_input,n_batches,
                part_buffer,
                eps_buffer,
                G_buffer,
                outbuffer_big,
                outbuffer_small,
                eval_part_buffer_big,
                eval_part_buffer_small,
                vao_big,
                vao_small)
        phi_acc_time_second = time.perf_counter()
        phi_acc_time += phi_acc_time_second-phi_acc_time_first
        
        gpu_time += temp_time
        total_sum_time += sum_time
        
        save_phi_acc[step+1] = out
        current_acc = out[:,[0,1,2]]
    
    end_eval = time.perf_counter()
                            
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

    start_save = time.perf_counter()

    step_labels = pd.DataFrame(np.reshape(np.repeat(np.arange(steps+1),particles.shape[0]),((steps+1)*particles.shape[0],1)),columns=["step"])
    ids = pd.DataFrame(np.reshape(np.array([np.arange(particles.shape[0])] * (steps+1)).flatten(),(particles.shape[0] * (steps+1),1)),columns=["id"])
    save_vel = pd.DataFrame(np.reshape(save_vel,(save_vel.shape[0] * save_vel.shape[1], save_vel.shape[2])),columns=["vx","vy","vz"])
    save_pos = pd.DataFrame(np.reshape(save_pos,(save_pos.shape[0] * save_pos.shape[1], save_pos.shape[2])),columns=["x","y","z"])
    save_phi_acc = pd.DataFrame(np.reshape(save_phi_acc,(save_phi_acc.shape[0] * save_phi_acc.shape[1], save_phi_acc.shape[2])),columns=["ax","ay","az","phi"])

    save_out = pd.concat((step_labels,ids,save_pos,save_vel,save_phi_acc),axis=1)

    end = time.perf_counter()

    stats["save_time"] = end-start_save
    stats["total_time"] = end-start
    stats["eval_time"] = end_eval - start_eval
    stats["release_time"] = start_save - end_eval
    stats["setup_time"] = start_eval - start
    stats["nbatches"] = n_batches * (steps+1)
    stats["gpu_time"] = gpu_time
    stats["phi_acc_time"] = phi_acc_time
    stats["sum_time"] = total_sum_time

    return save_out,stats


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