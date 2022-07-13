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
    in vec3 eval_pos;
    in float mass;
    in float eps;
    in float G;

    float d;
    float acc_mul;

    void main() {
        d = distance(part,eval_pos);
        if (d == 0){
            acc[0] = 0.;
            acc[1] = 0.;
            acc[2] = 0.;
            phi = 0.;
        } else{

            if (eps != 0){
                d = sqrt(pow(d,2) + pow(eps,2));
            }
            phi = (-1) * G * mass / d;
            acc_mul = G * mass / pow(d,3);
            acc[0] = (part[0] - eval_pos[0]) * acc_mul;
            acc[1] = (part[1] - eval_pos[1]) * acc_mul;
            acc[2] = (part[2] - eval_pos[2]) * acc_mul;
         }
    }

    """,

    varyings=["phi","acc"],

)

def do_batch(part_buffer,mass_buffer,eps_buffer,G_buffer,n_particles,eval_pos):
    first = time.perf_counter()

    n_evals = eval_pos.shape[0]

    eval_buffer = ctx.buffer(eval_pos.astype("f4").tobytes())

    outbuffer = ctx.buffer(reserve=n_particles*n_evals*4*4)

    vao = ctx.vertex_array(prog, [(part_buffer, "3f", "part"),(eval_buffer, "3f /i", "eval_pos"), (mass_buffer, "1f", "mass"), (eps_buffer, "1f /r", "eps"), (G_buffer, "1f /r", "G")])

    vao.transform(outbuffer,instances=n_evals)

    out = np.ndarray((n_particles*n_evals*4),"f4",outbuffer.read())
    x = np.sum(np.reshape(out[1::4],(n_evals,n_particles)),axis=1)
    y = np.sum(np.reshape(out[2::4],(n_evals,n_particles)),axis=1)
    z = np.sum(np.reshape(out[3::4],(n_evals,n_particles)),axis=1)
    phis = np.sum(np.reshape(out[::4],(n_evals,n_particles)),axis=1)
    acc = np.column_stack((x,y,z))

    vao.release()
    outbuffer.release()
    eval_buffer.release()

    second = time.perf_counter()

    return acc,phis,second-first

def phi_acc(particles,masses,evaluate_at,eps=0,G = constants.G):

    first = time.perf_counter()

    max_output_size = int((ctx.info["GL_MAX_TEXTURE_BUFFER_SIZE"]*2))/4
    n_particles = particles.shape[0]
    n_evals = evaluate_at.shape[0]
    max_input = int(max_output_size/n_particles)
    length = n_evals
    out_acc = np.zeros_like(evaluate_at,dtype=float)
    out_phi = np.zeros(len(evaluate_at),dtype=float)

    fbo = ctx.simple_framebuffer((1,1))
    fbo.use()
    part_buffer = ctx.buffer(particles.astype("f4").tobytes())
    mass_buffer = ctx.buffer(masses.astype("f4").tobytes())
    eps_buffer = ctx.buffer(np.array([[eps]]).astype("f4").tobytes())
    G_buffer = ctx.buffer(np.array([[G]]).astype("f4").tobytes())

    start = 0
    n_batches = ceil(n_evals/max_input)
    times = np.zeros(n_batches,dtype=float)
    for i in range(n_batches):
        if n_evals - start < max_input:
            end = n_evals
        else:
            end = start + max_input
        acc,phi,batch_time = do_batch(part_buffer,mass_buffer,eps_buffer,G_buffer,n_particles,evaluate_at[start:end])
        out_phi[start:end] = phi
        out_acc[start:end] = acc
        start = end
        times[i] = batch_time

    fbo.release()
    part_buffer.release()
    G_buffer.release()
    mass_buffer.release()
    eps_buffer.release()

    second = time.perf_counter()
    
    return out_phi,out_acc,{"eval_time":second-first,"n_batches":n_batches,"batch_times":times}

def evaluate(particles,velocities,masses,eval_pos = None, steps = 0, eps = 0, G = constants.G,dt = 1000):

    first = time.perf_counter()

    positions = np.zeros((steps+1,)+particles.shape,dtype=np.float32)
    vels = np.zeros((steps+1,)+particles.shape,dtype=np.float32)
    accs = np.zeros((steps+1,)+particles.shape,dtype=np.float32)
    phis = np.zeros((steps+1,len(particles)),dtype=np.float32)

    vels[0] = velocities.astype(np.float32)
    positions[0] = particles.astype(np.float32)

    pos = particles.astype(np.float32)
    vel = velocities.astype(np.float32)
    
    if type(eval_pos) != type(None):
        eval_phi = np.zeros((steps+1,) + len(eval_pos),dtype=np.float32)
        eval_acc = np.zeros((steps+1,) + eval_pos.shape,dtype=np.float32)
        evals = eval_pos.astype(np.float32)
        temp_phi,temp_acc,stats = phi_acc(pos,masses,evals,eps,G)
        eval_phi[0] = temp_phi
        eval_acc[0] = temp_acc

    phi,acc,stats = phi_acc(pos,masses,pos,eps,G)
    phis[0] = phi
    accs[0] = acc

    out_stats = {}
    average_batch_time = np.sum(stats["batch_times"])
    n_batches = stats["n_batches"]
    
    for step in range(steps):
        pos += (vel/2) * dt
        vel += (acc) * dt
        pos += (vel/2) * dt

        vels[step+1] = vel
        positions[step+1] = pos
        
        phi,acc,stats = phi_acc(pos,masses,pos,eps,G)
        average_batch_time += np.sum(stats["batch_times"])
        n_batches += stats["n_batches"]
        phis[step+1] = phi
        accs[step+1] = acc

        if type(eval_pos) != type(None):
            eval_phi = np.zeros((steps+1,) + len(eval_pos),dtype=np.float32)
            eval_acc = np.zeros((steps+1,) + eval_pos.shape,dtype=np.float32)
            evals = eval_pos.astype(np.float32)
            temp_phi,temp_acc,stats = phi_acc(pos,masses,evals,eps,G)
            eval_phi[step+1] = temp_phi
            eval_acc[step+1] = temp_acc
    
    second = time.perf_counter()

    out_stats["eval_time"] = second-first
    out_stats["n_batches"] = n_batches
    out_stats["batch_times"] = average_batch_time/n_batches

    step_labels = pd.DataFrame(np.reshape(np.repeat(np.arange(steps+1),particles.shape[0]),(1,(steps+1)*particles.shape[0])).T,columns=["step"])

    ids = pd.DataFrame(np.reshape(np.array([np.arange(particles.shape[0])] * (steps+1)).flatten(),(1,particles.shape[0] * (steps+1))).T,columns=["id"])

    positions = pd.DataFrame(np.reshape(positions,(positions.shape[0] * positions.shape[1],positions.shape[2])),columns=["x","y","z"])
    
    vels = pd.DataFrame(np.reshape(vels,(vels.shape[0] * vels.shape[1],vels.shape[2])),columns=["vx","vy","vz"])

    accs = pd.DataFrame(np.reshape(accs,(accs.shape[0] * accs.shape[1],accs.shape[2])),columns=["ax","ay","az"])

    phis = pd.DataFrame(np.reshape(phis,(1,phis.shape[0] * phis.shape[1])).T,columns=["phi"])

    part_out = pd.concat((step_labels,ids,positions,vels,accs,phis),axis=1)
    

    if type(eval_pos) != type(None):
        step_labels = pd.DataFrame(np.reshape(np.repeat(np.arange(steps+1),eval_pos.shape[0]),(1,(steps+1)*eval_pos.shape[0])).T,columns=["step"])

        ids = pd.DataFrame(np.reshape(np.array([np.arange(eval_pos.shape[0])] * (steps+1)).flatten(),(1,eval_pos.shape[0] * (steps+1))).T,columns=["id"])

        accs = pd.DataFrame(np.reshape(eval_acc,(eval_acc.shape[0] * eval_acc.shape[1],eval_acc.shape[2])),columns=["ax","ay","az"])

        phis = pd.DataFrame(np.reshape(eval_phi,(1,eval_phi.shape[0] * eval_phi.shape[1])).T,columns=["phi"])

        eval_out = pd.concat((step_labels,ids,accs,phis),axis=1)

        return part_out,eval_out,out_stats

    return part_out,out_stats
    

