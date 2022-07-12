#DO UNITS

import numpy as np
import struct
import moderngl
import time
from astropy import constants
from math import ceil

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

def evaluate(particles,masses,evaluate_at,eps=0,G = constants.G.value):

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
    
    return out_acc,out_phi,{"eval_time":second-first,"n_batches":n_batches,"batch_times":times}