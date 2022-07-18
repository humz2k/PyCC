import numpy as np
import struct
import moderngl
import time
from scipy import constants
from math import ceil
import pandas as pd

def get_prog(n):
    return """
    #version 410

    in vec3 pos;

    in float eps;
    in float G;

    out float phi;
    out vec3 acc;

    uniform myBlock{
        vec4 parts[""" + str(n) + """];
    };

    uniform int n;

    float d;
    float acc_mul;

    void main() {
        phi = 0.;
        acc[0] = 0.;
        acc[1] = 0.;
        acc[2] = 0.;

        for (int i = 0; i < n; ++i){
            d = sqrt(pow(parts[i][0] - pos[0],2) + pow(parts[i][1] - pos[1],2) + pow(parts[i][2] - pos[2],2));
            if (d != 0){
                if (eps != 0){
                    d = sqrt(pow(d,2) + eps);
                }
                phi = phi + (-1) * G * parts[i][3] / d;
                acc_mul = G * parts[i][3] / pow(d,3);
                acc[0] = acc[0] + (parts[i][0] - pos[0]) * acc_mul;
                acc[1] = acc[1] + (parts[i][1] - pos[1]) * acc_mul;
                acc[2] = acc[2] + (parts[i][2] - pos[2]) * acc_mul;
            }
            
        }

    }

    """

def evaluate(particles, masses, steps, pos, eps = 0, G = 1, batch_size=4096):

    first = time.perf_counter()

    ctx = moderngl.create_context(standalone=True)

    fbo = ctx.simple_framebuffer((1,1))
    fbo.use()

    n_particles = particles.shape[1]
    n_steps = particles.shape[0]
    n_pos = pos.shape[0]

    n_batches = ceil(n_particles/batch_size)

    prog = ctx.program(
        vertex_shader=get_prog(batch_size),
        varyings=["acc","phi"],
    )

    pos_buffer = ctx.buffer(pos.astype("f4").tobytes())
    allParts = ctx.buffer(reserve=n_particles*4*4)
    allParts.bind_to_uniform_block(0)
    eps_buffer = ctx.buffer(np.array([[eps**2]]).astype("f4").tobytes())
    G_buffer = ctx.buffer(np.array([[G]]).astype("f4").tobytes())
    out_buffer = ctx.buffer(reserve=n_pos*4*4)

    particles_f4 = particles.astype("f4")
    masses_f4 = masses.astype("f4").reshape(masses.shape[0],1)

    vao = ctx.vertex_array(prog, [(pos_buffer, "3f", "pos"),(eps_buffer, "1f /r", "eps"), (G_buffer, "1f /r", "G")])

    save_phi_acc = np.zeros((len(steps),n_pos,4),dtype=np.float32)

    for idx,step in enumerate(steps):

        out = phi_acc(prog,vao,pos_buffer,allParts,out_buffer,particles[step].astype("f4"),masses_f4,n_particles,n_batches,n_pos)

        save_phi_acc[idx] = out

    vao.release()
    prog.release()
    pos_buffer.release()
    allParts.release()
    eps_buffer.release()
    G_buffer.release()
    fbo.release()
    ctx.release()

    step_labels = pd.DataFrame(np.reshape(np.repeat(np.array(steps),pos.shape[0]),(len(steps)*pos.shape[0],1)),columns=["step"])
    ids = pd.DataFrame(np.reshape(np.array([np.arange(pos.shape[0])] * len(steps)).flatten(),(pos.shape[0] * len(steps),1)),columns=["id"])
    save_phi_acc = pd.DataFrame(np.reshape(save_phi_acc,(save_phi_acc.shape[0] * save_phi_acc.shape[1], save_phi_acc.shape[2])),columns=["ax","ay","az","phi"])

    second = time.perf_counter()

    return pd.concat((step_labels,ids,save_phi_acc),axis=1),{"eval_time":second-first}

def phi_acc(prog,vao,pos_buffer,allParts,out_buffer,particles,masses,n_particles,n_batches,n_pos):
    combined = np.concatenate((particles,masses),axis=1).tobytes()

    out = np.zeros((n_pos,4),dtype=np.float32)

    current_n = 4096
    if n_particles < current_n:
        current_n = n_particles

    prog.__getitem__("n").write(np.array([[4096]]).astype(np.int32).tobytes())

    start = 0
    end = n_particles
    for batch in range(n_batches - 1):
        start = batch * 4 * 4
        end = batch * 4 * 4 + (4096) * 4 * 4 
        allParts.write(combined[start:end])
        vao.transform(out_buffer)
        out += np.ndarray((n_pos,4),"f4",out_buffer.read())

    prog.__getitem__("n").write(np.array([[n_particles - (n_batches-1) * 4096]]).astype(np.int32).tobytes())
    allParts.write(combined[(n_batches-1) * 4096 * 4 * 4:n_particles * 4 * 4])

    vao.transform(out_buffer)

    out += np.ndarray((n_pos,4),"f4",out_buffer.read())

    return out