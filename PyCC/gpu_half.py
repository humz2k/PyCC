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

    in lowp vec4 pos;

    in lowp float eps;
    in lowp float G;

    out lowp float phi;
    out lowp vec3 acc;

    uniform myBlock{
        lowp vec4 parts[""" + str(n) + """];
    };

    uniform int n;

    lowp float d;
    lowp float acc_mul;

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
                phi = phi + (-1) * G * (pos[3] * parts[i][3]) / pow(d,2);
                acc_mul = G * parts[i][3] / pow(d,3);
                acc[0] = acc[0] + (parts[i][0] - pos[0]) * acc_mul;
                acc[1] = acc[1] + (parts[i][1] - pos[1]) * acc_mul;
                acc[2] = acc[2] + (parts[i][2] - pos[2]) * acc_mul;
            }
            
        }

    }

    """

def evaluate(particles, velocities, masses, steps = 0, eps = 0, G = 1,dt = 1):

    first = time.perf_counter()

    ctx = moderngl.create_context(standalone=True)

    fbo = ctx.simple_framebuffer((1,1))
    fbo.use()

    n_particles = particles.shape[0]

    batch_size = 4096
    n_batches = ceil(n_particles/4096)

    prog = ctx.program(
        vertex_shader=get_prog(4096),
        varyings=["acc","phi"],
    )

    pos_buffer = ctx.buffer(reserve=n_particles*4*2)
    allParts = ctx.buffer(reserve=n_particles*4*4)
    allParts.bind_to_uniform_block(0)
    eps_buffer = ctx.buffer(np.array([[eps**2]]).astype("f2").tobytes())
    G_buffer = ctx.buffer(np.array([[G]]).astype("f2").tobytes())
    out_buffer = ctx.buffer(reserve=n_particles*4*4)

    particles_f4 = particles.astype("f2")
    masses_f4 = masses.astype("f2").reshape(masses.shape[0],1)

    vao = ctx.vertex_array(prog, [(pos_buffer, "4f2", "pos"),(eps_buffer, "1f2 /r", "eps"), (G_buffer, "1f2 /r", "G")])

    save_phi_acc = np.zeros((steps+1,n_particles,4),dtype=np.float32)
    save_vel = np.zeros((steps+1,n_particles,3),dtype=np.float32)
    save_pos = np.zeros_like(save_vel,dtype=np.float32)

    current_pos = np.copy(particles)
    current_vel = np.copy(velocities)

    save_vel[0] = current_vel
    save_pos[0] = current_pos

    out = phi_acc(prog,vao,pos_buffer,allParts,out_buffer,particles_f4,masses_f4,n_particles,n_batches)

    save_phi_acc[0] = out
    current_acc = out[:,[0,1,2]]

    for step in range(steps):
        current_pos += current_vel/2
        current_vel += current_acc
        current_pos += current_vel/2

        save_vel[step+1] = current_vel
        save_pos[step+1] = current_pos

        out = phi_acc(prog,vao,pos_buffer,allParts,out_buffer,particles_f4,masses_f4,n_particles,n_batches)

        save_phi_acc[step+1] = out
        current_acc = out[:,[0,1,2]]

    vao.release()
    prog.release()
    pos_buffer.release()
    allParts.release()
    eps_buffer.release()
    G_buffer.release()
    fbo.release()
    ctx.release()

    step_labels = pd.DataFrame(np.reshape(np.repeat(np.arange(steps+1),particles.shape[0]),((steps+1)*particles.shape[0],1)),columns=["step"])
    ids = pd.DataFrame(np.reshape(np.array([np.arange(particles.shape[0])] * (steps+1)).flatten(),(particles.shape[0] * (steps+1),1)),columns=["id"])
    save_vel = pd.DataFrame(np.reshape(save_vel,(save_vel.shape[0] * save_vel.shape[1], save_vel.shape[2])),columns=["vx","vy","vz"])
    save_pos = pd.DataFrame(np.reshape(save_pos,(save_pos.shape[0] * save_pos.shape[1], save_pos.shape[2])),columns=["x","y","z"])
    save_phi_acc = pd.DataFrame(np.reshape(save_phi_acc,(save_phi_acc.shape[0] * save_phi_acc.shape[1], save_phi_acc.shape[2])),columns=["ax","ay","az","phi"])

    second = time.perf_counter()

    return pd.concat((step_labels,ids,save_pos,save_vel,save_phi_acc),axis=1),{"eval_time":second-first}

def phi_acc(prog,vao,pos_buffer,allParts,out_buffer,particles,masses,n_particles,n_batches):
    combined = np.concatenate((particles,masses),axis=1).astype("f2").tobytes()
    combined2 = np.concatenate((particles,masses),axis=1).astype("f2").astype("f4").tobytes()    

    out = np.zeros((n_particles,4),dtype=np.float32)

    current_n = 4096
    if n_particles < current_n:
        current_n = n_particles

    prog.__getitem__("n").write(np.array([[4096]]).astype(np.int32).tobytes())
    pos_buffer.write(combined)

    start = 0
    end = n_particles
    for batch in range(n_batches - 1):
        start = batch * 4 * 2
        end = batch * 4 * 2 + (4096) * 4 * 2
        allParts.write(combined2[start:end])
        vao.transform(out_buffer)
        out += np.ndarray((n_particles,4),"f4",out_buffer.read())

    prog.__getitem__("n").write(np.array([[n_particles - (n_batches-1) * 4096]]).astype(np.int32).tobytes())
    allParts.write(combined2[(n_batches-1) * 4096 * 4 * 2:n_particles * 4 * 2])

    vao.transform(out_buffer)

    out += np.ndarray((n_particles,4),"f4",out_buffer.read())

    return out