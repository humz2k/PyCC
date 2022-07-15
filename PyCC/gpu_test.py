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

    in vec4 pos;

    in float eps;
    in float G;

    out float phi;
    out vec3 acc;

    uniform myBlock{
        vec4 parts[""" + str(n) + """];
    };

    float d;
    float acc_mul;

    void main() {
        phi = 0.;
        acc[0] = 0.;
        acc[1] = 0.;
        acc[2] = 0.;

        for (int i = 0; i < """ + str(n) + """; ++i){
            d = sqrt(pow(parts[i][0] - pos[0],2) + pow(parts[i][1] - pos[1],2) + pow(parts[i][2] - pos[2],2) + eps);
            if (d != 0){
                phi = phi + (-1) * G * (pos[3] * parts[i][3]) / pow(d,2);
                acc_mul = G * parts[i][3] / pow(d,3);
                acc[0] = acc[0] + (parts[i][0] - pos[0]) * acc_mul;
                acc[1] = acc[0] + (parts[i][1] - pos[1]) * acc_mul;
                acc[2] = acc[0] + (parts[i][2] - pos[2]) * acc_mul;
            }
            
        }

    }

    """

def evaluate(particles, velocities, masses, steps = 0, eps = 0, G = 1,dt = 1):

    ctx = moderngl.create_context(standalone=True)

    fbo = ctx.simple_framebuffer((1,1))
    fbo.use()

    n_particles = particles.shape[0]

    prog = ctx.program(
        vertex_shader=get_prog(n_particles),
        varyings=["acc","phi"],
    )

    pos_buffer = ctx.buffer(reserve=n_particles*4*4)
    allParts = ctx.buffer(reserve=n_particles*4*4)
    eps_buffer = ctx.buffer(np.array([[eps**2]]).astype("f4").tobytes())
    G_buffer = ctx.buffer(np.array([[G]]).astype("f4").tobytes())
    out_buffer = ctx.buffer(reserve=n_particles*4*4)

    particles_f4 = particles.astype("f4")
    masses_f4 = masses.astype("f4").reshape(masses.shape[0],1)

    phi_acc(ctx,prog,pos_buffer,allParts,eps_buffer,G_buffer,out_buffer,particles_f4,masses_f4,n_particles)

    prog.release()
    pos_buffer.release()
    allParts.release()
    eps_buffer.release()
    G_buffer.release()
    fbo.release()
    ctx.release()

def phi_acc(ctx,prog,pos_buffer,allParts,eps_buffer,G_buffer,out_buffer,particles,masses,n_particles):
    combined = np.concatenate((particles,masses),axis=1).tobytes()
    
    print(prog.__getitem__("myBlock")) #.write(combined)
    pos_buffer.write(combined)
    allParts.write(combined)
    allParts.bind_to_uniform_block(0)

    vao = ctx.vertex_array(prog, [(pos_buffer, "4f", "pos"),(eps_buffer, "1f /r", "eps"), (G_buffer, "1f /r", "G")])

    vao.transform(out_buffer)

    out_buffer.read()
    print(np.ndarray((n_particles,4),"f4",out_buffer.read()))

    vao.release()





#second = time.perf_counter()

#temp = np.ones(10,dtype=np.float32).reshape((10,1))
#prog.__getitem__("parts").write(temp.tobytes())

#fbo = ctx.simple_framebuffer((1,1))
#fbo.use()

#outbuffer = ctx.buffer(reserve=10*4)
#vao = ctx.vertex_array(prog,[])

#vao.transform(outbuffer,vertices=10)

#print(np.ndarray((10),"f4",outbuffer.read()))

#outbuffer.release()
#vao.release()
#fbo.release()

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

df = Uniform(100,4000,1)
pos = df.loc[:,["x","y","z"]].to_numpy()
vel = df.loc[:,["vx","vy","vz"]].to_numpy()
mass = df.loc[:,"mass"].to_numpy()

first = time.perf_counter()
evaluate(pos,vel,mass,0,0,constants.G)
second = time.perf_counter()
print(second-first)