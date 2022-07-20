import numpy as np
import struct
import moderngl
import time
from scipy import constants
from math import ceil, comb
import pandas as pd

ctx = moderngl.create_context(standalone=True)

fbo = ctx.simple_framebuffer((1,1),dtype="f2")
fbo.use()

prog = ctx.program(

    vertex_shader="""
    #version 330

    in lowp float inp;
    out lowp float oup;

    void main() {
        oup = inp * 2.;
    }

    """,

    varyings=["oup"],

)

a = 2**8*np.ones(10,dtype=np.float16).reshape((10,1))

input_buffer = ctx.buffer(a.tobytes())

output_buffer = ctx.buffer(reserve=4*10)

vao = ctx.vertex_array(prog,[(input_buffer,"1f2","inp")])

vao.transform(output_buffer)

print(np.ndarray((10),"f4",output_buffer.read()))

vao.release()
output_buffer.release()
input_buffer.release()
