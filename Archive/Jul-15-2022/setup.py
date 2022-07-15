from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("treecode.pyx"),
    compiler_directives={'language_level' : "3"}
)
