from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize(["PyCC/direct_sum_double.pyx","PyCC/direct_sum_single.pyx","PyCC/analysis.pyx"]),
    compiler_directives={'language_level' : "3"},
    include_dirs=[np.get_include()]
)
