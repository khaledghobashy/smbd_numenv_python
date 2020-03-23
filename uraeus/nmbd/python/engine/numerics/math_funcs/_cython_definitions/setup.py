from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "matrix_funcs",
        ["matrix_funcs.pyx"])
]
setup(name='matrix_funcs',
      ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "3"}),
      include_dirs = [np.get_include()])