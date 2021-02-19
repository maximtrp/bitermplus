#from numpy import get_include as numpy_get_include
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("biterm/cbtm.pyx")  # include_path = [numpy_get_include()]
)
