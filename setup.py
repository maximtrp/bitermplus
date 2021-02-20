#from numpy import get_include as numpy_get_include
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("bitermplus.btm", sources=["bitermplus/btm.pyx"]),
    Extension(
        "bitermplus.metrics",
        sources=["bitermplus/metrics.pyx"],
        libraries=["m"]),
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'embedsignature': True,
            'language_level': 3})
)
