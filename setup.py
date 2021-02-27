from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "bitermplus.btm",
        sources=["src/bitermplus/btm.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']),
    Extension(
        "bitermplus.metrics",
        sources=["src/bitermplus/metrics.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']),
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'embedsignature': True,
            'language_level': 3})
)
