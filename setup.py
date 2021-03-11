from setuptools import setup, Extension
from Cython.Build import cythonize
from platform import system

extra_link_args = ['-lomp'] if system() == 'Darwin' else ['-fopenmp']
extra_compile_args = ['-Xpreprocessor', '-fopenmp']\
    if system() == 'Darwin'\
    else ['-fopenmp']

ext_modules = [
    Extension(
        "bitermplus.btm",
        sources=["src/bitermplus/btm.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args),
    Extension(
        "bitermplus.btm_depr",
        sources=["src/bitermplus/btm_depr.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args),
    Extension(
        "bitermplus.metrics",
        sources=["src/bitermplus/metrics.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args),
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'embedsignature': True,
            'language_level': 3})
)
