from platform import system
from setuptools import setup, Extension
from Cython.Build import cythonize

extra_link_args = ['-lomp'] if system() == 'Darwin' else ['-fopenmp']
extra_compile_args = ['-Xpreprocessor', '-fopenmp'] if system() == 'Darwin' else ['-fopenmp']

ext_modules = [
    Extension(
        "bitermplus._btm",
        sources=["src/bitermplus/_btm.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args),
    Extension(
        "bitermplus._metrics",
        # include_dirs=[get_include()],
        # library_dirs=[get_include()],
        sources=["src/bitermplus/_metrics.pyx"],
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
