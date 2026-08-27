import os

from Cython.Build import cythonize
from setuptools import Extension, setup

trace = os.environ.get("CYTHON_COVERAGE") == "1"
define_macros = [("CYTHON_TRACE", "1")] if trace else []
# -O3 only; no -ffast-math, so results stay bit-for-bit reproducible.
extra_compile_args = [] if trace else ["-O3"]

extensions = [
    Extension(
        "bitermplus._btm",
        ["src/bitermplus/_btm.pyx"],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "bitermplus._metrics",
        ["src/bitermplus/_metrics.pyx"],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        force=True,
        compiler_directives={
            "language_level": 3,
            "embedsignature": True,
            "linetrace": trace,
        },
    )
)
