import os

from Cython.Build import cythonize
from setuptools import Extension, setup


trace = os.environ.get("CYTHON_COVERAGE") == "1"
define_macros = [("CYTHON_TRACE", "1")] if trace else []
extensions = [
    Extension(
        "bitermplus._btm",
        ["src/bitermplus/_btm.pyx"],
        define_macros=define_macros,
    ),
    Extension(
        "bitermplus._metrics",
        ["src/bitermplus/_metrics.pyx"],
        define_macros=define_macros,
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
