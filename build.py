# build.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module
extensions = [
    Extension(
        "curvepy.inner_loop",          # The package location name
        ["curvepy/inner_loop.pyx"],    # The source file
        include_dirs=[numpy.get_include()], # Needed for numpy C-headers
    )
]

def build(setup_kwargs):
    """
    This function is called by Poetry to build the extensions.
    """
    setup_kwargs.update({
        "ext_modules": cythonize(extensions, compiler_directives={'language_level': "3"}),
        "include_dirs": [numpy.get_include()]
    })