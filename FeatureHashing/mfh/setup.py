from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("multi_feature_hashing_cy.pyx"),
    include_dirs=[np.get_include()]
)