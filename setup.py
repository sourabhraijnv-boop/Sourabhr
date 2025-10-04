from setuptools import setup, Extension
import sys
import os

# Path to Box2D headers
box2d_include = os.path.join(os.getcwd(), "Box2D")

# Define the extension
box2d_module = Extension(
    name="_Box2D",                     # SWIG module name must start with "_"
    sources=["Box2D/Box2D_wrap.cpp"],  # SWIG-generated source
    include_dirs=[box2d_include],
    language="c++",
)

setup(
    name="Box2D",
    version="0.1",
    ext_modules=[box2d_module],
    py_modules=["Box2D"],  # Python module name
)
