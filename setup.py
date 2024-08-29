#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from pybind11.setup_helpers import Pybind11Extension, build_ext
import setuptools

version = re.search(
    '^__version__\\s*=\\s*"(.*)"', open("src/lightspot/__init__.py").read(), re.M
).group(1)

with open("README.md", "r") as f:
    long_description = f.read()

ext_modules = [
    Pybind11Extension(
        "lightspot.macula",
        ["src/lightspot/macula.cpp"],
    ),
]

install_requires = [
    "dynesty >= 1.0",
    "h5py",
    "matplotlib",
    "scipy >= 0.19",
    "ultranest >= 3.0",
]

extras_require = {
    "cuda": [
        "cupy",
        "numba",
    ],
    "docs": [
        "jupyter >= 1.0",
        "myst-nb >= 0.17",
        "numpydoc",
        "sphinx_rtd_theme",
    ],
    "test": [
        "black == 22.3.0",
        "flake8",
        "isort",
        "pytest",
        "pytest-cov",
        "tox",
    ],
}

setup_requires = [
    "setuptools >= 46.0",
    "pybind11 >= 2.6",
    "cython",
]

setuptools.setup(
    name="lightspot",
    version=version,
    author="Eduardo Nunes",
    author_email="dioph@pm.me",
    license="MIT",
    description="Modelization of light curves from spotted stars",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dioph/lightspot",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    install_requires=install_requires,
    extras_require=extras_require,
    setup_requires=setup_requires,
    cmdclass={"build_ext": build_ext},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
)
