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
    "matplotlib",
    "numba",
    "scipy >= 0.19",
]

extras_require = {
    "docs": [
        "jupyter",
        "myst-nb<0.11",
        "numpydoc",
        "sphinx_rtd_theme",
    ],
    "test": [
        "black==20.8b1",
        "flake8",
        "isort",
        "pytest",
        "pytest-cov",
        "tox",
    ],
}

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
    cmdclass={"build_ext": build_ext},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
)
