import re

from setuptools import setup, Extension


class GetPybindInclude(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()


with open("README.md", 'r') as f:
    long_description = f.read()

extension = Extension(name="lightspot.macula",
                      sources=["lightspot/macula.cpp"],
                      include_dirs=[GetPybindInclude(), ],
                      language="c++")

version = re.search(
    '^__version__\\s*=\\s*"(.*)"',
    open('lightspot/__init__.py').read(),
    re.M
).group(1)

install_requires = [
    "dynesty >= 1.0",
    "numpy",
    "pybind11 >= 2.2",
    "scipy >= 0.19"
]

extras_require = {
    "docs": ["jupyter", "numpydoc", "pandoc", "sphinx_rtd_theme"],
    "test": ["pytest", "flake8"]
}

setup(
    name="lightspot",
    version=version,
    author="Eduardo Nunes",
    author_email="dioph@pm.me",
    license="MIT",
    description="Modelization of light curves from spotted stars",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dioph/lightspot",
    packages=["lightspot"],
    ext_modules=[extension],
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
)
