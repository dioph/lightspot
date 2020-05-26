import re

from numpy.distutils.core import Extension, setup

with open("README.md", 'r') as f:
    long_description = f.read()

extension = Extension(name="_macula",
                      sources=["lightspot/macula.f90"],
                      extra_compile_args=["-O3"])

version = re.search(
    '^__version__\\s*=\\s*"(.*)"',
    open('lightspot/__init__.py').read(),
    re.M
).group(1)

install_requires = [
    "numpy",
    "scipy >= 0.19",
    "dynesty >= 1.0"
]

extras_require = {
    "docs": ["jupyter", "numpydoc", "pandoc", "sphinx_rtd_theme"],
    "test": ["pytest"]
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
