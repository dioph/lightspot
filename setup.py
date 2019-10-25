import re

from numpy.distutils.core import Extension, setup

with open("README.md", 'r') as f:
    long_description = f.read()

extension = Extension(name="_macula",
                      sources=["spotlight/macula.f90"],
                      extra_compile_args=["-O3"])

version = re.search(
    '^__version__\\s*=\\s*"(.*)"',
    open('spotlight/__init__.py').read(),
    re.M
).group(1)

setup(
    name="spotlight",
    version=version,
    author="Eduardo Nunes",
    author_email="diofanto.nunes@gmail.com",
    license="MIT",
    description="Modelization of light curves from spotted stars",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dioph/spotlight",
    packages=["spotlight"],
    ext_modules=[extension],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ),
)
