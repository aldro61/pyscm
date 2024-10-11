"""
    pyscm -- The Set Covering Machine in Python
    Copyright (C) 2017 Alexandre Drouin

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import numpy
from platform import system as get_os_name
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

# Configure the compiler based on the OS
if get_os_name().lower() == "darwin":
    os_compile_flags = ["-mmacosx-version-min=10.9"]
else:
    os_compile_flags = []


solver_module = Extension(
    "pyscm._scm_utility",
    language="c++",
    sources=["cpp_extensions/utility_python_bindings.cpp", "cpp_extensions/solver.cpp"],
    extra_compile_args=["-std=c++0x"] + os_compile_flags,
    include_dirs=[numpy.get_include()]
)

dependencies = ["numpy<2", "scikit-learn", "six"]

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="pyscm-ml",
    version="1.1.2",
    packages=find_packages(),
    install_requires=dependencies,
    author="Alexandre Drouin",
    author_email="aldro61@gmail.com",
    maintainer="Alexandre Drouin",
    maintainer_email="aldro61@gmail.com",
    description="The Set Covering Machine algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL-3",
    keywords="machine-learning classification set-covering-machine rule-based-models",
    url="https://github.com/aldro61/pyscm",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=[solver_module],
)
