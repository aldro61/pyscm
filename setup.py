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
from numpy import get_include as get_numpy_include
from platform import system as get_os_name
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

# Configure the compiler based on the OS
if get_os_name().lower() == "darwin":
    os_compile_flags = ["-mmacosx-version-min=10.9"]
else:
    os_compile_flags = []


# Required for the automatic installation of numpy
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


solver_module = Extension('pyscm._scm_utility',
                          language="c++",
                          sources=['cpp_extensions/utility_python_bindings.cpp',
                                   'cpp_extensions/solver.cpp'],
                          extra_compile_args=["-std=c++0x"] + os_compile_flags)

dependencies = ["numpy", "scipy", "scikit-learn", "six"]

setup(
    name="pyscm",
    version="1.0.0",
    packages=find_packages(),

    cmdclass={'build_ext': build_ext},
    setup_requires=dependencies,
    install_requires=dependencies,

    author="Alexandre Drouin",
    author_email="aldro61@gmail.com",
    maintainer="Alexandre Drouin",
    maintainer_email="aldro61@gmail.com",
    description="The Set Covering Machine algorithm",
    long_description="A fast implementation of the Set Covering Machine algorithm using a dynamic programming algorithm"
                     " to select the rules of greatest utility.",
    license="GPL-3",
    keywords="machine learning binary classification set covering machine rules",
    url="https://github.com/aldro61/pyscm",

    ext_modules=[solver_module],

    test_suite='nose.collector',
    tests_require=['nose']
)
