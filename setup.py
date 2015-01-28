try:
    from numpy import get_include as get_numpy_include
except:
    print "Numpy 1.6.2 or greater is required to install this package. Please install Numpy and try again."
    exit()

from setuptools import setup, find_packages, Extension
setup(
    name = "pyscm",
    version = "0.1",
    packages = find_packages(),

    install_requires = ['numpy'],

    author = "Alexandre Drouin",
    author_email = "aldro61@gmail.com",
    description = "The Set Covering Machine in Python",
    license = "GPLv3",
    keywords = "set covering machine learning feature selection data",
    url = "http://github.com/aldro61/pyscm",

    # Cython Extension
    ext_modules = [Extension("pyscm/binary_attributes/classifications/popcount", ["pyscm/binary_attributes/classifications/popcount.c"], include_dirs=[get_numpy_include()], extra_compile_args=["-march=native"])]
)
