from setuptools import setup, find_packages
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
    url = "http://github.com/aldro61/pyscm"
)