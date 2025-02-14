#!/usr/bin/env python
from setuptools import setup, find_packages
import pathlib
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(
    name="AutoGS", 
    version="0.1.0", 
    packages=find_packages(), 
    description="Automated Machine Learning for Environmental Data-Driven Genome Prediction", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AIBreeding/AutoGS",
    python_requires='>=3.9',
    install_requires=requirements,
    
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
)
