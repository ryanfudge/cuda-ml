from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="cuda-ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Ryan Fudge",
    author_email="ryandfudge@gmail.com",
    description="A Python package for CUDA-accelerated machine learning operations",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ryanfudge/cuda-ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
) 