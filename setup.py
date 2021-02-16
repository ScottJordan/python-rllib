# from sys import platform as _platform

from setuptools import setup, find_packages
setup(
    name="rllib",
    version="0.0.1",
    description=("rl library"),
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "cloudpickle",
        "gym"
    ],
)

