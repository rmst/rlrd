from setuptools import setup
from setuptools import find_packages
from os.path import join, dirname
import sys

if sys.version_info < (3, 7):
    sys.exit('Sorry, Python < 3.7 is not supported. We use dataclasses that have been introduced in 3.7.')

setup(
    name='rlrd',
    version="0.1",
    description='',
    author='Yann Bouteiller and Simon Ramstedt',
    author_email='simonramstedt@gmail.com',
    download_url='',
    license='MIT',
    install_requires=[
        'numpy',
        'torch',
        'imageio',
        'imageio-ffmpeg',
        'pandas',
        'gym',
        'pyyaml',
        'wandb'
    ],
    extras_require={

    },
    scripts=[],
    packages=find_packages()
)
