#!/usr/bin/env python

from setuptools import setup

requirements = [
      'abc',
      'datetime',
      'logging',
      'multiply_prior_engine @ https://github.com/QCDIS/prior-engine.git',
      'numpy',
      'typing',
]

__version__ = None
with open('multiply_inference_engine/version.py') as f:
    exec(f.read())

setup(name='multiply-inference-engine',
      version=__version__,
      description='MULTIPLY Inference Engine',
      author='MULTIPLY Team',
      packages=['multiply_inference_engine'],
      requires=requirements
)
