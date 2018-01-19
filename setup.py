#!/usr/bin/env python

from setuptools import setup

requirements = [
      'abc',
      'datetime',
      'logging',
      'multiply_prior_engine',
      'numpy',
      'typing',
]

setup(name='multiply-inference-engine',
      version='0.1.dev1',
      description='MULTIPLY Inference Engine',
      author='MULTIPLY Team',
      packages=['multiply_inference_engine'],
      requires=requirements
)