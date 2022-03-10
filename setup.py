#!/usr/bin/env python

from distutils.core import setup

from setuptools import find_packages

setup(name='Matrix',
      version='1.0',
      description='Multiply, transpose, power, find the determinant, minor',
      author='Daniil Kim',
      author_email='l0remipsum91@yandex.ru',
      packages=find_packages(include=['matrix', 'matrix.*']),
      install_requires=[
            'numpy>=1.14.5'
      ]
     )