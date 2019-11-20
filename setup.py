#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='PytorchLRP',
      version='0.1',
      description='Basic LRP implementation in PyTorch ',
      author='moboehle',
      #author_email='arne.b.binder@gmail.com',
      url='https://github.com/moboehle/Pytorch-LRP',
      packages=find_packages(),
      install_requires=[
            'allennlp',
            'torch',
            'numpy'
      ],
      #packages=['pytorch_lrp']
      #py_modules=['innvestigator', 'inverter_util', 'utils']
     )
