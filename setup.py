#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

from setuptools import find_packages, setup

setup(
    name='gensim',
    packages=find_packages(
        include=["gensim"]
    ),
    version='0.5',
    description='Official code to the generative sea-ice model GenSIM',
    author='Tobias Finn',
    license='MIT',
)
