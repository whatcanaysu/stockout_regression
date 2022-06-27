# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 23:45:12 2022

@author: FDN-Aysu
"""

from setuptools import setup

#######
# Only needed if the version is imported from the library
# import os
# import sys

# base_dir = os.path.dirname(__file__)
# src_dir = os.path.join(base_dir, 'src')
# sys.path.insert(0, src_dir)

# import library_example
#######


# for the version you can either put here a string with the version number
# or you can create a variable called __version__ inside the __init__.py from
# the src/your_library folder that has a string with the version as a string.
setup(
    name='stock_in_out_regression',
    version='0.0.2',
    description='Regression models for analyzing stockout effects of product sales.',
    author='Aysu Demir, Alexandra Malaga',
    author_email='aysu.demir@bse.edu, alexandra.malaga@bse.edu',
    packages=["stock_in_out_regression"],
    setup_requires=['pytest-runner', 'wheel'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)