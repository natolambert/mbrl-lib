# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import re
from setuptools import setup, find_packages

install_requires = [line.rstrip() for line in open("requirements.txt", "r")]

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="mbrl_lib",
    version="0.0.1",
    author="luisenp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fairinternal/mbrl-lib",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
