# -*- coding: utf-8 -*-

"""\
TurbineData - Methods to read turbine blade geometry in Ontology format and process it
"""

from setuptools import setup, find_packages

VERSION = "0.0.1"

classifiers = [
    "Development Status :: 3 -Alpha",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: POSIX",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS :: MacOS X",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering :: Physics"
    "Topic :: Scientific/Engineering :: Visualization",
    "Topic :: Utilities",
]

setup(
    name="TurbineData",
    version=VERSION,
    url="https:://github.com/gantech/turbine_data",
    license="Apache License, Version 2.0",
    description="Methods to read wind turbine blade geometry in Ontology format and process it",
    long_description=__doc__,
    platforms="any",
    classifiers=classifiers,
    include_package_data=True,
    #package_dir={'' : 'src'},
    packages=['turbine_data']
)
