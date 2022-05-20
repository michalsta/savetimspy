# -*- coding: utf-8 -*-
import sys
import os
import os.path
from glob import glob

import setuptools
from distutils.core import setup, Extension
from distutils import sysconfig, spawn



import platform

build_asan = False

# If we're not on Windows, assume something POSIX-compatible (either Linux, OSX, *BSD or Cygwin) with a working gcc-like compiler
windows = platform.system() == 'Windows'

# Dual-build: work-around for the fact that we have both C and C++ files in the extension, and sometimes need
# to split it into two. Windows and CYGWIN for now seems to need dual_build set to False, OSX to True, Linux seems fine
# with either setting.
dual_build = not windows

if platform.system() == "Windows":
    assert not build_asan
    dual_build = False
elif platform.system().startswith("CYGWIN"):
    dual_build = False


native_build = "CIBUILDWHEEL" not in os.environ and 'darwin' not in platform.system().lower() and not 'aarch' in platform.machine().lower()
use_clang = (not windows) and spawn.find_executable('clang++') != None and os.getenv('OPENTIMS_USE_DEFAULT_CXX') == None
#use_ccache = (not windows) and spawn.find_executable('ccache') != None and native_build
use_ccache = os.path.exists("./use_ccache")

# Prefer clang on UNIX if available
if use_clang:
    if use_ccache:
        os.environ['CXX'] = 'ccache g++'
        os.environ['CC'] = 'ccache gcc'
    else:
        os.environ['CXX'] = 'clang++'
else:
    if use_ccache:
        os.environ['CXX'] = 'ccache c++'
        os.environ['CC'] = 'ccache cc'
    else:
        # leave defaults
        pass


def get_cflags(asan=False, warnings=True, std_flag=False):
    if windows:
        return ["/O2"]
    if asan:
        return "-Og -g -std=c++14 -fsanitize=address".split()
    res = ["-g", "-O3"]
    if std_flag:
        res.append("-std=c++14")
    if warnings:
        res.extend(["-Wall", "-Wextra"])
    if native_build:
        res.extend(["-march=native", "-mtune=native"])
    return res

cflags = get_cflags



setup(
    name='savetimspy',
    packages=['savetimspy'],
    version='0.0.4',
    author='Mateusz Krzysztof Łącki (MatteoLacki), Michał Startek (michalsta)',
    author_email='matteo.lacki@gmail.com, michal.startek@mimuw.edu.pl',
    description='opentimspy: An writer of Bruker Tims Data File (.tdf).',
    long_description='opentimspy: A writer of Bruker Tims Data File (.tdf).',
    keywords=['timsTOFpro', 'Bruker TDF', 'data science', 'mass spectrometry', 'rock and roll'],
    classifiers=["Development Status :: 4 - Beta",
             'Intended Audience :: Science/Research',
             'Topic :: Scientific/Engineering :: Chemistry',
             'Programming Language :: Python :: 3.6',
             'Programming Language :: Python :: 3.7',
             'Programming Language :: Python :: 3.8',
             'Programming Language :: Python :: 3.9'],
    zip_safe=False,
    setup_requires=[],
    install_requires=[
        'cffi',
        'zstd',
        'numpy<1.22',#numba requires that
        'opentimspy[bruker_proprietary]',
        'tqdm',
        'dia_common',
        'ncls',# nested containment list
        'pandas',
    ],
    # extras_require={
    #     'tests': ['pytest']
    # },
    # package_dir={"savetimspy":"savetimspy"},
    # package_data={"savetimspy": 
    #     [
    #         "data/test.d/analysis.tdf",
    #         "data/test.d/analysis.tdf_bin",
    #     ]
    # },
    scripts=[
        'scripts/collate_ms2.py',
        # 'scripts/get_frames.py',
        'savetimspy/get_frames.py',
        'scripts/get_midia_groups.py',
        'savetimspy/get_diagonals.py',
        'savetimspy/get_ms2.py',
    ],
)
