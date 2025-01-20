#!/usr/bin/env bash
# Script to setup CPU spack environment for GPRat on simcl1n1-4
set -e
# create environment and copy config file
spack env create gprat_cpu_gcc
cp spack_cpu_gcc.yaml $HOME/spack/var/spack/environments/gprat_cpu_gcc/spack.yaml
spack env activate gprat_cpu_gcc
# find external compiler
module load gcc/13.2.0
spack compiler find
# find external packages
#spack external find
spack external find python
spack external find ninja
# setup environment
spack concretize -f
spack install
