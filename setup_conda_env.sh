#!/bin/bash
# Setup the local Conda environment in ./venv from the `environment.yml` and `environment.local.yml`: the latter is
# used to install tensorflow-metal and python-snappy packages. Note that this will work only on a macOS system.

CONDA_PREFIX=./venv

# Deactivate and remove existing environment
eval "$(conda shell.bash deactivate)"
conda env remove --prefix $CONDA_PREFIX

# Flags to C++ compiler (needed to install python-snappy package)
export CPPFLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib"

# Create new environment
conda env create --prefix $CONDA_PREFIX --file environment.yml
conda env update --prefix $CONDA_PREFIX --file environment.local.yml
