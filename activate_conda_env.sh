#!/bin/bash
# Script to activate the Conda environment in ./venv. The environment variable $INTERACTIVE controls if the env will be
# activated in an interactive shell or not. You can pass '--interactive' as a flag to enable interactive mode.

INTERACTIVE=true

# Parse command line options
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --interactive)
            INTERACTIVE=true
            shift # past argument
            ;;
        *)    # unknown option
            break
            ;;
    esac
done

CONDA_PREFIX=./venv
eval "$(conda shell.bash activate "$CONDA_PREFIX")"

if [ "$INTERACTIVE" = true ]; then
  export PS1="$PS1"
  /bin/bash
fi
