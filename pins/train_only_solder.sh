#!/usr/bin/env bash

ROOT="$(pwd)/.."

source $ROOT/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ROOT

python train_only_solder.py "$*"
