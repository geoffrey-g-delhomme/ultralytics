#!/bin/sh

conda create -y -p ./.conda python=3.10.9 virtualenv
conda activate ./.conda
python -m venv .venv
. ./.venv/bin/activate
pip install -y -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cu117
pip install .


