#!/bin/bash

curl -O data/qm8.zip http://www.cs.toronto.edu/~rjliao/data/qm8.zip
unzip data/qm8.zip -d data/QM8

cd operators
python build_segment_reduction.py build_ext --inplace
