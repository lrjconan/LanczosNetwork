#!/bin/bash

wget http://www.cs.toronto.edu/~rjliao/data/qm8.zip -P data
unzip data/qm8.zip -d data/

cd operators
python build_segment_reduction.py build_ext --inplace
