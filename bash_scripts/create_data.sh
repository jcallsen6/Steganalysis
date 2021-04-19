#!/bin/bash

source delete_three_quarters.sh $1
source split_dirs.sh $1
python3 ../convertPng.py -data $1 -threads 10
python3 ../addSteg.py -data $1 -threads 5
python3 ../extractLSB.py $1
source combine_dirs.sh $1
source split_ml_dirs.sh $1
