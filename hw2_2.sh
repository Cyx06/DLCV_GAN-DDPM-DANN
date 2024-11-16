#!/bin/bash

wget -O model_78.pth https://www.dropbox.com/s/age8ytycmn24ta7/model_78.pth?dl=0
python3 ./training/p2/test.py --save_dir=$1 --test=./model_78.pth

