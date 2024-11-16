#!/bin/bash

wget -O 2NO2best.pth https://www.dropbox.com/s/97qrcry2b0ubtjv/2NO2best.pth?dl=0
python3 ./test.py --save_dir=$1 --test=./2NO2best.pth

