#!/bin/bash

wget -O 500_G.pth https://www.dropbox.com/s/jv0t5kcekgbg8z2/500_G.pth?dl=0
python3 ./training/p1/test.py --test=500_G.pth --random_seed=123 --save_test_result_dir=$1
