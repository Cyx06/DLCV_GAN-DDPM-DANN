#!/bin/bash

wget -O bestMtoS.pth https://www.dropbox.com/s/mff61vl5mdl8map/bestMtoS.pth?dl=0
wget -O bestMtoU.pth https://www.dropbox.com/s/9icfmy3hswd4259/bestMtoU.pth?dl=0
usps="usps"
svhn="svhn"
str=$1
if echo ${str} | grep ${usps};then
    python3 predict.py --image_dir=$1  --pre_label_path=$2 --model_path="bestMtoU.pth"
elif echo ${str} | grep ${svhn};then
    python3 predict.py --image_dir=$1  --pre_label_path=$2 --model_path="bestMtoS.pth"
fi

