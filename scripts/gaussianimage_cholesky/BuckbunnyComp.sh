#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 50000
do
CUDA_VISIBLE_DEVICES=8 python train_deform2.py -d $data_path \
--data_name bunny --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000
done
 