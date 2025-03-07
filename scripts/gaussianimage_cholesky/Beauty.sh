#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 100000 
do
CUDA_VISIBLE_DEVICES=6 python train_deform.py -d $data_path \
--data_name UVG/Beauty --model_name GaussianImage_Cholesky --num_points $num_points --iterations 80000
done
