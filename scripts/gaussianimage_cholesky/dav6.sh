#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

# 创建日志文件
log_file="training_log_cow.txt"
echo "Training Log - $(date)" > "$log_file"

find "$data_path/cows" -mindepth 1 -maxdepth 1 -type d -name "cows*" | sort | while read -r segment; do
    echo "Training on segment: $segment" | tee -a "$log_file"

    for num_points in 40000; do
        # 临时文件保存完整输出
        temp_log="temp_${RANDOM}.log"

        # 启动训练进程，将输出保存在临时文件，同时显示进度条
        CUDA_VISIBLE_DEVICES=4 python train_deform.py -d "$segment" \
            --data_name UVG2/$(basename "$segment") --model_name GaussianImage_Cholesky --num_points "$num_points" --iterations 130000 \
            2>&1 | tee "$temp_log"

        # 提取 PSNR 和 SSIM
        psnr=$(grep -oP "PSNR: \K[0-9.]+" "$temp_log" | tail -n 1)
        ssim=$(grep -oP "SSIM: \K[0-9.]+" "$temp_log" | tail -n 1)

        # 将结果写入日志
        echo "Segment: $(basename "$segment"), PSNR: $psnr, SSIM: $ssim" | tee -a "$log_file"

        # 删除临时日志文件
        rm -f "$temp_log"
    done
done