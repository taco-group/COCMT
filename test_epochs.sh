#!/bin/bash

# 设置输出文件路径
output_file="test_results.txt"

# 获取命令行参数
inference_script=$1
model_dir=$2
fusion_method=$3
epochs=$4
shift 4

# 初始化可选参数
lightning_flag=""
qbm_flag=""
homo_flag=""
range_flag=""

# 解析可选参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --lightning) lightning_flag="--lightning";;
        --qbm) qbm_flag="--qbm";;
        --homo) homo_flag="--homo";;
        --range) range_flag="--range $2"; shift;;  # 解析 --range 参数并获取其值
    esac
    shift
done

# 清空输出文件
echo "" > $output_file

# 将epoch列表转换为数组
IFS=',' read -ra epoch_list <<< "$epochs"

# 循环测试指定的epoch列表
for epoch in "${epoch_list[@]}"
do
    echo "$epoch Epoch" >> $output_file
    echo "Testing epoch $epoch..."
    # 执行传入的inference脚本，并传入其他参数
    python $inference_script --model_dir $model_dir --fusion_method $fusion_method --epoch $epoch $lightning_flag $qbm_flag $homo_flag $range_flag >> $output_file
    echo "" >> $output_file
done

echo "Testing completed. Results are saved in $output_file"

### Examples:
# OPV2V:
# bash test_epochs.sh <inference_script> <model_dir> <fusion_method> <epoch列表> --lightning --qbm --range "102.4,102.4"
# bash test_epochs.sh opencood/logs/cmt_camera_lidar_att_fuse_2024_03_25_10_57_59 intermediate 0,1 --lightning --qbm --range "102.4,102.4"
# V2V4Real:
# bash test_epochs.sh <inference_script> <model_dir> <fusion_method> <epoch列表> --lightning --qbm --homo --range "102.4,102.4"
# bash test_epochs.sh opencood/logs/cmt_lidar_intermediatefusion_v2v4real intermediate 0,1 --lightning --qbm --homo --range "102.4,102.4"
