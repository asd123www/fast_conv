#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J hw1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:05:00

./basic_conv2d

# 需要numpy及pytorch环境
# 该脚本可以比较和pytorch的conv2d结果是否一致，以及给出pytorch的用时作为优化目标参考
# module load anaconda/3-py37
# python baseline_pytorch.py

#### 可使用下面两行命令安装pytorch ####
# module load anaconda/3-py37
# source activate
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu --user #本行为安装pytorch 只需执行一次