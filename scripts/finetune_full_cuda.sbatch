#!/bin/sh
#SBATCH -A uppmax2025-3-5
#SBATCH -p node
#SBATCH -M snowy
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -J indictrans2_full_ft
#SBATCH --mail-type=END
#SBATCH --output=/home/tizh1048/MT/indictrans2-assamese/logs/%x_%j.out
#SBATCH --error=/home/tizh1048/MT/indictrans2-assamese/logs/%x_%j.err

# 可选：按需资源（与原 sbatch 一致）
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# 激活 Conda 环境（对齐 train.sh）
source ~/.bashrc
conda activate mt25

# 确保日志目录存在（Slurm 不会自动创建目录）
mkdir -p /home/tizh1048/MT/indictrans2-assamese/logs

# 进入代码仓库目录（数据与脚本均按此相对路径查找）
cd /home/tizh1048/MT/indictrans2-assamese
echo "Changed to script directory: $(pwd)"

PROJECT_DIR=/proj/uppmax2025-3-5/nobackup/tizh1048/MT/Group_project
mkdir -p $PROJECT_DIR
DATA_DIR=$PROJECT_DIR/data
OUTPUTS_DIR=$PROJECT_DIR/outputs
mkdir -p $OUTPUTS_DIR

export OUTPUTS_DIR_BASE=$OUTPUTS_DIR 
#export DIRECTION=INDIC_EN   # INDIC_EN

# 运行模型微调
python /home/tizh1048/MT/indictrans2-assamese/organized_scripts/finetune_full_cuda.py