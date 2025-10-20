#!/bin/bash
#SBATCH --job-name=indictrans2_evaluate
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# 作业信息
echo "=========================================="
echo "作业开始时间: $(date)"
echo "作业 ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 激活环境
source ~/projects/indictrans2-assamese/indictrans2_env/bin/activate

# 创建日志目录
mkdir -p logs

# 切换到项目目录
cd ~/projects/indictrans2-assamese

# 检查环境
echo "Python 版本: $(python --version)"
echo "PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 运行模型评估
echo "开始模型评估..."
python organized_scripts/evaluate_lora_model.py

echo "开始简化评估..."
python organized_scripts/evaluate_lora_simple.py

echo "=========================================="
echo "模型评估完成: $(date)"
echo "=========================================="
