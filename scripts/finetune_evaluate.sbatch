#!/bin/bash
#SBATCH -A uppmax2025-3-5
#SBATCH -p node
#SBATCH -M snowy
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -J indictrans2_evaluate_fullft
#SBATCH --output=/home/tizh1048/MT/indictrans2-assamese/logs/evaluate_full_%j.out
#SBATCH --error=/home/tizh1048/MT/indictrans2-assamese/logs/evaluate_full_%j.err



# 作业信息
echo "=========================================="
echo "作业开始时间: $(date)"
echo "作业 ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 激活环境
source ~/.bashrc
conda activate mt25

# 确保日志目录存在（Slurm 不会自动创建目录）
mkdir -p /home/tizh1048/MT/indictrans2-assamese/logs

# 进入仓库目录（数据与脚本均按此相对路径查找）
cd /home/tizh1048/MT/indictrans2-assamese
echo "Changed to script directory: $(pwd)"

# 检查环境
echo "Python 版本: $(python --version)"
echo "PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 运行全量微调评估
echo "开始全量微调模型简化评估..."
python organized_scripts/ful_finetune_eval_simple.py

echo "=========================================="
echo "评估完成: $(date)"
echo "=========================================="