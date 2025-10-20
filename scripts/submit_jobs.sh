#!/bin/bash
# 作业提交管理脚本
# 用于按顺序提交 SLURM 作业

echo "=========================================="
echo "开始提交 IndicTrans2 作业序列"
echo "=========================================="

# 检查是否在正确的目录
if [ ! -f "scripts/preprocess.sbatch" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 创建日志目录
mkdir -p logs

echo "1. 提交数据预处理作业..."
PREPROCESS_JOB=$(sbatch scripts/preprocess.sbatch | awk '{print $4}')
echo "预处理作业 ID: $PREPROCESS_JOB"

echo "2. 提交模型微调作业 (依赖预处理完成)..."
FINETUNE_JOB=$(sbatch --dependency=afterok:$PREPROCESS_JOB scripts/finetune.sbatch | awk '{print $4}')
echo "微调作业 ID: $FINETUNE_JOB"

echo "3. 提交模型评估作业 (依赖微调完成)..."
EVALUATE_JOB=$(sbatch --dependency=afterok:$FINETUNE_JOB scripts/evaluate.sbatch | awk '{print $4}')
echo "评估作业 ID: $EVALUATE_JOB"

echo "=========================================="
echo "所有作业已提交！"
echo "=========================================="
echo "作业 ID:"
echo "  预处理: $PREPROCESS_JOB"
echo "  微调: $FINETUNE_JOB"
echo "  评估: $EVALUATE_JOB"
echo ""
echo "监控命令:"
echo "  squeue -u \$USER"
echo "  tail -f logs/preprocess_$PREPROCESS_JOB.out"
echo "  tail -f logs/finetune_$FINETUNE_JOB.out"
echo "  tail -f logs/evaluate_$EVALUATE_JOB.out"
echo "=========================================="
