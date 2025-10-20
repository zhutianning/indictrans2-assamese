#!/bin/bash
# 作业监控脚本
# 用于监控 SLURM 作业状态和日志

echo "=========================================="
echo "IndicTrans2 作业监控"
echo "=========================================="

# 显示当前作业状态
echo "当前作业状态:"
squeue -u $USER

echo ""
echo "最近的作业日志:"
ls -la logs/ | tail -10

echo ""
echo "GPU 使用情况:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "未检测到 NVIDIA GPU"
fi

echo ""
echo "存储使用情况:"
df -h

echo ""
echo "内存使用情况:"
free -h

echo ""
echo "=========================================="
echo "监控完成"
echo "=========================================="
