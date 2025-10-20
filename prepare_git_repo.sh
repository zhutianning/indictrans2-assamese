#!/bin/bash
# Git 仓库准备脚本
# 用于准备项目上传到 Git 仓库

echo "=========================================="
echo "准备 Git 仓库..."
echo "=========================================="

# 检查是否在项目根目录
if [ ! -f "docs/PROJECT_SUMMARY.md" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 初始化 Git 仓库（如果还没有）
if [ ! -d ".git" ]; then
    echo "初始化 Git 仓库..."
    git init
    git branch -M main
fi

# 添加所有文件
echo "添加文件到 Git..."
git add .

# 创建提交
echo "创建提交..."
git commit -m "Initial commit: IndicTrans2 Assamese-English translation project

- 完成数据预处理和模型微调
- 解决技术问题和环境配置
- 创建完整的项目文档
- 准备服务器和虚拟机部署脚本"

# 创建 Windows + 虚拟机分支
echo "创建 Windows + 虚拟机分支..."
git checkout -b windows-vm

# 添加 Windows/VM 专用文件
echo "添加 Windows/VM 专用文件..."
git add setup_vm_env.sh
git add debug_model_generation.py
git add debug_model_simple.py
git commit -m "Add Windows and VM environment setup scripts

- Windows environment configuration
- VM setup scripts
- Debug scripts for model generation issues
- Windows-specific troubleshooting"

# 创建服务器分支
echo "创建服务器分支..."
git checkout -b school-server

# 添加服务器专用文件
echo "添加服务器专用文件..."
git add setup_server_env.sh
git add scripts/
git commit -m "Add school server deployment scripts and SLURM job configurations

- SLURM job scripts for preprocessing, training, and evaluation
- Server environment setup
- Job monitoring and submission scripts
- GPU cluster optimization"

# 切换回主分支
git checkout main

echo "=========================================="
echo "Git 仓库准备完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 添加远程仓库: git remote add origin <your-git-repo-url>"
echo "2. 推送主分支: git push -u origin main"
echo "3. 推送 Windows/VM 分支: git push -u origin windows-vm"
echo "4. 推送服务器分支: git push -u origin school-server"
echo ""
echo "分支说明:"
echo "  main: 主分支，包含完整项目和通用文档"
echo "  windows-vm: Windows + 虚拟机专用分支"
echo "    - Windows 环境配置脚本"
echo "    - 虚拟机搭建指南"
echo "    - 调试脚本"
echo "    - Windows 特定问题解决方案"
echo "  school-server: 学校服务器专用分支"
echo "    - SLURM 作业脚本"
echo "    - 服务器环境配置"
echo "    - GPU 集群优化"
echo "    - 作业监控脚本"
echo "=========================================="
