# 学校服务器迁移和部署计划

## 📋 项目概述

**目标**: 将当前 Windows 环境下的 IndicTrans2 阿萨姆语-英语翻译微调项目迁移到学校 Linux 服务器，并建立完整的开发工作流。

**服务器配置**:
- **存储空间**: 33GB
- **GPU**: 14GB UPPMAX 单点 GPU
- **系统**: Linux
- **作业调度**: SLURM (sbatch)

## 🎯 迁移计划

### 阶段 1: 项目文档化和 Git 上传

#### 1.1 完善项目文档
- [x] 项目总结文档 (`PROJECT_SUMMARY.md`)
- [x] 技术问题解决方案 (`TECHNICAL_ISSUES_AND_SOLUTIONS.md`)
- [x] 快速开始指南 (`QUICK_START_GUIDE.md`)
- [x] 项目状态报告 (`PROJECT_STATUS_REPORT.md`)
- [ ] 学校服务器部署指南 (本文档)
- [ ] 虚拟机环境搭建指南

#### 1.2 Git 仓库准备
```bash
# 在 Windows 上
git init
git add .
git commit -m "Initial commit: IndicTrans2 Assamese-English translation project"
git branch -M main
git remote add origin <your-git-repo-url>
git push -u origin main
```

#### 1.3 创建专用分支
```bash
# 创建 Windows + 虚拟机分支
git checkout -b windows-vm
git push -u origin windows-vm

# 创建服务器专用分支
git checkout -b school-server
git push -u origin school-server
```

#### 1.4 分支策略说明
- **`main`**: 主分支，包含完整项目和通用文档
- **`windows-vm`**: Windows + 虚拟机专用分支
  - Windows 环境配置脚本
  - 虚拟机搭建指南
  - 调试脚本
  - Windows 特定问题解决方案
- **`school-server`**: 学校服务器专用分支
  - SLURM 作业脚本
  - 服务器环境配置
  - GPU 集群优化
  - 作业监控脚本

### 阶段 2: 学校服务器环境搭建

#### 2.1 服务器环境检查
```bash
# 检查系统信息
uname -a
nvidia-smi
df -h
free -h

# 检查 Python 环境
python3 --version
pip3 --version
```

#### 2.2 创建项目目录
```bash
# 在服务器上创建项目目录
mkdir -p ~/projects/indictrans2-assamese
cd ~/projects/indictrans2-assamese
```

#### 2.3 克隆项目
```bash
# 克隆项目到服务器
git clone <your-git-repo-url> .
git checkout school-server
```

#### 2.4 环境配置脚本
创建 `setup_server_env.sh`:
```bash
#!/bin/bash
# 学校服务器环境配置脚本

echo "开始配置学校服务器环境..."

# 1. 创建虚拟环境
python3 -m venv indictrans2_env
source indictrans2_env/bin/activate

# 2. 升级 pip
pip install --upgrade pip

# 3. 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.28.1 peft accelerate
pip install huggingface_hub sentencepiece
pip install pandas scikit-learn sacrebleu

# 4. 安装 C++ 编译器和 fairseq
sudo apt-get update
sudo apt-get install -y build-essential python3-dev
pip install fairseq

# 5. 安装 IndicTransToolkit
pip install IndicTransToolkit

# 6. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import fairseq; print('Fairseq installed successfully')"

echo "环境配置完成！"
```

### 阶段 3: SLURM 作业脚本

#### 3.1 数据预处理作业脚本
创建 `scripts/preprocess.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=indictrans2_preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# 激活环境
source ~/projects/indictrans2-assamese/indictrans2_env/bin/activate

# 创建日志目录
mkdir -p logs

# 运行数据预处理
cd ~/projects/indictrans2-assamese
python organized_scripts/preprocess_indictrans2_fixed.py
python organized_scripts/prepare_data_for_official_lora.py

echo "数据预处理完成"
```

#### 3.2 模型微调作业脚本
创建 `scripts/finetune.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=indictrans2_finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# 激活环境
source ~/projects/indictrans2-assamese/indictrans2_env/bin/activate

# 创建日志目录
mkdir -p logs

# 运行模型微调
cd ~/projects/indictrans2-assamese
python organized_scripts/finetune_lora_cuda_fixed.py

echo "模型微调完成"
```

#### 3.3 模型评估作业脚本
创建 `scripts/evaluate.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=indictrans2_evaluate
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# 激活环境
source ~/projects/indictrans2-assamese/indictrans2_env/bin/activate

# 创建日志目录
mkdir -p logs

# 运行模型评估
cd ~/projects/indictrans2-assamese
python organized_scripts/evaluate_lora_model.py

echo "模型评估完成"
```

### 阶段 4: 工作流管理

#### 4.1 作业提交脚本
创建 `submit_jobs.sh`:
```bash
#!/bin/bash
# 作业提交管理脚本

echo "提交数据预处理作业..."
PREPROCESS_JOB=$(sbatch scripts/preprocess.sbatch | awk '{print $4}')
echo "预处理作业 ID: $PREPROCESS_JOB"

echo "等待预处理完成..."
sbatch --dependency=afterok:$PREPROCESS_JOB scripts/finetune.sbatch

echo "提交微调作业..."
FINETUNE_JOB=$(sbatch --dependency=afterok:$PREPROCESS_JOB scripts/finetune.sbatch | awk '{print $4}')
echo "微调作业 ID: $FINETUNE_JOB"

echo "提交评估作业..."
sbatch --dependency=afterok:$FINETUNE_JOB scripts/evaluate.sbatch

echo "所有作业已提交！"
```

#### 4.2 监控脚本
创建 `monitor_jobs.sh`:
```bash
#!/bin/bash
# 作业监控脚本

echo "当前作业状态:"
squeue -u $USER

echo "最近的作业日志:"
ls -la logs/ | tail -10

echo "GPU 使用情况:"
nvidia-smi
```

### 阶段 5: 虚拟机环境搭建

#### 5.1 虚拟机配置
- **系统**: Ubuntu 20.04 LTS 或 22.04 LTS
- **内存**: 16GB+ (推荐 32GB)
- **存储**: 100GB+ 可用空间
- **CPU**: 4+ 核心

#### 5.2 虚拟机环境脚本
创建 `setup_vm_env.sh`:
```bash
#!/bin/bash
# 虚拟机环境配置脚本

echo "开始配置虚拟机环境..."

# 1. 更新系统
sudo apt update && sudo apt upgrade -y

# 2. 安装基础工具
sudo apt install -y build-essential python3-dev git curl wget

# 3. 安装 Python 3.10
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# 4. 创建项目目录
mkdir -p ~/projects/indictrans2-assamese
cd ~/projects/indictrans2-assamese

# 5. 克隆项目
git clone <your-git-repo-url> .
git checkout school-server

# 6. 创建虚拟环境
python3.10 -m venv indictrans2_env
source indictrans2_env/bin/activate

# 7. 安装依赖
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.28.1 peft accelerate
pip install huggingface_hub sentencepiece
pip install pandas scikit-learn sacrebleu
pip install fairseq
pip install IndicTransToolkit

echo "虚拟机环境配置完成！"
```

## 📁 项目结构

```
indictrans2-assamese/
├── docs/                           # 项目文档
│   ├── PROJECT_SUMMARY.md
│   ├── TECHNICAL_ISSUES_AND_SOLUTIONS.md
│   ├── QUICK_START_GUIDE.md
│   ├── PROJECT_STATUS_REPORT.md
│   └── SCHOOL_SERVER_MIGRATION_PLAN.md
├── organized_scripts/              # 核心脚本
├── scripts/                        # SLURM 作业脚本
│   ├── preprocess.sbatch
│   ├── finetune.sbatch
│   ├── evaluate.sbatch
│   ├── submit_jobs.sh
│   └── monitor_jobs.sh
├── data/                          # 数据文件
├── outputs/                       # 训练输出
├── logs/                          # 作业日志
├── setup_server_env.sh            # 服务器环境配置
├── setup_vm_env.sh                # 虚拟机环境配置
└── README.md                      # 项目说明
```

## 🚀 执行步骤

### 步骤 1: 完善文档并上传 Git
1. 完善所有项目文档
2. 创建 Git 仓库并上传
3. 创建 `school-server` 分支

### 步骤 2: 学校服务器部署
1. 登录服务器
2. 运行 `setup_server_env.sh`
3. 提交 SLURM 作业进行测试

### 步骤 3: 虚拟机环境搭建
1. 安装 Ubuntu 虚拟机
2. 运行 `setup_vm_env.sh`
3. 测试所有功能

### 步骤 4: 工作流验证
1. 在虚拟机上测试脚本
2. 在服务器上提交作业
3. 验证完整流程

## 📊 预期结果

### 服务器环境优势
- ✅ 解决所有 C++ 编译器问题
- ✅ 正常安装 fairseq 和 IndicTransToolkit
- ✅ 解决模型生成错误
- ✅ 获得更好的 GPU 性能
- ✅ 支持长时间训练任务

### 开发工作流
- ✅ 本地虚拟机快速调试
- ✅ 服务器高性能训练
- ✅ Git 版本控制
- ✅ 自动化作业调度

## 🔧 故障排除

### 常见问题
1. **权限问题**: 确保脚本有执行权限 `chmod +x *.sh`
2. **路径问题**: 检查所有路径是否正确
3. **依赖问题**: 确保所有依赖都已安装
4. **GPU 问题**: 检查 GPU 是否可用 `nvidia-smi`

### 调试命令
```bash
# 检查作业状态
squeue -u $USER

# 查看作业日志
tail -f logs/finetune_*.out

# 检查 GPU 使用
nvidia-smi

# 检查环境
which python
pip list
```

---

**文档状态**: 待执行  
**创建时间**: 2025-10-21  
**维护者**: AI Assistant
