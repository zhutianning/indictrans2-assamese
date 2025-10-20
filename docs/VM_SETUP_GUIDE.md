# 虚拟机环境搭建指南

## 📋 概述

本指南将帮助您在 Windows 上安装 Ubuntu 虚拟机，用于 IndicTrans2 项目的本地开发和调试。

## 🎯 目标

- 在 Windows 上运行 Linux 环境
- 避免 C++ 编译器安装问题
- 快速调试和测试
- 与学校服务器环境保持一致

## 🛠️ 准备工作

### 系统要求
- **Windows 10/11** (64位)
- **内存**: 16GB+ (推荐 32GB)
- **存储**: 100GB+ 可用空间
- **CPU**: 4+ 核心

### 软件准备
1. **VirtualBox** 或 **VMware Workstation**
2. **Ubuntu 20.04 LTS** 或 **22.04 LTS** ISO 镜像

## 📥 安装步骤

### 步骤 1: 安装虚拟化软件

#### 选项 A: VirtualBox (免费)
1. 下载 VirtualBox: https://www.virtualbox.org/
2. 安装 VirtualBox
3. 启用虚拟化功能 (BIOS 设置)

#### 选项 B: VMware Workstation (付费)
1. 下载 VMware Workstation
2. 安装并激活
3. 启用虚拟化功能

### 步骤 2: 下载 Ubuntu ISO
1. 访问 Ubuntu 官网: https://ubuntu.com/download
2. 下载 Ubuntu 20.04 LTS 或 22.04 LTS
3. 选择 Desktop 版本

### 步骤 3: 创建虚拟机

#### VirtualBox 配置
```
名称: Ubuntu-IndicTrans2
类型: Linux
版本: Ubuntu (64-bit)
内存: 16384 MB (16GB)
硬盘: 100 GB (动态分配)
```

#### VMware 配置
```
名称: Ubuntu-IndicTrans2
类型: Linux
版本: Ubuntu 20.04 LTS
内存: 16384 MB (16GB)
硬盘: 100 GB
```

### 步骤 4: 安装 Ubuntu
1. 启动虚拟机
2. 选择 Ubuntu ISO 镜像
3. 按照安装向导完成安装
4. 设置用户名和密码
5. 安装完成后重启

## 🔧 环境配置

### 步骤 1: 系统更新
```bash
sudo apt update && sudo apt upgrade -y
```

### 步骤 2: 安装基础工具
```bash
sudo apt install -y build-essential python3-dev git curl wget vim
```

### 步骤 3: 安装 Python 3.10
```bash
sudo apt install -y python3.10 python3.10-venv python3.10-dev
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
```

### 步骤 4: 创建项目目录
```bash
mkdir -p ~/projects/indictrans2-assamese
cd ~/projects/indictrans2-assamese
```

### 步骤 5: 克隆项目
```bash
git clone <your-git-repo-url> .
git checkout school-server
```

### 步骤 6: 创建虚拟环境
```bash
python3.10 -m venv indictrans2_env
source indictrans2_env/bin/activate
```

### 步骤 7: 安装 Python 依赖
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.28.1 peft accelerate
pip install huggingface_hub sentencepiece
pip install pandas scikit-learn sacrebleu
```

### 步骤 8: 安装 fairseq
```bash
pip install fairseq
```

### 步骤 9: 安装 IndicTransToolkit
```bash
pip install IndicTransToolkit
```

## 🧪 测试环境

### 测试脚本
创建 `test_environment.py`:
```python
#!/usr/bin/env python3
import torch
import transformers
import fairseq
import IndicTransToolkit

print("环境测试开始...")
print(f"PyTorch 版本: {torch.__version__}")
print(f"Transformers 版本: {transformers.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"Fairseq 安装: 成功")
print(f"IndicTransToolkit 安装: 成功")
print("环境测试完成！")
```

运行测试:
```bash
python test_environment.py
```

## 🚀 快速开始

### 激活环境
```bash
cd ~/projects/indictrans2-assamese
source indictrans2_env/bin/activate
```

### 运行数据预处理
```bash
python organized_scripts/preprocess_indictrans2_fixed.py
```

### 运行模型微调
```bash
python organized_scripts/finetune_lora_cuda_fixed.py
```

## 📁 虚拟机项目结构

```
~/projects/indictrans2-assamese/
├── docs/                           # 项目文档
├── organized_scripts/              # 核心脚本
├── data/                          # 数据文件
├── outputs/                       # 训练输出
├── indictrans2_env/               # Python 虚拟环境
├── test_environment.py            # 环境测试脚本
└── README.md                      # 项目说明
```

## 🔄 与服务器同步

### 推送到服务器
```bash
# 在虚拟机上
git add .
git commit -m "Update from VM"
git push origin school-server
```

### 在服务器上拉取
```bash
# 在服务器上
cd ~/projects/indictrans2-assamese
git pull origin school-server
```

## 🛠️ 故障排除

### 常见问题

#### 1. 虚拟机启动慢
- 增加内存分配
- 启用硬件加速
- 关闭不必要的服务

#### 2. 网络连接问题
- 检查网络适配器设置
- 使用 NAT 或桥接模式
- 重启网络服务

#### 3. 性能问题
- 分配更多 CPU 核心
- 增加内存
- 使用 SSD 存储

#### 4. 依赖安装失败
```bash
# 更新包管理器
sudo apt update

# 安装缺失的依赖
sudo apt install -y python3-dev build-essential

# 重新安装
pip install --no-cache-dir <package-name>
```

### 调试命令
```bash
# 检查系统信息
uname -a
free -h
df -h

# 检查 Python 环境
which python3
python3 --version
pip list

# 检查虚拟环境
which python
pip list
```

## 📊 性能优化

### 虚拟机设置
- **内存**: 16GB+ (推荐 32GB)
- **CPU**: 4+ 核心
- **存储**: 100GB+ SSD
- **网络**: 桥接模式

### 系统优化
```bash
# 禁用不必要的服务
sudo systemctl disable snapd
sudo systemctl disable bluetooth

# 优化内存使用
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## 🎯 下一步

1. **完成虚拟机环境搭建**
2. **测试所有功能**
3. **与服务器环境同步**
4. **开始项目开发**

---

**文档状态**: 待执行  
**创建时间**: 2025-10-21  
**维护者**: AI Assistant
