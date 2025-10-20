#!/bin/bash
# 虚拟机环境配置脚本
# 用于在 Ubuntu 虚拟机上设置 IndicTrans2 项目环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始配置虚拟机环境..."
echo "=========================================="

# 检查系统信息
echo "系统信息:"
uname -a
echo ""

# 检查存储空间
echo "存储空间:"
df -h
echo ""

# 检查内存
echo "内存信息:"
free -h
echo ""

# 更新系统
echo "更新系统包..."
sudo apt update && sudo apt upgrade -y

# 安装基础工具
echo "安装基础工具..."
sudo apt install -y build-essential python3-dev git curl wget vim

# 安装 Python 3.10
echo "安装 Python 3.10..."
sudo apt install -y python3.10 python3.10-venv python3.10-dev
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 创建项目目录
PROJECT_DIR="$HOME/projects/indictrans2-assamese"
echo "创建项目目录: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 检查是否已有项目
if [ -d ".git" ]; then
    echo "项目已存在，更新代码..."
    git pull origin school-server
else
    echo "请先克隆项目到 $PROJECT_DIR"
    echo "git clone <your-git-repo-url> ."
    echo "git checkout school-server"
    exit 1
fi

# 创建虚拟环境
echo "创建 Python 虚拟环境..."
python3.10 -m venv indictrans2_env
source indictrans2_env/bin/activate

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 安装 PyTorch (CUDA 版本，虚拟机可能没有 GPU)
echo "安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 Transformers 和相关依赖
echo "安装 Transformers 和相关依赖..."
pip install transformers==4.28.1 peft accelerate
pip install huggingface_hub sentencepiece
pip install pandas scikit-learn sacrebleu

# 安装 fairseq
echo "安装 fairseq..."
pip install fairseq

# 安装 IndicTransToolkit
echo "安装 IndicTransToolkit..."
pip install IndicTransToolkit

# 创建必要的目录
echo "创建项目目录结构..."
mkdir -p logs
mkdir -p scripts
mkdir -p data
mkdir -p outputs

# 验证安装
echo "验证安装..."
echo "PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers 版本: $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c "import fairseq" 2>/dev/null; then
    echo "Fairseq: 安装成功"
else
    echo "Fairseq: 安装失败"
fi

if python -c "import IndicTransToolkit" 2>/dev/null; then
    echo "IndicTransToolkit: 安装成功"
else
    echo "IndicTransToolkit: 安装失败"
fi

# 创建环境测试脚本
echo "创建环境测试脚本..."
cat > test_environment.py << 'EOF'
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
EOF

# 运行环境测试
echo "运行环境测试..."
python test_environment.py

echo "=========================================="
echo "虚拟机环境配置完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 激活环境: source $PROJECT_DIR/indictrans2_env/bin/activate"
echo "2. 设置认证: python organized_scripts/setup_hf_auth.py"
echo "3. 运行数据预处理: python organized_scripts/preprocess_indictrans2_fixed.py"
echo "4. 运行模型微调: python organized_scripts/finetune_lora_cuda_fixed.py"
echo ""
echo "项目目录: $PROJECT_DIR"
echo "虚拟环境: $PROJECT_DIR/indictrans2_env"
echo "=========================================="
