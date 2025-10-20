#!/bin/bash
# 学校服务器环境配置脚本
# 用于在 Linux 服务器上设置 IndicTrans2 项目环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始配置学校服务器环境..."
echo "=========================================="

# 检查系统信息
echo "系统信息:"
uname -a
echo ""

# 检查 GPU 信息
echo "GPU 信息:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "未检测到 NVIDIA GPU"
fi
echo ""

# 检查存储空间
echo "存储空间:"
df -h
echo ""

# 检查内存
echo "内存信息:"
free -h
echo ""

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
python3 -m venv indictrans2_env
source indictrans2_env/bin/activate

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 安装 PyTorch (CUDA 版本)
echo "安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 Transformers 和相关依赖
echo "安装 Transformers 和相关依赖..."
pip install transformers==4.28.1 peft accelerate
pip install huggingface_hub sentencepiece
pip install pandas scikit-learn sacrebleu

# 安装 C++ 编译器和开发工具
echo "安装 C++ 编译器和开发工具..."
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

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

# 设置 Hugging Face 认证
echo "设置 Hugging Face 认证..."
echo "请运行以下命令设置认证:"
echo "python organized_scripts/setup_hf_auth.py"

echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 激活环境: source $PROJECT_DIR/indictrans2_env/bin/activate"
echo "2. 设置认证: python organized_scripts/setup_hf_auth.py"
echo "3. 运行数据预处理: sbatch scripts/preprocess.sbatch"
echo "4. 运行模型微调: sbatch scripts/finetune.sbatch"
echo ""
echo "项目目录: $PROJECT_DIR"
echo "虚拟环境: $PROJECT_DIR/indictrans2_env"
echo "=========================================="
