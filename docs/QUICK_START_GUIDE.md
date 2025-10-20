# 快速开始指南

## 🚀 环境设置

### 1. 创建虚拟环境
```bash
conda create -n indictrans2_py310 python=3.10 -y
conda activate indictrans2_py310
```

### 2. 安装依赖
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.28.1 peft accelerate
pip install huggingface_hub sentencepiece
pip install pandas scikit-learn sacrebleu
```

### 3. Hugging Face 认证
```bash
python organized_scripts/setup_hf_auth.py
```

## 📊 数据准备

### 1. 数据预处理
```bash
python organized_scripts/preprocess_indictrans2_fixed.py
```

### 2. 官方格式转换
```bash
python organized_scripts/prepare_data_for_official_lora.py
```

## 🎯 模型微调

### 1. 开始训练
```bash
python organized_scripts/finetune_lora_cuda_fixed.py
```

### 2. 监控训练
- 检查 `outputs/` 目录中的训练日志
- 查看 `trainer_state.json` 了解训练进度

## 🔍 模型评估

### 1. 基础测试
```bash
python organized_scripts/test_base_model.py
```

### 2. 模型评估
```bash
python organized_scripts/evaluate_lora_model.py
```

## 📁 重要文件说明

| 文件 | 用途 | 状态 |
|------|------|------|
| `preprocess_indictrans2_fixed.py` | 数据预处理 | ✅ 可用 |
| `finetune_lora_cuda_fixed.py` | LoRA 微调 | ✅ 可用 |
| `prepare_data_for_official_lora.py` | 官方格式数据 | ✅ 可用 |
| `evaluate_lora_model.py` | 模型评估 | ⚠️ 生成错误 |
| `test_base_model.py` | 基础模型测试 | ⚠️ 生成错误 |

## ⚠️ 已知问题

1. **模型生成错误**: 基础模型存在 `AttributeError` 问题
2. **IndicTransToolkit**: 需要 C++ 编译器
3. **评估指标**: 由于生成错误，BLEU/chrF 分数为 0

## 🛠️ 故障排除

### 问题 1: CUDA 内存不足
```bash
# 减少批次大小
per_device_train_batch_size=2
gradient_accumulation_steps=8
```

### 问题 2: 认证失败
```bash
# 重新认证
python organized_scripts/setup_hf_auth.py
```

### 问题 3: 数据格式错误
```bash
# 检查数据格式
head -5 assamese_english_official_format/train/eng_Latn-asm_Beng/train.eng_Latn
```

## 📞 获取帮助

1. 查看 `PROJECT_SUMMARY.md` 了解项目详情
2. 查看 `TECHNICAL_ISSUES_AND_SOLUTIONS.md` 了解技术问题
3. 检查 `cleanup_log.json` 了解项目清理情况

---

**快速开始指南** | **版本**: 1.0 | **更新**: 2025-10-21
