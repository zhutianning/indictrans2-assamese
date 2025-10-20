# IndicTrans2 阿萨姆语-英语翻译微调项目

## 📋 项目概述

本项目使用 `ai4bharat/indictrans2-indic-en-dist-200M` 模型对阿萨姆语到英语的翻译进行微调。项目基于 WMT 2025 Indic MT Task 数据集，使用 LoRA (Low-Rank Adaptation) 技术进行高效微调。

## 🎯 项目目标

- **数据集**: WMT 2025 Indic MT Task - English-Assamese Training Data 2025.csv
- **模型**: ai4bharat/indictrans2-indic-en-dist-200M (Hugging Face)
- **任务**: 阿萨姆语 (asm_Beng) → 英语 (eng_Latn) 翻译
- **方法**: LoRA 微调
- **数据分割**: 50,000 训练 + 2,000 验证 + 2,000 测试 + 500 迷你训练集

## 📁 项目结构

```
project/
├── docs/                           # 项目文档
│   ├── PROJECT_SUMMARY.md          # 项目总结
│   ├── TECHNICAL_ISSUES_AND_SOLUTIONS.md  # 技术问题解决方案
│   ├── QUICK_START_GUIDE.md        # 快速开始指南
│   ├── PROJECT_STATUS_REPORT.md    # 项目状态报告
│   ├── SCHOOL_SERVER_MIGRATION_PLAN.md  # 服务器迁移计划
│   └── VM_SETUP_GUIDE.md           # 虚拟机搭建指南
├── organized_scripts/              # 核心脚本
│   ├── preprocess_indictrans2_fixed.py  # 数据预处理
│   ├── finetune_lora_cuda_fixed.py     # LoRA 微调
│   ├── prepare_data_for_official_lora.py # 官方格式数据准备
│   ├── evaluate_lora_model.py      # 模型评估
│   └── setup_hf_auth.py            # Hugging Face 认证
├── scripts/                        # SLURM 作业脚本
│   ├── preprocess.sbatch           # 数据预处理作业
│   ├── finetune.sbatch             # 模型微调作业
│   ├── evaluate.sbatch             # 模型评估作业
│   ├── submit_jobs.sh              # 作业提交脚本
│   └── monitor_jobs.sh             # 作业监控脚本
├── data/                          # 数据文件
│   ├── as-eng_split/              # 分割后的CSV数据
│   └── assamese_english_official_format/  # 官方格式数据
├── outputs/                       # 训练输出
├── downloads/                     # 原始数据集
├── IndicTrans2/                   # 官方项目（未修改）
├── setup_server_env.sh            # 服务器环境配置
├── setup_vm_env.sh                # 虚拟机环境配置
├── prepare_git_repo.sh            # Git 仓库准备
└── README.md                      # 项目说明
```

## 🌿 分支策略

本项目采用多分支策略，针对不同环境优化：

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

## 🚀 快速开始

### 环境要求

- **Python**: 3.10+
- **PyTorch**: 2.5.1+ (CUDA 支持)
- **Transformers**: 4.28.1
- **PEFT**: LoRA 微调
- **GPU**: 推荐 8GB+ VRAM

### 本地开发 (Windows + 虚拟机)

1. **设置虚拟机环境**
   ```bash
   chmod +x setup_vm_env.sh
   ./setup_vm_env.sh
   ```

2. **激活环境**
   ```bash
   source ~/projects/indictrans2-assamese/indictrans2_env/bin/activate
   ```

3. **设置认证**
   ```bash
   python organized_scripts/setup_hf_auth.py
   ```

4. **运行数据预处理**
   ```bash
   python organized_scripts/preprocess_indictrans2_fixed.py
   ```

5. **运行模型微调**
   ```bash
   python organized_scripts/finetune_lora_cuda_fixed.py
   ```

### 服务器部署

1. **设置服务器环境**
   ```bash
   chmod +x setup_server_env.sh
   ./setup_server_env.sh
   ```

2. **提交作业**
   ```bash
   chmod +x scripts/submit_jobs.sh
   ./scripts/submit_jobs.sh
   ```

3. **监控作业**
   ```bash
   chmod +x scripts/monitor_jobs.sh
   ./scripts/monitor_jobs.sh
   ```

## 📊 项目状态

### ✅ 已完成
- [x] 数据预处理和格式转换
- [x] LoRA 微调实现
- [x] CUDA 加速训练
- [x] 项目文档化
- [x] 服务器部署脚本
- [x] 虚拟机环境配置

### ⚠️ 部分完成
- [x] 模型训练 (成功)
- [ ] 模型推理 (存在生成错误)
- [ ] 评估指标计算 (依赖推理修复)

### ❌ 待解决
- [ ] 模型生成错误修复
- [ ] BLEU/chrF 分数计算
- [ ] 生产环境部署

## 🔧 技术栈

- **Python**: 3.10
- **PyTorch**: 2.5.1+cu121 (CUDA 支持)
- **Transformers**: 4.28.1
- **PEFT**: LoRA 微调
- **Hugging Face Hub**: 模型和数据集访问
- **CUDA**: GPU 加速训练
- **SLURM**: 作业调度 (服务器)

## 📈 训练结果

### 成功指标
- **训练完成**: ✅ 成功完成 3 个 epoch
- **损失下降**: 从 4.5+ 降至 3.6
- **模型保存**: ✅ 保存了多个检查点
- **LoRA 适配器**: ✅ 成功生成适配器文件

### 输出文件
```
outputs/assamese_english_lora_cuda_fixed_20251021_005208/
├── adapter_config.json          # LoRA 配置
├── adapter_model.safetensors    # LoRA 权重
├── checkpoint-*/                # 训练检查点
├── trainer_state.json          # 训练状态
└── simple_evaluation_results.json # 评估结果
```

## 🚨 已知问题

### 1. 模型生成错误 (Critical)
**问题**: `AttributeError: 'NoneType' object has no attribute 'shape'`
- **影响**: 影响所有模型推理和评估
- **原因**: 基础模型内部实现问题
- **解决方案**: 需要安装完整的 IndicTransToolkit 或使用 fairseq

### 2. 环境依赖问题
**问题**: 需要 C++ 编译器
- **影响**: 无法安装 fairseq 和 IndicTransToolkit
- **解决方案**: 使用 Linux 环境或安装 Visual Studio Build Tools

## 🛠️ 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```bash
   # 减少批次大小
   per_device_train_batch_size=2
   gradient_accumulation_steps=8
   ```

2. **认证失败**
   ```bash
   # 重新认证
   python organized_scripts/setup_hf_auth.py
   ```

3. **数据格式错误**
   ```bash
   # 检查数据格式
   head -5 assamese_english_official_format/train/eng_Latn-asm_Beng/train.eng_Latn
   ```

## 📞 获取帮助

1. 查看 `docs/PROJECT_SUMMARY.md` 了解项目详情
2. 查看 `docs/TECHNICAL_ISSUES_AND_SOLUTIONS.md` 了解技术问题
3. 查看 `docs/QUICK_START_GUIDE.md` 快速开始
4. 查看 `docs/SCHOOL_SERVER_MIGRATION_PLAN.md` 服务器部署

## 🔗 相关资源

- **IndicTrans2 官方仓库**: https://github.com/AI4Bharat/IndicTrans2
- **Hugging Face 模型**: https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
- **WMT 2025 数据集**: WMT_INDIC_MT_Task_2025
- **LoRA 论文**: Low-Rank Adaptation of Large Language Models

## 📝 许可证

本项目遵循 MIT 许可证。

---

**项目状态**: 微调成功，推理待解决  
**最后更新**: 2025-10-21  
**维护者**: AI Assistant  
**版本**: 1.0
