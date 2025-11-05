# IndicTrans2 阿萨姆语-英语翻译微调项目

## 📋 项目概述

本项目使用 `ai4bharat/indictrans2-indic-en-dist-200M` 模型对阿萨姆语到英语的翻译进行微调。项目基于 WMT 2025 Indic MT Task 数据集，分别使用 LoRA (Low-Rank Adaptation) 进行高效微调和全量参数训练。

## 🎯 项目目标

- **数据集**: WMT 2025 Indic MT Task - English-Assamese Training Data 2025.csv
- **模型**: ai4bharat/indictrans2-indic-en-dist-200M (Hugging Face)
- **任务**: 阿萨姆语 (asm_Beng) → 英语 (eng_Latn) 翻译
- **方法**: LoRA 微调
- **数据分割**: 50,000 训练 + 2,000 验证 + 2,000 测试 + 500 迷你训练集

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

### 配置设置

在使用项目之前，请阅读 `CONFIG_SETUP.md` 文件来设置 Hugging Face 认证。

### 本地开发 (Windows + 虚拟机)

1. **克隆项目**
   ```bash
   git clone https://github.com/SeanSha/indictrans2-assamese
   cd indictrans2-assamese
   git checkout windows-vm
   ```

2. **设置虚拟机环境**
   ```bash
   chmod +x setup_vm_env.sh
   ./setup_vm_env.sh
   ```

3. **激活环境**
   ```bash
   source ~/projects/indictrans2-assamese/indictrans2_env/bin/activate
   ```

4. **设置认证**
   ```bash
   export HF_TOKEN="your_token_here"
   ```

5. **运行数据预处理**
   ```bash
   python organized_scripts/preprocess_indictrans2_fixed.py
   ```

6. **分别运行lora模型&全量微调**
   ```bash
   python organized_scripts/finetune_lora_cuda_fixed.py
   python organized_scripts/finetune_full_cuda.py  
   ```

### 服务器部署

1. **克隆项目**
   ```bash
   git clone https://github.com/SeanSha/indictrans2-assamese
   cd indictrans2-assamese
   git checkout school-server
   ```

2. **设置服务器环境**
   ```bash
   chmod +x setup_server_env.sh
   ./setup_server_env.sh
   ```

3. **提交作业**
   ```bash
   chmod +x scripts/submit_jobs.sh
   ./scripts/submit_jobs.sh
   ```

4. **监控作业**
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
- [x] 评估指标计算 (依赖推理修复)

### ❌ 待解决
- [ ] 模型生成错误修复
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

## 📞 获取帮助

1. 查看 `docs/PROJECT_SUMMARY.md` 了解项目详情
2. 查看 `docs/TECHNICAL_ISSUES_AND_SOLUTIONS.md` 了解技术问题
3. 查看 `docs/QUICK_START_GUIDE.md` 快速开始
4. 查看 `docs/SCHOOL_SERVER_MIGRATION_PLAN.md` 服务器部署
5. 查看 `CONFIG_SETUP.md` 配置设置

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
