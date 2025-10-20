# 技术问题与解决方案总结

## 🚨 关键问题

### 1. 模型生成错误 (Critical)

**问题描述**:
```
AttributeError: 'NoneType' object has no attribute 'shape'
```

**影响范围**:
- 影响所有模型推理和评估
- 导致 BLEU/chrF 分数为 0
- 无法进行正常的翻译测试

**错误位置**:
- `model.generate()` 调用时
- 在 `modeling_indictrans.py` 内部
- 与 attention 机制相关

**尝试的解决方案**:
1. ✅ **训练时禁用生成**: `predict_with_generate=False`
2. ✅ **简化评估指标**: 仅计算损失，不生成文本
3. ❌ **修改生成参数**: 各种 beam search 和 sampling 策略
4. ❌ **使用不同模型版本**: 问题存在于基础模型

**根本原因**:
- 基础模型 `ai4bharat/indictrans2-indic-en-dist-200M` 内部实现问题
- 可能是 attention 权重或 hidden states 为 None
- 与 IndicTransToolkit 依赖相关

**推荐解决方案**:
1. **安装完整 IndicTransToolkit** (需要 C++ 编译器)
2. **使用 fairseq 推理** (官方推荐)
3. **尝试其他模型版本**

### 2. IndicTransToolkit 安装问题

**问题描述**:
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**影响**:
- 无法使用 `IndicProcessor` 进行预处理
- 无法使用 `IndicDataCollator`
- 可能影响模型生成功能

**解决方案**:
1. ✅ **安装 Visual Studio Build Tools** (已安装)
2. 🔄 **安装 C++ 工作负载** (进行中)
3. **使用 Linux 环境** (推荐)
4. **绕过 IndicProcessor** (临时方案)

### 3. 语言标签格式问题

**问题描述**:
```
AssertionError: Invalid source language tag: <asm_Beng>
```

**解决方案**:
- ✅ **正确格式**: `asm_Beng eng_Latn text`
- ❌ **错误格式**: `<asm_Beng> text`

**实现**:
```python
# 正确的格式化
formatted_input = f"asm_Beng eng_Latn {text}"
```

### 4. Hugging Face 认证问题

**问题描述**:
```
401 Client Error: Unauthorized for url: https://huggingface.co/...
OSError: You are trying to access a gated repo
```

**解决方案**:
```python
from huggingface_hub import login
login(token="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")
```

## 🔧 技术细节

### LoRA 配置优化

**最终配置**:
```python
lora_config = LoraConfig(
    r=16,                    # 低秩维度
    lora_alpha=32,           # 缩放参数
    target_modules=[         # 目标模块
        "q_proj", "v_proj", 
        "k_proj", "o_proj"
    ],
    lora_dropout=0.1,        # Dropout 率
    bias="none",             # 不训练 bias
    task_type="SEQ_2_SEQ_LM" # 任务类型
)
```

**训练参数优化**:
```python
training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=4,    # GPU 内存限制
    gradient_accumulation_steps=4,    # 等效批次大小 16
    num_train_epochs=3,               # 训练轮数
    learning_rate=5e-4,               # LoRA 推荐学习率
    fp16=True,                        # 混合精度训练
    predict_with_generate=False,      # 关键: 避免生成错误
    eval_strategy="steps",            # 评估策略
    eval_steps=50,                    # 评估间隔
    save_steps=50,                    # 保存间隔
)
```

### CUDA 优化

**成功配置**:
- PyTorch 2.5.1+cu121
- CUDA 12.1 支持
- 混合精度训练 (fp16)
- 梯度累积

**性能提升**:
- 训练速度提升 3-5 倍
- 内存使用优化
- 支持更大批次大小

## 📊 数据预处理经验

### 官方格式要求

**目录结构**:
```
assamese_english_official_format/
├── train/eng_Latn-asm_Beng/
│   ├── train.eng_Latn
│   └── train.asm_Beng
├── dev/eng_Latn-asm_Beng/
│   ├── dev.eng_Latn
│   └── dev.asm_Beng
└── test/eng_Latn-asm_Beng/
    ├── test.eng_Latn
    └── test.asm_Beng
```

**数据格式**:
- 每行一个句子
- UTF-8 编码
- 无特殊标记

### 数据分割策略

**比例分配**:
- 训练集: 70% (约 35,000 句)
- 验证集: 15% (约 7,500 句)
- 测试集: 15% (约 7,500 句)
- 迷你训练集: 500 句 (用于快速测试)

**质量控制**:
- 移除空值和异常数据
- 长度过滤 (过短或过长的句子)
- 字符编码检查

## 🎯 成功指标

### 训练成功
- ✅ 模型训练完成 (3 epochs)
- ✅ 损失从 4.5+ 降至 3.6
- ✅ LoRA 适配器生成成功
- ✅ 检查点保存正常

### 技术成功
- ✅ CUDA 加速工作正常
- ✅ 混合精度训练稳定
- ✅ 内存使用优化
- ✅ 错误处理机制

### 待解决问题
- ❌ 模型生成功能
- ❌ 评估指标计算
- ❌ 翻译质量测试

## 🛠️ 调试技巧

### 1. 模型加载调试
```python
# 检查模型状态
print(f"模型设备: {model.device}")
print(f"模型数据类型: {model.dtype}")
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
```

### 2. 输入验证
```python
# 验证输入格式
print(f"输入 tokens: {inputs['input_ids'][0][:10].tolist()}")
print(f"输入 shape: {inputs['input_ids'].shape}")
print(f"注意力掩码: {inputs['attention_mask'][0][:10].tolist()}")
```

### 3. 生成调试
```python
# 尝试不同的生成策略
strategies = [
    {"max_length": 32, "num_beams": 1, "do_sample": False},
    {"max_length": 32, "num_beams": 3, "do_sample": False},
    {"max_length": 32, "do_sample": True, "temperature": 0.7}
]
```

## 📋 环境要求

### 必需组件
- Python 3.10+
- PyTorch 2.5.1+ (CUDA 支持)
- Transformers 4.28.1
- PEFT (LoRA 支持)
- Hugging Face Hub

### 可选组件
- IndicTransToolkit (需要 C++ 编译器)
- fairseq (用于推理)
- sentencepiece (用于分词)

### 硬件要求
- GPU: 推荐 8GB+ VRAM
- RAM: 推荐 16GB+
- 存储: 推荐 50GB+ 可用空间

## 🔄 下一步行动计划

### 短期目标 (1-2 天)
1. 完成 C++ 编译器安装
2. 安装完整的 IndicTransToolkit
3. 测试模型生成功能

### 中期目标 (1 周)
1. 实现完整的评估流程
2. 计算 BLEU/chrF 分数
3. 优化模型性能

### 长期目标 (1 月)
1. 部署生产环境
2. 实现批量翻译
3. 性能优化和监控

---

**文档状态**: 持续更新  
**最后更新**: 2025-10-21  
**维护者**: AI Assistant
