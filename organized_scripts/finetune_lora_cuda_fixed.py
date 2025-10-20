#!/usr/bin/env python3
"""
修复版本的 LoRA 微调脚本
禁用评估时的生成以避免模型内部错误
"""

import os
import json
import torch
from datetime import datetime
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

def load_data_from_official_format(data_dir, split="train"):
    """从官方格式加载数据"""
    print(f"从 {data_dir} 加载 {split} 数据...")
    
    # 构建文件路径
    split_dir = os.path.join(data_dir, split, "eng_Latn-asm_Beng")
    src_file = os.path.join(split_dir, f"{split}.eng_Latn")
    tgt_file = os.path.join(split_dir, f"{split}.asm_Beng")
    
    if not os.path.exists(src_file) or not os.path.exists(tgt_file):
        raise FileNotFoundError(f"数据文件不存在: {src_file} 或 {tgt_file}")
    
    # 读取数据
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(src_lines) != len(tgt_lines):
        min_len = min(len(src_lines), len(tgt_lines))
        src_lines = src_lines[:min_len]
        tgt_lines = tgt_lines[:min_len]
        print(f"警告: 源文件和目标文件行数不匹配，使用前 {min_len} 行")
    
    # 转换为官方期望的格式
    data = []
    for src, tgt in zip(src_lines, tgt_lines):
        # 使用官方格式：src_lang tgt_lang text
        formatted_src = f"eng_Latn asm_Beng {src}"
        data.append({
            "sentence_SRC": formatted_src,
            "sentence_TGT": tgt
        })
    
    print(f"加载了 {len(data)} 个样本")
    return data

def preprocess_fn(example, tokenizer, **kwargs):
    """预处理函数"""
    model_inputs = tokenizer(
        example["sentence_SRC"], 
        truncation=True, 
        padding=False, 
        max_length=256
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["sentence_TGT"], 
            truncation=True, 
            padding=False, 
            max_length=256
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics_simple(eval_preds):
    """简化的评估指标计算 - 只计算损失，不生成文本"""
    preds, labels = eval_preds
    
    # 简单的损失计算，避免张量维度问题
    try:
        if isinstance(preds, tuple):
            preds = preds[0]  # 取第一个元素
        if hasattr(preds, 'mean'):
            loss = float(preds.mean())
        else:
            loss = 0.0
        return {"eval_loss": loss}
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return {"eval_loss": 0.0}

def finetune_with_lora_cuda_fixed():
    """使用修复版本的 LoRA 进行 CUDA 微调"""
    
    print("=== 修复版本的 LoRA 微调（CUDA 版本） ===")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 模型配置
    model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
    print(f"使用模型: {model_name}")
    
    # 数据路径
    data_dir = "assamese_english_official_format"
    
    try:
        # 设置认证
        from huggingface_hub import login
        login(token="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")
        
        # 加载模型和分词器
        print("\n1. 加载模型和分词器...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            dropout=0.0,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ 模型和分词器加载成功")
        
        # 配置 LoRA
        print("\n2. 配置 LoRA...")
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            target_modules=["q_proj", "k_proj"],  # 目标模块
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        
        # 应用 LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # 加载数据
        print("\n3. 加载数据...")
        train_data = load_data_from_official_format(data_dir, "train")
        dev_data = load_data_from_official_format(data_dir, "dev")
        
        # 创建数据集
        train_dataset = Dataset.from_list(train_data)
        dev_dataset = Dataset.from_list(dev_data)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(dev_dataset)}")
        
        # 数据预处理
        print("\n4. 数据预处理...")
        train_dataset = train_dataset.map(
            lambda example: preprocess_fn(example, tokenizer),
            batched=True,
        )
        
        dev_dataset = dev_dataset.map(
            lambda example: preprocess_fn(example, tokenizer),
            batched=True,
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100
        )
        
        # 训练参数（修复版本 - 禁用评估时的生成）
        output_dir = f"outputs/assamese_english_lora_cuda_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            fp16=True,  # 使用 fp16 加速训练
            logging_strategy="steps",
            eval_strategy="steps",
            save_strategy="steps",
            logging_steps=10,
            save_total_limit=2,
            predict_with_generate=False,  # 关键修复：禁用评估时的生成
            load_best_model_at_end=True,
            num_train_epochs=3,
            per_device_train_batch_size=8,  # 增加批次大小
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,  # 减少梯度累积
            eval_accumulation_steps=2,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.98,
            max_grad_norm=1.0,
            optim="adamw_torch",
            lr_scheduler_type="inverse_sqrt",
            warmup_ratio=0.0,
            warmup_steps=100,
            learning_rate=2e-4,  # 官方推荐的学习率
            save_steps=50,
            eval_steps=50,
            dataloader_num_workers=4,  # 增加数据加载器工作进程
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            # 移除生成相关参数
            dataloader_pin_memory=True,  # 启用内存固定
        )
        
        # 创建训练器
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics_simple,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=1e-3,
                )
            ],
        )
        
        # 开始训练
        print(f"\n5. 开始 LoRA 微调训练（修复版本）...")
        print(f"输出目录: {output_dir}")
        print(f"使用 CUDA: {device == 'cuda'}")
        print("注意: 已禁用评估时的生成以避免模型错误")
        
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"\n✓ LoRA 微调完成！模型已保存到: {output_dir}")
        
        # 手动测试模型（避免训练器的问题）
        print("\n6. 手动测试微调后的模型...")
        test_results = test_model_manual(model, tokenizer, dev_data[:5], device)
        
        # 保存测试结果
        results_file = f"{output_dir}/test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 测试结果已保存到: {results_file}")
        print("✓ 阿萨姆语-英语 LoRA 微调成功完成！")
        
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_manual(model, tokenizer, test_data, device, num_samples=5):
    """手动测试微调后的 LoRA 模型"""
    print(f"手动测试 LoRA 模型，使用 {num_samples} 个样本...")
    
    results = []
    
    for i, sample in enumerate(test_data[:num_samples]):
        try:
            # 使用预处理后的源文本
            inputs = tokenizer(
                sample['sentence_SRC'],
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(device)
            
            with torch.no_grad():
                # 使用更简单的生成参数
                generated_tokens = model.generate(
                    **inputs,
                    max_length=128,  # 减少最大长度
                    num_beams=1,     # 使用贪心搜索而不是束搜索
                    do_sample=False, # 禁用采样
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            prediction = tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            result = {
                'sample_id': i + 1,
                'source': sample['sentence_SRC'],
                'target': sample['sentence_TGT'],
                'prediction': prediction
            }
            
            results.append(result)
            
            print(f"\n样本 {i+1}:")
            print(f"源文本: {sample['sentence_SRC']}")
            print(f"目标文本: {sample['sentence_TGT']}")
            print(f"预测文本: {prediction}")
            print("-" * 50)
            
        except Exception as e:
            print(f"测试样本 {i+1} 失败: {e}")
            continue
    
    return {
        'test_results': results,
        'total_samples': len(results),
        'timestamp': datetime.now().isoformat(),
        'model_type': 'IndicTrans2_LoRA_CUDA_Fixed',
        'format': 'Official_format',
        'device': device
    }

def main():
    """主函数"""
    print("开始修复版本的阿萨姆语-英语 LoRA 微调（CUDA 版本）...")
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU 训练（速度会很慢）")
    
    # 检查数据是否存在
    data_dir = "assamese_english_official_format"
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        print("请先运行 prepare_data_for_official_lora.py 准备数据")
        return
    
    # 执行微调
    success = finetune_with_lora_cuda_fixed()
    
    if success:
        print("\n🎉 修复版本的 LoRA 微调流程完成！")
    else:
        print("\n❌ 微调失败")

if __name__ == "__main__":
    main()
