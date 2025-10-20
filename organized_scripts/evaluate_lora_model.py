#!/usr/bin/env python3
"""
基于官方指南的 LoRA 模型评估脚本
使用测试数据进行 BLEU 和 chrF 评分
"""

import os
import json
import torch
import pandas as pd
from datetime import datetime
from datasets import Dataset
from sacrebleu.metrics import BLEU, CHRF
from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# 初始化评估指标
bleu_metric = BLEU()
chrf_metric = CHRF()

def load_data_from_official_format(data_dir, split="test"):
    """从官方格式加载测试数据"""
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
    
    print(f"加载了 {len(src_lines)} 个测试样本")
    return src_lines, tgt_lines

def load_lora_model(model_path, base_model_name="ai4bharat/indictrans2-indic-en-dist-200M"):
    """加载 LoRA 微调后的模型"""
    print(f"加载基础模型: {base_model_name}")
    
    # 设置认证
    from huggingface_hub import login
    login(token="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")
    
    # 加载基础模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # 加载 LoRA 适配器
    print(f"加载 LoRA 适配器: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # 移动到设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功，使用设备: {device}")
    return model, tokenizer, device

def batch_translate_official(src_sentences, model, tokenizer, device, batch_size=4):
    """使用官方方法进行批量翻译"""
    translations = []
    
    for i in range(0, len(src_sentences), batch_size):
        batch = src_sentences[i : i + batch_size]
        
        # 使用官方格式：src_lang tgt_lang text
        formatted_batch = [f"eng_Latn asm_Beng {text}" for text in batch]
        
        # 分词
        inputs = tokenizer(
            formatted_batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        
        # 生成翻译
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        batch_translations = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        translations.extend(batch_translations)
        
        # 清理内存
        del inputs
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return translations

def compute_metrics_official(predictions, references, metric_dict=None):
    """使用官方方法计算评估指标"""
    if metric_dict is None:
        metric_dict = {"BLEU": bleu_metric, "chrF": chrf_metric}
    
    results = {}
    for metric_name, metric in metric_dict.items():
        try:
            score = metric.corpus_score(predictions, [references]).score
            results[metric_name] = score
            print(f"{metric_name}: {score:.4f}")
        except Exception as e:
            print(f"计算 {metric_name} 时出错: {e}")
            results[metric_name] = 0.0
    
    return results

def evaluate_lora_model():
    """评估 LoRA 微调后的模型"""
    
    print("=== 基于官方指南的 LoRA 模型评估 ===")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 数据路径
    data_dir = "assamese_english_official_format"
    
    # 找到最新的模型路径
    outputs_dir = "outputs"
    model_dirs = [d for d in os.listdir(outputs_dir) if d.startswith("assamese_english_lora_cuda_fixed_")]
    if not model_dirs:
        print("错误: 找不到训练好的 LoRA 模型")
        return
    
    latest_model_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(outputs_dir, latest_model_dir)
    print(f"使用模型: {model_path}")
    
    try:
        # 加载模型
        model, tokenizer, device = load_lora_model(model_path)
        
        # 加载测试数据
        print("\n1. 加载测试数据...")
        src_sentences, tgt_sentences = load_data_from_official_format(data_dir, "test")
        
        # 如果测试数据太多，使用前100个样本进行快速评估
        if len(src_sentences) > 100:
            print(f"测试数据较多({len(src_sentences)}个)，使用前100个样本进行快速评估")
            src_sentences = src_sentences[:100]
            tgt_sentences = tgt_sentences[:100]
        
        print(f"评估样本数: {len(src_sentences)}")
        
        # 生成翻译
        print("\n2. 生成翻译...")
        predictions = batch_translate_official(src_sentences, model, tokenizer, device)
        
        # 计算评估指标
        print("\n3. 计算评估指标...")
        metrics = compute_metrics_official(predictions, tgt_sentences)
        
        # 显示样本结果
        print("\n4. 样本结果:")
        n_samples = min(5, len(src_sentences))
        for i in range(n_samples):
            print(f"\n样本 {i+1}:")
            print(f"英语: {src_sentences[i]}")
            print(f"参考翻译: {tgt_sentences[i]}")
            print(f"模型翻译: {predictions[i]}")
            print("-" * 50)
        
        # 保存评估结果
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'test_samples': len(src_sentences),
            'metrics': metrics,
            'samples': [
                {
                    'source': src_sentences[i],
                    'reference': tgt_sentences[i],
                    'prediction': predictions[i]
                }
                for i in range(min(10, len(src_sentences)))
            ]
        }
        
        results_file = f"{model_path}/evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 评估完成！结果已保存到: {results_file}")
        print(f"\n=== 最终评估结果 ===")
        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.4f}")
        
        return results
        
    except Exception as e:
        print(f"✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("开始基于官方指南的 LoRA 模型评估...")
    
    # 检查数据是否存在
    data_dir = "assamese_english_official_format"
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        print("请先运行 prepare_data_for_official_lora.py 准备数据")
        return
    
    # 执行评估
    results = evaluate_lora_model()
    
    if results:
        print("\n🎉 LoRA 模型评估完成！")
    else:
        print("\n❌ 评估失败")

if __name__ == "__main__":
    main()
