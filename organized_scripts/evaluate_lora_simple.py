#!/usr/bin/env python3
"""
简化的 LoRA 模型评估脚本
使用更简单的生成参数避免模型错误
"""

import os
import json
import torch
import pandas as pd
from datetime import datetime
from sacrebleu.metrics import BLEU, CHRF
from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# 初始化评估指标
bleu_metric = BLEU()
chrf_metric = CHRF()

def load_test_data():
    """加载测试数据"""
    print("加载测试数据...")
    
    data_dir = "assamese_english_official_format"
    test_dir = os.path.join(data_dir, "test", "eng_Latn-asm_Beng")
    src_file = os.path.join(test_dir, "test.eng_Latn")
    tgt_file = os.path.join(test_dir, "test.asm_Beng")
    
    with open(src_file, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f.readlines() if line.strip()]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_sentences = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"加载了 {len(src_sentences)} 个测试样本")
    return src_sentences, tgt_sentences

def load_lora_model(model_path):
    """加载 LoRA 微调后的模型"""
    print(f"加载 LoRA 模型: {model_path}")
    
    # 设置认证
    from huggingface_hub import login
    login(token="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")
    
    # 加载基础模型和分词器
    base_model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # 移动到设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功，使用设备: {device}")
    return model, tokenizer, device

def simple_translate(src_sentences, model, tokenizer, device, max_samples=20):
    """使用简单参数进行翻译"""
    print(f"开始翻译，最多处理 {max_samples} 个样本...")
    
    predictions = []
    successful = 0
    failed = 0
    
    for i, src_text in enumerate(src_sentences[:max_samples]):
        try:
            # 使用官方格式：src_lang tgt_lang text
            formatted_input = f"eng_Latn asm_Beng {src_text}"
            
            # 分词
            inputs = tokenizer(
                formatted_input,
                return_tensors="pt",
                truncation=True,
                max_length=128  # 减少最大长度
            ).to(device)
            
            # 使用最简单的生成参数
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    max_length=64,  # 进一步减少最大长度
                    num_beams=1,    # 使用贪心搜索
                    do_sample=False, # 禁用采样
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # 解码
            prediction = tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            predictions.append(prediction)
            successful += 1
            
            if (i + 1) % 5 == 0:
                print(f"已处理 {i + 1}/{max_samples} 个样本")
            
        except Exception as e:
            print(f"翻译样本 {i+1} 失败: {e}")
            predictions.append("")  # 添加空字符串保持索引一致
            failed += 1
            continue
    
    print(f"翻译完成: {successful} 成功, {failed} 失败")
    return predictions

def compute_metrics(predictions, references):
    """计算评估指标"""
    print("计算评估指标...")
    
    # 过滤掉空的预测
    valid_pairs = [(pred, ref) for pred, ref in zip(predictions, references) if pred.strip()]
    if not valid_pairs:
        print("没有有效的预测结果")
        return {"BLEU": 0.0, "chrF": 0.0}
    
    valid_predictions, valid_references = zip(*valid_pairs)
    
    try:
        bleu_score = bleu_metric.corpus_score(valid_predictions, [valid_references]).score
        chrf_score = chrf_metric.corpus_score(valid_predictions, [valid_references]).score
        
        return {"BLEU": bleu_score, "chrF": chrf_score}
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return {"BLEU": 0.0, "chrF": 0.0}

def evaluate_simple():
    """简化的评估流程"""
    print("=== 简化的 LoRA 模型评估 ===")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 找到最新的模型
    outputs_dir = "outputs"
    model_dirs = [d for d in os.listdir(outputs_dir) if d.startswith("assamese_english_lora_cuda_fixed_")]
    if not model_dirs:
        print("错误: 找不到训练好的 LoRA 模型")
        return
    
    latest_model_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(outputs_dir, latest_model_dir)
    print(f"使用模型: {model_path}")
    
    try:
        # 加载模型和数据
        model, tokenizer, device = load_lora_model(model_path)
        src_sentences, tgt_sentences = load_test_data()
        
        # 生成翻译
        print("\n开始生成翻译...")
        predictions = simple_translate(src_sentences, model, tokenizer, device, max_samples=20)
        
        # 计算指标
        print("\n计算评估指标...")
        metrics = compute_metrics(predictions, tgt_sentences[:len(predictions)])
        
        # 显示结果
        print(f"\n=== 评估结果 ===")
        print(f"BLEU: {metrics['BLEU']:.4f}")
        print(f"chrF: {metrics['chrF']:.4f}")
        
        # 显示样本
        print(f"\n=== 样本结果 ===")
        n_samples = min(5, len(predictions))
        for i in range(n_samples):
            if predictions[i].strip():  # 只显示成功的翻译
                print(f"\n样本 {i+1}:")
                print(f"英语: {src_sentences[i]}")
                print(f"参考翻译: {tgt_sentences[i]}")
                print(f"模型翻译: {predictions[i]}")
                print("-" * 50)
        
        # 保存结果
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'test_samples': len(predictions),
            'metrics': metrics,
            'samples': [
                {
                    'source': src_sentences[i],
                    'reference': tgt_sentences[i],
                    'prediction': predictions[i]
                }
                for i in range(min(10, len(predictions)))
            ]
        }
        
        results_file = f"{model_path}/simple_evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 评估完成！结果已保存到: {results_file}")
        return results
        
    except Exception as e:
        print(f"✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("开始简化的 LoRA 模型评估...")
    
    # 检查数据是否存在
    data_dir = "assamese_english_official_format"
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    
    # 执行评估
    results = evaluate_simple()
    
    if results:
        print("\n🎉 简化评估完成！")
    else:
        print("\n❌ 评估失败")

if __name__ == "__main__":
    main()
