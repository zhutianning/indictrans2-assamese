#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试模型生成问题的脚本
尝试不同的方法来修复 AttributeError: 'NoneType' object has no attribute 'shape'
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
import traceback

def setup_auth():
    """设置 Hugging Face 认证"""
    try:
        login(token="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")
        print("Hugging Face 认证成功")
        return True
    except Exception as e:
        print(f"认证失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    try:
        print("🔄 正在加载模型...")
        model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer 加载成功")
        
        # 加载模型
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ 模型加载成功")
        
        return model, tokenizer
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return None, None

def test_simple_generation(model, tokenizer):
    """测试简单生成"""
    try:
        print("🔄 测试简单生成...")
        
        # 准备输入
        text = "Hello world"
        inputs = tokenizer(text, return_tensors="pt")
        
        # 移动到 GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        print(f"输入 tokens: {inputs['input_ids'][0][:10].tolist()}")
        print(f"输入 shape: {inputs['input_ids'].shape}")
        
        # 尝试生成
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=32,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ 生成成功: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        traceback.print_exc()
        return False

def test_indic_format(model, tokenizer):
    """测试 IndicTrans2 格式"""
    try:
        print("🔄 测试 IndicTrans2 格式...")
        
        # 使用正确的格式
        text = "asm_Beng eng_Latn মই ভাল আছো"
        inputs = tokenizer(text, return_tensors="pt")
        
        # 移动到 GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        print(f"输入 tokens: {inputs['input_ids'][0][:10].tolist()}")
        print(f"输入 shape: {inputs['input_ids'].shape}")
        
        # 尝试生成
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=64,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ IndicTrans2 格式生成成功: {result}")
        return True
        
    except Exception as e:
        print(f"❌ IndicTrans2 格式生成失败: {e}")
        traceback.print_exc()
        return False

def test_different_strategies(model, tokenizer):
    """测试不同的生成策略"""
    strategies = [
        {"name": "Greedy", "max_length": 32, "num_beams": 1, "do_sample": False},
        {"name": "Beam Search", "max_length": 32, "num_beams": 3, "do_sample": False},
        {"name": "Sampling", "max_length": 32, "do_sample": True, "temperature": 0.7},
        {"name": "Top-k", "max_length": 32, "do_sample": True, "top_k": 50},
        {"name": "Top-p", "max_length": 32, "do_sample": True, "top_p": 0.9},
    ]
    
    text = "asm_Beng eng_Latn মই ভাল আছো"
    inputs = tokenizer(text, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda()
    
    for strategy in strategies:
        try:
            print(f"🔄 测试策略: {strategy['name']}")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    **{k: v for k, v in strategy.items() if k != 'name'},
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ {strategy['name']} 成功: {result}")
            
        except Exception as e:
            print(f"❌ {strategy['name']} 失败: {e}")

def test_model_internals(model, tokenizer):
    """测试模型内部状态"""
    try:
        print("🔄 检查模型内部状态...")
        
        text = "asm_Beng eng_Latn মই ভাল আছো"
        inputs = tokenizer(text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        # 检查模型状态
        print(f"模型设备: {model.device}")
        print(f"模型数据类型: {model.dtype}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 尝试前向传播
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"前向传播成功")
            print(f"输出 keys: {outputs.keys()}")
            if hasattr(outputs, 'logits'):
                print(f"Logits shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型内部检查失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始调试模型生成问题")
    print("=" * 50)
    
    # 1. 设置认证
    if not setup_auth():
        return
    
    # 2. 加载模型
    model, tokenizer = test_model_loading()
    if model is None or tokenizer is None:
        return
    
    # 3. 检查模型内部状态
    test_model_internals(model, tokenizer)
    
    # 4. 测试简单生成
    test_simple_generation(model, tokenizer)
    
    # 5. 测试 IndicTrans2 格式
    test_indic_format(model, tokenizer)
    
    # 6. 测试不同策略
    test_different_strategies(model, tokenizer)
    
    print("=" * 50)
    print("🎯 调试完成")

if __name__ == "__main__":
    main()
