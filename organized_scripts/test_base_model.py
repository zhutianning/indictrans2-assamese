#!/usr/bin/env python3
"""
测试基础模型是否能正常生成翻译
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def test_base_model():
    """测试基础模型"""
    print("=== 测试基础模型 ===")
    
    # 设置认证
    from huggingface_hub import login
    login(token="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")
    
    # 加载基础模型
    model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
    print(f"加载基础模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✓ 基础模型加载成功，使用设备: {device}")
    
    # 测试翻译
    test_sentences = [
        "Hello, how are you?",
        "What is your name?",
        "I am fine, thank you."
    ]
    
    print("\n开始测试基础模型翻译...")
    
    for i, text in enumerate(test_sentences):
        try:
            # 使用官方格式
            formatted_input = f"eng_Latn asm_Beng {text}"
            print(f"\n测试 {i+1}: {text}")
            print(f"格式化输入: {formatted_input}")
            
            # 分词
            inputs = tokenizer(
                formatted_input,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(device)
            
            print(f"输入 tokens: {inputs['input_ids'].shape}")
            
            # 生成
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 解码
            prediction = tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            print(f"✓ 翻译成功: {prediction}")
            
        except Exception as e:
            print(f"✗ 翻译失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n基础模型测试完成")

if __name__ == "__main__":
    test_base_model()
