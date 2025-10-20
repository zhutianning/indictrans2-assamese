#!/usr/bin/env python3
"""
设置 Hugging Face 认证
"""

import os
from huggingface_hub import login

def setup_huggingface_auth():
    """设置 Hugging Face 认证"""
    print("=== 设置 Hugging Face 认证 ===")
    
    # 方法1: 使用环境变量
    hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    if hf_token:
        print("找到环境变量中的 Hugging Face Token")
        login(token=hf_token)
        return True
    
    # 方法2: 手动输入
    print("请输入您的 Hugging Face Token:")
    print("1. 访问 https://huggingface.co/settings/tokens")
    print("2. 创建新的 Access Token")
    print("3. 复制 Token 并粘贴到下方")
    
    token = input("Hugging Face Token: ").strip()
    
    if token:
        try:
            login(token=token)
            print("✓ 认证成功！")
            
            # 保存到环境变量（可选）
            save_token = input("是否保存 Token 到环境变量？(y/n): ").lower().strip()
            if save_token == 'y':
                # 在 Windows 上设置环境变量
                os.system(f'setx HUGGINGFACE_HUB_TOKEN "{token}"')
                print("✓ Token 已保存到环境变量")
            
            return True
        except Exception as e:
            print(f"✗ 认证失败: {e}")
            return False
    else:
        print("✗ 未提供 Token")
        return False

def test_model_access():
    """测试模型访问权限"""
    print("\n=== 测试模型访问权限 ===")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
        print(f"测试访问模型: {model_name}")
        
        # 尝试加载分词器
        print("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ 分词器加载成功")
        
        # 尝试加载模型
        print("正在加载模型...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto"
        )
        print("✓ 模型加载成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型访问失败: {e}")
        print("\n可能的解决方案:")
        print("1. 确保您有模型访问权限")
        print("2. 访问 https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M")
        print("3. 点击 'Request access' 申请访问权限")
        print("4. 等待 AI4Bharat 团队批准")
        return False

if __name__ == "__main__":
    if setup_huggingface_auth():
        test_model_access()
    else:
        print("请先完成认证设置")
