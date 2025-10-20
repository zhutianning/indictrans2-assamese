#!/usr/bin/env python3
"""
按照官方指南准备数据格式用于 LoRA 微调
官方要求的数据格式：
en-indic-exp/
├── train/
│   ├── eng_Latn-asm_Beng/
│   │   ├── train.eng_Latn
│   │   └── train.asm_Beng
│   └── ...
└── dev/
    ├── eng_Latn-asm_Beng/
    │   ├── dev.eng_Latn
    │   └── dev.asm_Beng
    └── ...
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_official_data_format():
    """按照官方指南准备数据格式"""
    
    print("=== 按照官方指南准备数据格式 ===")
    
    # 读取原始数据
    csv_path = "downloads/WMT_INDIC_MT_Task_2025/WMT INDIC MT Task 2025/Category I/English-Assamese Training Data 2025.csv"
    
    print(f"读取原始数据: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 移除标题行（如果存在）
    if len(df) > 0 and df.iloc[0]['en'] == 'en' and df.iloc[0]['as'] == 'as':
        df = df.iloc[1:].reset_index(drop=True)
    
    # 移除空值
    df.dropna(subset=['en', 'as'], inplace=True)
    
    print(f"原始数据行数: {len(df)}")
    
    # 按照官方要求切分数据
    # 使用较小的数据集进行测试
    if len(df) > 1000:
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)
        print(f"使用前1000行数据进行测试")
    
    # 切分数据：70% 训练，15% 验证，15% 测试
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"训练集: {len(train_df)} 行")
    print(f"验证集: {len(dev_df)} 行")
    print(f"测试集: {len(test_df)} 行")
    
    # 创建官方要求的目录结构
    output_dir = "assamese_english_official_format"
    train_dir = os.path.join(output_dir, "train", "eng_Latn-asm_Beng")
    dev_dir = os.path.join(output_dir, "dev", "eng_Latn-asm_Beng")
    test_dir = os.path.join(output_dir, "test", "eng_Latn-asm_Beng")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 保存训练数据
    train_src_file = os.path.join(train_dir, "train.eng_Latn")
    train_tgt_file = os.path.join(train_dir, "train.asm_Beng")
    
    with open(train_src_file, 'w', encoding='utf-8') as f:
        for text in train_df['en']:
            f.write(f"{text.strip()}\n")
    
    with open(train_tgt_file, 'w', encoding='utf-8') as f:
        for text in train_df['as']:
            f.write(f"{text.strip()}\n")
    
    # 保存验证数据
    dev_src_file = os.path.join(dev_dir, "dev.eng_Latn")
    dev_tgt_file = os.path.join(dev_dir, "dev.asm_Beng")
    
    with open(dev_src_file, 'w', encoding='utf-8') as f:
        for text in dev_df['en']:
            f.write(f"{text.strip()}\n")
    
    with open(dev_tgt_file, 'w', encoding='utf-8') as f:
        for text in dev_df['as']:
            f.write(f"{text.strip()}\n")
    
    # 保存测试数据
    test_src_file = os.path.join(test_dir, "test.eng_Latn")
    test_tgt_file = os.path.join(test_dir, "test.asm_Beng")
    
    with open(test_src_file, 'w', encoding='utf-8') as f:
        for text in test_df['en']:
            f.write(f"{text.strip()}\n")
    
    with open(test_tgt_file, 'w', encoding='utf-8') as f:
        for text in test_df['as']:
            f.write(f"{text.strip()}\n")
    
    print(f"\n✓ 数据已按照官方格式保存到: {output_dir}")
    print(f"训练数据: {train_src_file}, {train_tgt_file}")
    print(f"验证数据: {dev_src_file}, {dev_tgt_file}")
    print(f"测试数据: {test_src_file}, {test_tgt_file}")
    
    # 显示数据样本
    print(f"\n=== 数据样本 ===")
    print("训练集样本:")
    print(f"英语: {train_df['en'].iloc[0]}")
    print(f"阿萨姆语: {train_df['as'].iloc[0]}")
    print("\n验证集样本:")
    print(f"英语: {dev_df['en'].iloc[0]}")
    print(f"阿萨姆语: {dev_df['as'].iloc[0]}")
    print("\n测试集样本:")
    print(f"英语: {test_df['en'].iloc[0]}")
    print(f"阿萨姆语: {test_df['as'].iloc[0]}")
    
    return output_dir

if __name__ == "__main__":
    prepare_official_data_format()
