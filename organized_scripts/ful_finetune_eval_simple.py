#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简版：全量微调模型评估（优先使用 GPU,参考 LoRA 脚本添加 BLEU/chrF）
- 使用基座 tokenizer
- 评估 dev 集前 N 条，保存结果到 out_dir/simple_evaluation_results.json（含 metrics 与 samples）
"""

import os
import json
import torch
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from safetensors.torch import load_file
from sacrebleu.metrics import BLEU, CHRF

# 避免 CPU 多线程过载（对 GPU 推理无影响）
import time
torch.set_num_threads(min(4, os.cpu_count() or 4))

bleu_metric = BLEU()
chrf_metric = CHRF()

def resolve_model_dir(out_dir: str) -> str:
    ckpt = os.path.join(out_dir, "checkpoint-4000") # checkpoint-4000 is the latest checkpoint
    if os.path.isdir(ckpt) and os.path.isfile(os.path.join(ckpt, "config.json")):
        return ckpt
    return out_dir


def load_data(data_dir: str, pair_dir: str, src_file: str, tgt_file: str, limit: int):
    pair_path = os.path.join(data_dir, pair_dir)
    src_path = os.path.join(pair_path, src_file)
    tgt_path = os.path.join(pair_path, tgt_file)

    with open(src_path, "r", encoding="utf-8") as f:
        src = [l.strip() for l in f if l.strip()]
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt = [l.strip() for l in f if l.strip()]

    if limit > 0:
        src = src[:limit]
        tgt = tgt[:limit]
    return src, tgt


def compute_metrics(pred_list, ref_list):
    # 过滤空预测，仅在有效对上计算
    valid_pairs = [(p, r) for p, r in zip(pred_list, ref_list) if p.strip()]
    if not valid_pairs:
        return {"BLEU": 0.0, "chrF": 0.0}
    preds, refs = zip(*valid_pairs)
    try:
        bleu_score = bleu_metric.corpus_score(list(preds), [list(refs)]).score
        chrf_score = chrf_metric.corpus_score(list(preds), [list(refs)]).score
        return {"BLEU": bleu_score, "chrF": chrf_score}
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return {"BLEU": 0.0, "chrF": 0.0}


def main():
    # 路径（按你项目的默认）
    out_dir = "/proj/uppmax2025-3-5/nobackup/tizh1048/MT/Group_project/outputs/assamese_english_full_ft_20251023_020135"
    data_dir = "/home/tizh1048/MT/indictrans2-assamese/assamese_english_official_format"

    # 方向：INDIC_EN（阿萨姆语→英语）
    src_lang = "asm_Beng"
    tgt_lang = "eng_Latn"
    pair_dir = "dev/eng_Latn-asm_Beng"
    src_file = "dev.asm_Beng"
    tgt_file = "dev.eng_Latn"

    # 简化参数
    limit = 0
    max_src_len = 64

    # 设备：优先 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"使用设备: {device}, dtype: {torch_dtype}")

    # 分词器：用基座，避免自定义 tokenizer 配置冲突
    base_model = "ai4bharat/indictrans2-indic-en-dist-200M"
    print("加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # 模型目录:
    model_dir = resolve_model_dir(out_dir)

    # 使用 from_pretrained 安全加载（强制 eager 注意力）
    print(f"加载模型: {model_dir}")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    try:
        setattr(config, "attn_implementation", "eager")
    except Exception as e:
        print(f"warn(set attn_implementation): {e}")

    def _has_meta(m):
        for _, p in m.named_parameters(recurse=True):
            if getattr(p, "is_meta", False):
                return True
        for _, b in m.named_buffers(recurse=True):
            if getattr(b, "is_meta", False):
                return True
        return False

    # 路径1：优先使用 from_pretrained
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        config=config,
        low_cpu_mem_usage=False,   # 禁用 meta 初始化
        dtype=torch_dtype,
        attn_implementation="eager",
    )

    if _has_meta(model):
        print("detected META tensors after from_pretrained → falling back to manual load")
        # 路径2：回退为从 config 构建 + 手动加载真实权重
        state_path_st = os.path.join(model_dir, "model.safetensors")
        state_path_pt = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(state_path_st):
            state_dict = load_file(state_path_st)
        elif os.path.isfile(state_path_pt):
            state_dict = torch.load(state_path_pt, map_location="cpu")
        else:
            raise FileNotFoundError(f"未找到权重文件: {state_path_st} 或 {state_path_pt}")

        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"missing keys: {len(missing)} (展示最多5条)：{missing[:5]}")
        if unexpected:
            print(f"unexpected keys: {len(unexpected)} (展示最多5条)：{unexpected[:5]}")
        if hasattr(model, "tie_weights"):
            try:
               model.tie_weights()
            except Exception:
               pass

    # 迁移到设备并 eval
    model = model.to(device).eval()
    model.config.use_cache = False  # 额外保险：全局关闭 cache
    print("模型加载完成并已迁移到设备")

    # 补齐必要 token id，防止 generate 分支取到 None
    try:
        model.config.eos_token_id = tok.eos_token_id
        model.config.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        # set decoder_start_token into target language token :  eng_Latn
        tgt_lang_id = tok.convert_tokens_to_ids(tgt_lang)
        if tgt_lang_id is None or (hasattr(tok, "unk_token_id") and tgt_lang_id == tok.unk_token_id):
            raise ValueError(f"未找到目标语言 token: {tgt_lang}")
        model.config.decoder_start_token_id = tgt_lang_id
        try:
            model.config.forced_bos_token_id = tgt_lang_id
        except Exception:
            pass
        
        # 同步到 generation_config（部分实现读取这个）
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.eos_token_id = model.config.eos_token_id
            model.generation_config.pad_token_id = model.config.pad_token_id
            model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
            try:
                model.generation_config.forced_bos_token_id = tgt_lang_id
            except Exception:
                pass
    except Exception as e:
        print(f"warn(sync generation tokens): {e}")
    print("start token id 修改完成")

    # 加载数据
    print("加载数据 ...")
    src_lines, tgt_lines = load_data(data_dir, pair_dir, src_file, tgt_file, limit)
    print(f"样本数: {len(src_lines)}")

    # 生成
    print("开始生成 ...")
    results = []
    preds = []
    for i, (s, t) in enumerate(zip(src_lines, tgt_lines), 1):
        try:
            text = f"{src_lang} {tgt_lang} {s}"
            enc = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_src_len
            )
            # 兜底 attention_mask，避免 None，并迁移到设备
            if "attention_mask" not in enc or enc["attention_mask"] is None:
                enc["attention_mask"] = torch.ones_like(enc["input_ids"])
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.inference_mode():
                gen = model.generate(
                    input_ids=enc["input_ids"], 
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=64,
                    min_new_tokens=1, 
                    num_beams=1, 
                    do_sample=False,
                    pad_token_id=model.config.pad_token_id, 
                    eos_token_id=model.config.eos_token_id,
                    use_cache=False,
                    forced_bos_token_id=getattr(model.config, "forced_bos_token_id", None),
                )

            pred = tok.decode(gen[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            results.append({"src": s, "tgt": t, "pred": pred})
            preds.append(pred)
            if i % 5 == 0:
                print(f"已生成 {i}/{len(src_lines)}")
        except Exception as e:
            import traceback
            print(f"样本 {i} 失败: {e}")
            traceback.print_exc()
            results.append({"src": s, "tgt": t, "pred": ""})
            preds.append("")
            continue

    # 计算指标（BLEU / chrF）
    print("\n计算评估指标 ...")
    metrics = compute_metrics(preds, tgt_lines[:len(preds)])
    print(f"BLEU: {metrics['BLEU']:.4f}")
    print(f"chrF: {metrics['chrF']:.4f}")

    # 保存（整合 metrics 与 samples）
    out_file = os.path.join(out_dir, "simple_evaluation_results.json")
    payload = {
        "metrics": metrics,
        "samples": results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"完成，结果已写入: {out_file}")

    # 展示几个样本
    print("\n=== 样本结果（最多3条） ===")
    shown = 0
    for r in results:
        if r["pred"].strip():
            print(f"\nSRC: {r['src']}\nREF: {r['tgt']}\nPRED: {r['pred']}")
            shown += 1
            if shown >= 3:
                break


if __name__ == "__main__":
    main()