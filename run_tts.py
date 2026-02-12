#!/usr/bin/env python
# E2C test-time scaling (Table 3, AIME 2024)
import os
import json
import argparse
import yaml
from pathlib import Path

import torch
from tqdm import tqdm

from util.model import load_model
from util.dataset import load_aime2024
from tts_methods import (
    greedy_cot,
    self_consistency,
    e2c_select_lm_judge,
    e2c_select_semantic_cluster,
    e2c_sc,
    e2c_rp,
    tree_of_thoughts,
    forest_of_thought,
    evaluate_predictions,
)

METHOD_FUNCS = {
    "greedy_cot": greedy_cot,
    "self_consistency": self_consistency,
    "e2c_select_lm_judge": e2c_select_lm_judge,
    "e2c_select_semantic_cluster": e2c_select_semantic_cluster,
    "e2c_sc": e2c_sc,
    "e2c_rp": e2c_rp,
    "tree_of_thoughts": tree_of_thoughts,
    "forest_of_thought": forest_of_thought,
}


def run_method(
    name: str,
    model,
    tokenizer,
    dataset: list,
    cfg: dict,
    device: str,
    encoder=None,
) -> dict:
    results = {}
    budgets = cfg["tts"]["budgets"]
    temp = cfg["tts"]["temperature"]
    max_exp = cfg["tts"]["max_explore_tokens"]
    max_exec = cfg["tts"]["max_exec_tokens"]
    max_full = cfg["tts"]["max_full_tokens"]
    n_clusters = cfg["tts"].get("n_clusters", 3)

    if name == "greedy_cot":
        preds = []
        total_tokens = 0
        for item in tqdm(dataset, desc=f"{name} N=1", leave=False):
            q = item["question"]
            try:
                ans, tokens = greedy_cot(model, tokenizer, q, max_full, device)
            except Exception as e:
                print(f"Error {name}: {e}")
                ans, tokens = "", 0
            preds.append(ans)
            total_tokens += tokens
        acc = evaluate_predictions(preds, [x["answer"] for x in dataset])
        avg_tokens = total_tokens / len(dataset) / 1000.0
        for budget in budgets:
            results[f"N{budget}"] = {"acc": round(acc, 1), "tokens_k": round(avg_tokens, 1)}
        print(f"  {name} (N=1): Acc={acc:.1f}% Tokens={avg_tokens:.1f}k")
        return results

    for budget in budgets:
        preds = []
        total_tokens = 0
        for item in tqdm(dataset, desc=f"{name} K/N={budget}", leave=False):
            q, gt = item["question"], item["answer"]
            try:
                if name == "self_consistency":
                    ans, tokens = self_consistency(
                        model, tokenizer, q, budget, max_full, temp, device
                    )
                elif name == "e2c_select_lm_judge":
                    ans, tokens = e2c_select_lm_judge(
                        model, tokenizer, q, budget, max_exp, max_exec, temp, device
                    )
                elif name == "e2c_select_semantic_cluster":
                    ans, tokens = e2c_select_semantic_cluster(
                        model, tokenizer, q, budget, n_clusters,
                        max_exp, max_exec, temp, device, encoder
                    )
                elif name == "e2c_sc":
                    ans, tokens = e2c_sc(
                        model, tokenizer, q, budget, max_exp, max_exec, temp, device
                    )
                elif name == "e2c_rp":
                    ans, tokens = e2c_rp(
                        model, tokenizer, q, budget, max_exp, max_exec, temp, device
                    )
                elif name == "tree_of_thoughts":
                    ans, tokens = tree_of_thoughts(
                        model, tokenizer, q, budget, max_full, temp, device
                    )
                elif name == "forest_of_thought":
                    ans, tokens = forest_of_thought(
                        model, tokenizer, q, budget, max_full, temp, device
                    )
                else:
                    raise ValueError(f"Unknown method: {name}")
            except Exception as e:
                print(f"Error {name} K={budget}: {e}")
                ans, tokens = "", 0
            preds.append(ans)
            total_tokens += tokens

        acc = evaluate_predictions(preds, [x["answer"] for x in dataset])
        avg_tokens = total_tokens / len(dataset) / 1000.0  # k
        results[f"K{budget}" if name.startswith("e2c") else f"N{budget}"] = {
            "acc": round(acc, 1),
            "tokens_k": round(avg_tokens, 1),
        }
        print(f"  {name} budget={budget}: Acc={acc:.1f}% Tokens={avg_tokens:.1f}k")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/tts.yaml", help="Config file")
    parser.add_argument("--methods", nargs="+", default=None, help="Override methods")
    parser.add_argument("--budgets", nargs="+", type=int, default=None, help="Override budgets")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size (for debug)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.methods:
        cfg["methods"] = args.methods
    if args.budgets:
        cfg["tts"]["budgets"] = args.budgets

    torch.manual_seed(cfg["tts"]["seed"])

    print("Loading model...")
    model, tokenizer = load_model(cfg["model"]["model_path"], args.device)

    print("Loading dataset...")
    base_dir = cfg["data"].get("base_dir", "./data")
    dataset = load_aime2024(base_dir)
    if args.limit:
        dataset = dataset[: args.limit]
        print(f"Limited to {len(dataset)} samples")

    encoder = None
    if "e2c_select_semantic_cluster" in cfg["methods"]:
        try:
            from util.embedding import get_encoder
            emb_cfg = cfg.get("embedding", {})
            encoder = get_encoder(
                backend=emb_cfg.get("backend", "modelscope"),
                modelscope_model=emb_cfg.get("modelscope_model", "damo/nlp_gte_sentence-embedding_english-base"),
                huggingface_model=emb_cfg.get("huggingface_model", "all-mpnet-base-v2"),
            )
            print(f"Embedding backend: {encoder.backend_used}")
        except Exception as e:
            print(f"Warning: embedding encoder failed ({e}), semantic cluster will use fallback")

    out_dir = cfg["output"]["save_path"]
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for name in cfg["methods"]:
        if name not in METHOD_FUNCS:
            print(f"Skip unknown method: {name}")
            continue
        print(f"\n=== {name} ===")
        res = run_method(name, model, tokenizer, dataset, cfg, args.device, encoder)
        all_results[name] = res

    # Summary table
    print("\n" + "=" * 60)
    print("Results Summary (Acc %, Tokens k)")
    print("=" * 60)
    for name, res in all_results.items():
        row = " | ".join(f"{k}: {v['acc']}% ({v['tokens_k']}k)" for k, v in res.items())
        print(f"{name}: {row}")

    if cfg["output"].get("save_results"):
        out_path = os.path.join(out_dir, "tts_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
