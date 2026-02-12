# TTS methods
import random
import numpy as np
import torch
from typing import List, Tuple, Optional
from util.reward import boxed_evaluate, check_answer_match, extract_boxed_content
from prompts import (
    build_exploration_prompt,
    build_e2c_prompt,
    build_full_cot_prompt,
    format_llm_judge_prompt,
)

# E2C special token IDs (Qwen3)
TOKEN_EXPLORATION_END = 151672   # </EXPLORATION>
TOKEN_EXECUTION_END = 151674     # </EXECUTION>


def _extract_exploration(text: str) -> str:
    if "<EXPLORATION>" in text and "</EXPLORATION>" in text:
        start = text.index("<EXPLORATION>") + len("<EXPLORATION>")
        end = text.index("</EXPLORATION>")
        return text[start:end].strip()
    if "<EXPLORATION>" in text and "<EXECUTION>" in text:
        start = text.index("<EXPLORATION>") + len("<EXPLORATION>")
        end = text.index("<EXECUTION>")
        return text[start:end].strip()
    return text.strip()


def _extract_answer(text: str) -> str:
    contents = extract_boxed_content(text)
    return contents[-1] if contents else ""


def _majority_vote(answers: List[str]) -> str:
    if not answers:
        return ""
    from collections import Counter
    valid = [a for a in answers if a and a.strip()]
    if not valid:
        return answers[0] if answers else ""
    return Counter(valid).most_common(1)[0][0]


def _weighted_majority(answers: List[str], weights: List[float]) -> str:
    if not answers or not weights or len(answers) != len(weights):
        return _majority_vote(answers)
    from collections import defaultdict
    scores = defaultdict(float)
    for a, w in zip(answers, weights):
        if a and a.strip():
            scores[a.strip()] += w
    return max(scores, key=scores.get) if scores else (answers[0] if answers else "")


def generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float = 0.9,
    do_sample: bool = True,
    stop_token_ids: Optional[List[int]] = None,
    device: str = "cuda",
) -> Tuple[List[str], int]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=True,
        max_length=4096,
    ).to(device)
    eos_ids = [tokenizer.eos_token_id]
    if stop_token_ids:
        eos_ids.extend(stop_token_ids)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            top_p=0.95 if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
    new_ids = out[:, inputs.input_ids.shape[1]:]
    texts = tokenizer.batch_decode(new_ids, skip_special_tokens=False)
    total_tokens = new_ids.numel()
    return texts, total_tokens


def sample_explorations(
    model, tokenizer, question: str, K: int, max_explore_tokens: int,
    temperature: float, device: str
) -> Tuple[List[str], int]:
    prompt = build_exploration_prompt(question, tokenizer)
    prompts = [prompt] * K
    texts, total = generate(
        model, tokenizer, prompts, max_explore_tokens,
        temperature=temperature, do_sample=True,
        stop_token_ids=[TOKEN_EXPLORATION_END],
        device=device,
    )
    plans = [_extract_exploration(t) for t in texts]
    return plans, total


def run_execution(
    model, tokenizer, question: str, exploration: str, max_exec_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    prompt = build_e2c_prompt(question, exploration, tokenizer)
    texts, total = generate(
        model, tokenizer, [prompt], max_exec_tokens,
        temperature=temperature, do_sample=(temperature > 0),
        device=device,
    )
    return texts[0] if texts else "", total


def run_full_cot(
    model, tokenizer, question: str, max_tokens: int,
    temperature: float, do_sample: bool, device: str
) -> Tuple[str, int]:
    prompt = build_full_cot_prompt(question, tokenizer)
    texts, total = generate(
        model, tokenizer, [prompt], max_tokens,
        temperature=temperature, do_sample=do_sample,
        device=device,
    )
    return texts[0] if texts else "", total


# ---------- TTS Methods ----------

def greedy_cot(model, tokenizer, question: str, max_tokens: int, device: str) -> Tuple[str, int]:
    return run_full_cot(model, tokenizer, question, max_tokens, 0.0, False, device)


def self_consistency(
    model, tokenizer, question: str, N: int, max_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    prompt = build_full_cot_prompt(question, tokenizer)
    prompts = [prompt] * N
    texts, total = generate(
        model, tokenizer, prompts, max_tokens,
        temperature=temperature, do_sample=True,
        device=device,
    )
    answers = [_extract_answer(t) for t in texts]
    return _majority_vote(answers), total


def e2c_select_lm_judge(
    model, tokenizer, question: str, K: int,
    max_explore_tokens: int, max_exec_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    plans, t1 = sample_explorations(
        model, tokenizer, question, K, max_explore_tokens, temperature, device
    )
    if not plans:
        return "", t1
    judge_prompt = format_llm_judge_prompt(question, plans)
    messages = [{"role": "user", "content": judge_prompt}]
    judge_inp = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chosen, t2 = generate(
        model, tokenizer, [judge_inp], max_new_tokens=512,
        temperature=0.0, do_sample=False, device=device
    )
    best_plan = _extract_exploration(chosen[0]) if chosen else plans[0]
    if not best_plan.strip():
        best_plan = plans[0]
    out, t3 = run_execution(
        model, tokenizer, question, best_plan, max_exec_tokens, 0.0, device
    )
    return _extract_answer(out), t1 + t2 + t3


def e2c_select_semantic_cluster(
    model, tokenizer, question: str, K: int, M: int,
    max_explore_tokens: int, max_exec_tokens: int,
    temperature: float, device: str,
    encoder=None,
) -> Tuple[str, int]:
    plans, t1 = sample_explorations(
        model, tokenizer, question, K, max_explore_tokens, temperature, device
    )
    if not plans:
        return "", t1
    if encoder is None:
        try:
            from util.embedding import get_encoder
            encoder = get_encoder(backend="auto")
        except Exception:
            # Fallback: random selection
            centroids = plans[:min(M, len(plans))]
            weights = [1.0] * len(centroids)
            answers = []
            total_exec = 0
            for c in centroids:
                out, t = run_execution(model, tokenizer, question, c, max_exec_tokens, 0.0, device)
                answers.append(_extract_answer(out))
                total_exec += t
            return _weighted_majority(answers, weights), t1 + total_exec
    emb = encoder.encode(plans)
    # L2-normalize for cosine similarity (paper A.4)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-8)
    from sklearn.cluster import KMeans
    n_clusters = min(M, len(plans))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(emb)
    labels = kmeans.labels_
    centroid_plans = []
    weights = []
    for i in range(n_clusters):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        center = kmeans.cluster_centers_[i]
        dists = np.linalg.norm(emb[idx] - center, axis=1)
        best_local = idx[np.argmin(dists)]
        centroid_plans.append(plans[best_local])
        weights.append(len(idx))
    answers = []
    total_exec = 0
    for p in centroid_plans:
        out, t = run_execution(model, tokenizer, question, p, max_exec_tokens, 0.0, device)
        answers.append(_extract_answer(out))
        total_exec += t
    return _weighted_majority(answers, weights), t1 + total_exec


def e2c_sc(
    model, tokenizer, question: str, K: int,
    max_explore_tokens: int, max_exec_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    plans, t1 = sample_explorations(
        model, tokenizer, question, K, max_explore_tokens, temperature, device
    )
    if not plans:
        return "", t1
    answers = []
    total_exec = 0
    for p in plans:
        out, t = run_execution(model, tokenizer, question, p, max_exec_tokens, 0.0, device)
        answers.append(_extract_answer(out))
        total_exec += t
    return _majority_vote(answers), t1 + total_exec


def e2c_rp(
    model, tokenizer, question: str, K: int,
    max_explore_tokens: int, max_exec_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    plans, t1 = sample_explorations(
        model, tokenizer, question, K, max_explore_tokens, temperature, device
    )
    if not plans:
        return "", t1
    p = random.choice(plans)
    out, t2 = run_execution(model, tokenizer, question, p, max_exec_tokens, 0.0, device)
    return _extract_answer(out), t1 + t2


def tree_of_thoughts(
    model, tokenizer, question: str, N: int, max_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    # ToT: tree with b=2, depth ~log2(N), then full chain per leaf, majority vote
    import math
    b = 2
    depth = max(1, math.ceil(math.log(N) / math.log(b)))
    thought_tokens = min(256, max_tokens // 4)
    total_tokens = 0
    current = [question]
    for _ in range(depth):
        next_level = []
        for _node in current[:N]:
            prompt = build_full_cot_prompt(question, tokenizer)
            texts, t = generate(model, tokenizer, [prompt], thought_tokens, temperature=temperature, do_sample=True, device=device)
            total_tokens += t
            next_level.extend(texts)
        current = next_level[:N] if len(next_level) > N else next_level
    if not current:
        return "", total_tokens
    answers = []
    for leaf in current[:N]:
        prompt = build_full_cot_prompt(question, tokenizer)
        texts, t = generate(model, tokenizer, [prompt], max_tokens, temperature=temperature, do_sample=True, device=device)
        total_tokens += t
        answers.append(_extract_answer(texts[0]))
    return _majority_vote(answers), total_tokens


def forest_of_thought(
    model, tokenizer, question: str, N: int, max_tokens: int,
    temperature: float, device: str
) -> Tuple[str, int]:
    # FoT: N trees, 2-step expand each, then one full chain per tree, vote
    thought_tokens = min(256, max_tokens // 4)
    total_tokens = 0
    representatives = []
    for _ in range(N):
        node = question
        for _ in range(2):
            prompt = build_full_cot_prompt(question, tokenizer)
            texts, t = generate(model, tokenizer, [prompt], thought_tokens, temperature=temperature, do_sample=True, device=device)
            total_tokens += t
            node = texts[0] if texts else node
        representatives.append(node)
    answers = []
    for rep in representatives:
        prompt = build_full_cot_prompt(question, tokenizer)
        texts, t = generate(model, tokenizer, [prompt], max_tokens, temperature=temperature, do_sample=True, device=device)
        total_tokens += t
        answers.append(_extract_answer(texts[0]))
    return _majority_vote(answers), total_tokens


def evaluate_predictions(predictions: List[str], ground_truths: List[str]) -> float:
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        succ, _ = boxed_evaluate(pred, gt)
        if not succ and pred and "boxed" not in pred:
            succ = check_answer_match(pred, gt)
        if succ:
            correct += 1
    return correct / len(predictions) * 100.0 if predictions else 0.0
