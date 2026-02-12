# Prompts (Appendix A.6)
from typing import List

QUESTION_SUFFIX = "Provide the final answer in the boxed{}. Please reasoning step-by-step."
SOLUTION_PREFIX = "<EXPLORATION>"

LLM_JUDGE_PROMPT = """Role: You are an expert mathematical reasoner and an impartial judge. Your task is to evaluate several proposed plans for solving a given math problem and identify the single best one.

Input:
• Problem: {problem}

• Candidate Plans:
{plans}

Instructions:
1. Carefully analyze the problem and each of the K candidate plans.
2. Assess the plans based on their logical soundness, potential for success, and efficiency.
3. Select the single best plan that is most likely to lead to a correct and complete solution.

Output Format: Output only the full text of the single best plan you have selected. Do not add any extra commentary, explanation, or formatting."""


def format_llm_judge_prompt(problem: str, plans: List[str]) -> str:
    plans_text = "\n".join(f"Plan {i+1}:\n{p.strip()}" for i, p in enumerate(plans))
    return LLM_JUDGE_PROMPT.format(problem=problem, plans=plans_text)


def build_e2c_prompt(question: str, exploration: str, tokenizer, question_suffix: str = None) -> str:
    qs = question_suffix or QUESTION_SUFFIX
    content = question + " " + qs
    if exploration and exploration.strip():
        content = content + "\n\nGuideline (follow exactly):\n" + exploration.strip()
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + "<EXPLORATION>" + (exploration.strip() or "") + "</EXPLORATION><EXECUTION>"


def build_exploration_prompt(question: str, tokenizer, question_suffix: str = None) -> str:
    qs = question_suffix or QUESTION_SUFFIX
    content = question + " " + qs
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + SOLUTION_PREFIX


def build_full_cot_prompt(question: str, tokenizer, question_suffix: str = None) -> str:
    qs = question_suffix or QUESTION_SUFFIX
    content = question + " " + qs
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + SOLUTION_PREFIX
