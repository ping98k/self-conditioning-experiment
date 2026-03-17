"""
Prompt construction module for the Long Horizon Execution Benchmark.

Builds system prompts and user messages following the exact format
from the paper's Appendix E, including few-shot examples for the
dict-sum running-total task.
"""


def build_system_prompt(dictionary: dict[str, int], K: int) -> str:
    """Build the system prompt with dictionary and few-shot examples.

    Follows the format specified in Appendix E of the paper.
    """
    dict_str = ", ".join(f"'{k}': {v}" for k, v in dictionary.items())

    if K == 1:
        intro = (
            "You are an AI assistant. I will provide you with a dictionary "
            "and then give you keys one at a time. "
            "Your task is to keep a running total (starting from 0) "
            "by adding the value associated with each key I provide.\n\n"
            "For each key I provide, respond with the current total "
            "enclosed between <answer></answer> tags.\n\n"
        )
    else:
        intro = (
            f"You are an AI assistant. I will provide you with a dictionary "
            f"and then give you keys in groups of {K}. "
            f"Your task is to keep a running total (starting from 0) "
            f"by adding the values associated with the keys I provide.\n\n"
            f"In each turn, I'll provide {K} keys (comma-separated). "
            f"Respond with the current running sum, enclosed in <answer> tags.\n\n"
        )

    examples = (
        "Examples:\n"
        "Dictionary to maintain: 'apple': 5, 'banana': 0, 'cherry': 7, "
        "'grape': -4, 'kiwi': 2, 'mango': -1\n\n"
        "Example 1: keys in groups of 2\n"
        "User: apple, banana\n"
        "Assistant: <answer>5</answer>\n"
        "User: cherry, grape\n"
        "Assistant: <answer>8</answer>\n"
        "User: kiwi, mango\n"
        "Assistant: <answer>9</answer>\n\n"
        "Example 2: keys in groups of 3\n"
        "User: apple, banana, cherry\n"
        "Assistant: <answer>12</answer>\n"
        "User: grape, kiwi, mango\n"
        "Assistant: <answer>9</answer>\n\n"
        "Example 3: keys in groups of 6\n"
        "User: apple, banana, cherry, grape, kiwi, mango\n"
        "Assistant: <answer>9</answer>\n\n"
    )

    task = (
        f"Now, here is the actual task:\n"
        f"Dictionary to maintain: {dict_str}\n\n"
        f"Ready to start!\n"
        f"IMPORTANT: DO NOT OUTPUT ANY OTHER TEXT OUTSIDE ANSWER TAGS. "
        f"Only provide the final running sum OF ALL TURNS in <answer> tags."
    )

    return intro + examples + task


def format_turn_keys(keys: list[str]) -> str:
    """Format keys for a single turn's user message."""
    return ", ".join(keys)
