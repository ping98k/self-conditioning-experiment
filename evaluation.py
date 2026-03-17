"""
Evaluation module for the Long Horizon Execution Benchmark.

Provides answer parsing from model responses and computation of
turn accuracy, task accuracy, and horizon length metrics as
defined in Section 2 of the paper.
"""

import re

import numpy as np


def parse_answer(response: str) -> int | None:
    """Extract integer answer from <answer> tags in model response.

    Returns None if the response doesn't contain valid <answer> tags
    with an integer value (counted as a format failure).
    """
    match = re.search(r"<answer>\s*(-?\d+)\s*</answer>", response)
    if match:
        return int(match.group(1))
    return None


def compute_turn_accuracy(all_results: list[dict], turn_idx: int) -> float:
    """Fraction of samples with correct state update at a given turn."""
    n = len(all_results)
    if n == 0:
        return 0.0
    correct = sum(1 for r in all_results if r["turn_correct"][turn_idx])
    return correct / n


def compute_task_accuracy(all_results: list[dict], up_to_turn: int) -> float:
    """Fraction of samples where all turns up to given turn are correct."""
    n = len(all_results)
    if n == 0:
        return 0.0
    correct = sum(
        1 for r in all_results if all(r["turn_correct"][: up_to_turn + 1])
    )
    return correct / n


def compute_horizon_length(
    all_results: list[dict], num_turns: int, success_rate: float = 0.5
) -> int:
    """First turn where task accuracy drops below success_rate (H_s)."""
    for t in range(num_turns):
        if compute_task_accuracy(all_results, t) < success_rate:
            return t
    return num_turns


def compute_all_metrics(all_results: list[dict], num_turns: int) -> dict:
    """Compute all metrics across turns.

    Returns dict with turn_accuracy, task_accuracy, horizon_length,
    and format_failure_rate per turn.
    """
    turn_accuracies = []
    task_accuracies = []
    format_failure_rates = []

    for t in range(num_turns):
        turn_accuracies.append(compute_turn_accuracy(all_results, t))
        task_accuracies.append(compute_task_accuracy(all_results, t))
        n = len(all_results)
        fmt_fails = sum(
            1 for r in all_results if r["format_failures"][t]
        )
        format_failure_rates.append(fmt_fails / n if n > 0 else 0.0)

    horizon = compute_horizon_length(all_results, num_turns)

    return {
        "turn_accuracy": turn_accuracies,
        "task_accuracy": task_accuracies,
        "format_failure_rate": format_failure_rates,
        "horizon_length_0.5": horizon,
        "num_samples": len(all_results),
        "num_turns": num_turns,
    }
