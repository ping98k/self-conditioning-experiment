"""
Experiment runners for the Long Horizon Execution Benchmark.

Implements the two core experiments from the paper:
1. Standard execution benchmark — measures turn accuracy degradation
   as task length increases (Figures 3/4).
2. Self-conditioning experiment — injects artificial error histories
   at varying rates and measures turn accuracy at a fixed turn (Figure 5).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np

from config import ExperimentConfig, ModelConfig
from dataset import generate_sample
from evaluation import compute_all_metrics, parse_answer
from llm_client import LLMClient
from prompts import build_system_prompt, format_turn_keys


def _run_standard_sample(
    sample_idx: int,
    sample: dict,
    model_config: ModelConfig,
    working_capacity: int,
    num_samples: int,
) -> dict:
    """Run a single standard experiment sample (thread-safe)."""
    client = LLMClient(model_config)
    system_prompt = build_system_prompt(
        sample["dictionary"], working_capacity
    )
    messages = [{"role": "system", "content": system_prompt}]

    turn_correct = []
    model_answers = []
    format_failures = []
    prev_model_answer = 0

    for t, turn_keys in enumerate(sample["turns"]):
        user_msg = format_turn_keys(turn_keys)
        messages.append({"role": "user", "content": user_msg})

        response = client.chat(messages)
        messages.append({"role": "assistant", "content": response})

        predicted = parse_answer(response)
        format_failures.append(predicted is None)

        expected_delta = sum(
            sample["dictionary"][k] for k in turn_keys
        )

        if predicted is not None:
            model_delta = predicted - prev_model_answer
            is_correct = model_delta == expected_delta
            prev_model_answer = predicted
        else:
            is_correct = False

        turn_correct.append(is_correct)
        model_answers.append(predicted)

        status = "ok" if is_correct else "WRONG"
        print(
            f"  [Sample {sample_idx + 1}/{num_samples}] "
            f"Turn {t + 1:3d}: [{status:>5s}]  "
            f"predicted={predicted}  "
            f"expected_sum={sample['ground_truth'][t]}"
        )

    return {
        "sample_idx": sample_idx,
        "turn_correct": turn_correct,
        "model_answers": [
            int(a) if a is not None else None for a in model_answers
        ],
        "ground_truth": sample["ground_truth"],
        "format_failures": format_failures,
    }


def run_standard_experiment(
    model_config: ModelConfig, exp_config: ExperimentConfig
) -> dict:
    """Run the standard execution benchmark.

    Measures how turn accuracy degrades as the number of turns increases.
    Each sample runs a full multi-turn conversation where the model
    maintains a running sum of dictionary values. Samples are processed
    in parallel using a thread pool.
    """
    rng = np.random.default_rng(exp_config.seed)

    print(
        f"Running standard experiment: {exp_config.num_samples} samples, "
        f"{exp_config.num_turns} turns, K={exp_config.working_capacity}, "
        f"workers={exp_config.num_workers}"
    )

    # Pre-generate all samples sequentially (RNG is not thread-safe)
    samples = []
    for _ in range(exp_config.num_samples):
        samples.append(
            generate_sample(
                exp_config.dict_size,
                exp_config.min_value,
                exp_config.max_value,
                exp_config.num_turns,
                exp_config.working_capacity,
                rng,
            )
        )

    # Process samples in parallel
    all_results = [None] * exp_config.num_samples
    with ThreadPoolExecutor(max_workers=exp_config.num_workers) as executor:
        futures = {
            executor.submit(
                _run_standard_sample,
                idx,
                sample,
                model_config,
                exp_config.working_capacity,
                exp_config.num_samples,
            ): idx
            for idx, sample in enumerate(samples)
        }
        for future in as_completed(futures):
            idx = futures[future]
            all_results[idx] = future.result()
            print(f"--- Sample {idx + 1}/{exp_config.num_samples} done ---")

    metrics = compute_all_metrics(all_results, exp_config.num_turns)

    print(f"\n{'=' * 40}")
    print(f"Horizon length (H_0.5): {metrics['horizon_length_0.5']}")
    print(f"Mean turn accuracy: {np.mean(metrics['turn_accuracy']):.4f}")

    return {"results": all_results, "metrics": metrics}


def _run_self_cond_sample(
    sample_idx: int,
    sample: dict,
    injected_history: tuple[list[dict], int],
    model_config: ModelConfig,
    working_capacity: int,
    eval_turn: int,
    num_samples: int,
    error_rate: float,
) -> dict:
    """Run a single self-conditioning sample (thread-safe)."""
    client = LLMClient(model_config)
    history_messages, last_injected_answer = injected_history

    system_prompt = build_system_prompt(sample["dictionary"], working_capacity)
    messages = [{"role": "system", "content": system_prompt}] + history_messages

    # Evaluate at the final turn
    eval_keys = sample["turns"][eval_turn - 1]
    user_msg = format_turn_keys(eval_keys)
    messages.append({"role": "user", "content": user_msg})

    response = client.chat(messages)
    predicted = parse_answer(response)

    expected_delta = sum(sample["dictionary"][k] for k in eval_keys)

    if predicted is not None:
        model_delta = predicted - last_injected_answer
        is_correct = model_delta == expected_delta
    else:
        is_correct = False

    status = "ok" if is_correct else "WRONG"
    print(
        f"  [Rate {error_rate:.0%}] "
        f"Sample {sample_idx + 1}/{num_samples}  [{status}]"
    )

    return {
        "sample_idx": sample_idx,
        "predicted": predicted,
        "expected_delta": expected_delta,
        "last_injected": last_injected_answer,
        "correct": is_correct,
    }


def run_self_conditioning_experiment(
    model_config: ModelConfig, exp_config: ExperimentConfig
) -> dict:
    """Run the self-conditioning experiment.

    Pre-generates samples with a fixed seed, then for each error rate,
    builds artificial chat histories with that fraction of wrong answers
    injected, and evaluates the model's turn accuracy at the evaluation
    turn. Samples are processed in parallel using a thread pool.
    """
    eval_turn = exp_config.self_cond_eval_turn

    # Pre-generate all samples so they're identical across error rates
    sample_rng = np.random.default_rng(exp_config.seed)
    samples = []
    for _ in range(exp_config.num_samples):
        samples.append(
            generate_sample(
                exp_config.dict_size,
                exp_config.min_value,
                exp_config.max_value,
                eval_turn,
                exp_config.working_capacity,
                sample_rng,
            )
        )

    results_by_rate = {}

    print(
        f"Running self-conditioning experiment: "
        f"{exp_config.num_samples} samples, eval at turn {eval_turn}, "
        f"workers={exp_config.num_workers}"
    )

    for error_rate in exp_config.error_rates:
        print(f"\n=== Error rate: {error_rate:.0%} ===")
        # Pre-build injected histories per sample (RNG is sequential)
        error_rng = np.random.default_rng(
            exp_config.seed + int(error_rate * 1000)
        )
        injected_histories = []
        for sample in samples:
            history_messages = []
            last_injected = 0
            for t in range(eval_turn - 1):
                turn_keys = sample["turns"][t]
                user_msg = format_turn_keys(turn_keys)
                history_messages.append({"role": "user", "content": user_msg})

                correct_answer = sample["ground_truth"][t]
                if error_rng.random() < error_rate:
                    perturbation = int(error_rng.integers(-50, 51))
                    if perturbation == 0:
                        perturbation = 1
                    injected = correct_answer + perturbation
                else:
                    injected = correct_answer

                history_messages.append(
                    {"role": "assistant", "content": f"<answer>{injected}</answer>"}
                )
                last_injected = injected
            injected_histories.append((history_messages, last_injected))

        # Evaluate samples in parallel
        sample_results = [None] * exp_config.num_samples
        with ThreadPoolExecutor(max_workers=exp_config.num_workers) as executor:
            futures = {
                executor.submit(
                    _run_self_cond_sample,
                    sample_idx,
                    samples[sample_idx],
                    injected_histories[sample_idx],
                    model_config,
                    exp_config.working_capacity,
                    eval_turn,
                    exp_config.num_samples,
                    error_rate,
                ): sample_idx
                for sample_idx in range(exp_config.num_samples)
            }
            for future in as_completed(futures):
                idx = futures[future]
                sample_results[idx] = future.result()

        accuracy = sum(r["correct"] for r in sample_results) / len(
            sample_results
        )
        results_by_rate[str(error_rate)] = {
            "accuracy": accuracy,
            "samples": sample_results,
        }
        print(f"  Turn accuracy at turn {eval_turn}: {accuracy:.4f}")

    print(f"\n{'=' * 40}")
    print("Self-Conditioning Summary:")
    for rate_str, data in results_by_rate.items():
        rate = float(rate_str)
        print(f"  Error rate {rate:>5.0%}: accuracy = {data['accuracy']:.4f}")

    return results_by_rate


def save_results(results: dict, exp_config: ExperimentConfig, label: str):
    """Save experiment results and config to a timestamped JSON file."""
    os.makedirs(exp_config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.json"
    filepath = os.path.join(exp_config.output_dir, filename)

    output = {
        "experiment": label,
        "config": {
            "num_samples": exp_config.num_samples,
            "dict_size": exp_config.dict_size,
            "working_capacity": exp_config.working_capacity,
            "num_turns": exp_config.num_turns,
            "seed": exp_config.seed,
        },
        "data": results,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {filepath}")
