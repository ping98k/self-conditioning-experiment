"""
Main entry point for the Long Horizon Execution Benchmark.

Reproduces the two core experiments from 'The Illusion of Diminishing
Returns: Measuring Long Horizon Execution in LLMs' (arXiv:2509.09677):

1. Standard execution benchmark — measures turn accuracy degradation
   as task length increases across model sizes (Figures 3/4 in paper).
2. Self-conditioning experiment — injects artificial error histories
   at varying rates (0%–100%) and measures turn accuracy (Figure 5).

Usage:
    uv run main.py standard [options]
    uv run main.py self-conditioning [options]
"""

import argparse
import sys

from config import ExperimentConfig, ModelConfig
from experiment import (
    run_self_conditioning_experiment,
    run_standard_experiment,
    save_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long Horizon Execution Benchmark"
    )
    parser.add_argument(
        "experiment",
        choices=["standard", "self-conditioning"],
        help="Which experiment to run",
    )

    model_group = parser.add_argument_group("Model settings")
    model_group.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="LLM server base URL (default: http://localhost:8080/v1)",
    )
    model_group.add_argument(
        "--model",
        default="lmstudio-community/Qwen3.5-9B-Q8_0.gguf",
        help="Model name for the API",
    )
    model_group.add_argument("--temperature", type=float, default=0.6)
    model_group.add_argument("--top-p", type=float, default=0.95)
    model_group.add_argument("--max-tokens", type=int, default=16384)

    exp_group = parser.add_argument_group("Experiment settings")
    exp_group.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of rollout samples (default: 100, paper uses 100)",
    )
    exp_group.add_argument(
        "--dict-size",
        type=int,
        default=100,
        help="Number of words in dictionary (default: 100)",
    )
    exp_group.add_argument(
        "--working-capacity",
        type=int,
        default=1,
        help="Keys per turn / turn complexity K (default: 1)",
    )
    exp_group.add_argument(
        "--num-turns",
        type=int,
        default=100,
        help="Number of turns for standard experiment (default: 100)",
    )
    exp_group.add_argument(
        "--eval-turn",
        type=int,
        default=100,
        help="Evaluation turn for self-conditioning (default: 100)",
    )
    exp_group.add_argument(
        "--error-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Error rates for self-conditioning (default: 0 0.25 0.5 0.75 1.0)",
    )
    exp_group.add_argument("--seed", type=int, default=42)
    exp_group.add_argument("--output-dir", default="output")
    exp_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for batch processing (default: 5)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_config = ModelConfig(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    exp_config = ExperimentConfig(
        experiment=args.experiment,
        num_samples=args.num_samples,
        dict_size=args.dict_size,
        working_capacity=args.working_capacity,
        num_turns=args.num_turns,
        self_cond_eval_turn=args.eval_turn,
        error_rates=args.error_rates,
        seed=args.seed,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )

    if args.experiment == "standard":
        results = run_standard_experiment(model_config, exp_config)
        save_results(results, exp_config, "standard")
    elif args.experiment == "self-conditioning":
        results = run_self_conditioning_experiment(model_config, exp_config)
        save_results(results, exp_config, "self_conditioning")


if __name__ == "__main__":
    main()
