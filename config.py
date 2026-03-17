"""
Configuration module for the Long Horizon Execution Benchmark.

Defines configuration dataclasses for model connection settings and
experiment parameters used across the standard execution and
self-conditioning experiments.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the LLM server connection."""

    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"
    model: str = "lmstudio-community/Qwen3.5-9B-Q8_0.gguf"
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 16384


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters."""

    experiment: str = "standard"
    num_samples: int = 100
    dict_size: int = 100
    working_capacity: int = 1  # K — keys per turn (turn complexity)
    num_turns: int = 100
    min_value: int = -99
    max_value: int = 99
    error_rates: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    self_cond_eval_turn: int = 100
    output_dir: str = "output"
    seed: int = 42
    num_workers: int = 10
