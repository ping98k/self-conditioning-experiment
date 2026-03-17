"""
Dataset generation module for the Long Horizon Execution Benchmark.

Handles creation of word-to-integer dictionaries and key sequences
for the dict-sum task. Generates samples with ground-truth running
sums for evaluation. Words are loaded from words_alpha.txt, matching
the original paper's repository.
"""

from pathlib import Path

import numpy as np

_WORDS_FILE = Path(__file__).parent / "words_alpha.txt"


def _load_words() -> list[str]:
    """Load five-letter words from words_alpha.txt."""
    return [w.strip() for w in _WORDS_FILE.read_text().splitlines() if w.strip()]


FIVE_LETTER_WORDS = _load_words()


def generate_sample(
    dict_size: int,
    min_value: int,
    max_value: int,
    num_turns: int,
    working_capacity: int,
    rng: np.random.Generator,
) -> dict:
    """Generate a single sample for the dict-sum task.

    Returns a dict with keys: dictionary, turns, ground_truth.
    """
    selected_words = list(
        rng.choice(FIVE_LETTER_WORDS, size=dict_size, replace=False)
    )
    values = rng.integers(min_value, max_value + 1, size=dict_size).tolist()
    dictionary = dict(zip(selected_words, values))

    total_steps = num_turns * working_capacity
    keys = list(dictionary.keys())
    key_indices = rng.integers(0, len(keys), size=total_steps)
    key_sequence = [keys[i] for i in key_indices]

    turns = []
    ground_truth = []
    running_sum = 0
    for t in range(num_turns):
        start = t * working_capacity
        end = start + working_capacity
        turn_keys = key_sequence[start:end]
        turns.append(turn_keys)
        for k in turn_keys:
            running_sum += dictionary[k]
        ground_truth.append(running_sum)

    return {
        "dictionary": dictionary,
        "turns": turns,
        "ground_truth": ground_truth,
    }
