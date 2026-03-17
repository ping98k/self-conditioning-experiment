Long Horizon Execution Benchmark

> Paper: "The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs"  
> Repo: https://github.com/long-horizon-execution/measuring-execution  
> Dataset: https://huggingface.co/datasets/arvindh75/Long-Horizon-Execution


## Mission

Reproduce the two core experiments from the paper:

1. **Standard execution benchmark** — measure how turn accuracy degrades as task length increases across model sizes (Figure 3/4 in paper)
2. **Self-conditioning experiment** — inject artificial error histories at varying rates (0%, 25%, 50%, 75%, 100%) and measure turn accuracy at turn 100 (Figure 5 in paper)


## Project Structure

```
main.py              # CLI entry point (argparse with standard / self-conditioning subcommands)
config.py            # ModelConfig and ExperimentConfig dataclasses
dataset.py           # Dictionary + key sequence generation with ground truth
prompts.py           # System prompt construction (matches paper Appendix E)
llm_client.py        # OpenAI-compatible client for llama.cpp server
experiment.py        # Standard + self-conditioning experiment runners (ThreadPoolExecutor)
evaluation.py        # Answer parsing, turn/task accuracy, horizon length (H_0.5)
ui.py                # Gradio dashboard for plotting results
words_alpha.txt      # 101 five-letter words (from paper's repo)
pyproject.toml       # uv project config (openai, numpy, gradio)
output/              # Timestamped JSON result files
```


## Usage

```bash
# Install dependencies
uv sync

# Standard execution benchmark (Figure 3/4)
uv run main.py standard --num-samples 100 --num-turns 100

# Self-conditioning experiment (Figure 5)
uv run main.py self-conditioning --num-samples 100 --eval-turn 100

# Adjust turn complexity (K keys per turn)
uv run main.py standard --working-capacity 2 --num-turns 100

# Control parallelism
uv run main.py standard --num-workers 4

# Launch results dashboard
uv run ui.py
# Opens at http://127.0.0.1:7860
```


## Coding
- use uv
- always add multi line docstrings to top of files


## Data
- use llama.cpp server with model "lmstudio-community\Qwen3.5-9B-Q8_0.gguf"
- i started server in "http://0.0.0.0:8080"
