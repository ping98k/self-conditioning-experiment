"""
Gradio UI for visualizing Long Horizon Execution Benchmark results.

Loads experiment output JSON files and plots turn accuracy, task accuracy,
format failure rate (standard experiment), and accuracy vs error rate
(self-conditioning experiment), reproducing Figures 3-5 from the paper.

Usage:
    uv run ui.py
"""

import glob
import json
import os

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Paper color scheme
# ---------------------------------------------------------------------------

ERROR_RATE_COLORS = {
    0.0: "#2ca02c",     # Green — 0% Error Rate
    0.25: "#a8d600",    # Yellow-Green — 25%
    0.5: "#f0c000",     # Gold — 50%
    0.75: "#ff7f0e",    # Orange — 75%
    1.0: "#d62728",     # Red — 100%
}
ORIGINAL_COLOR = "#1f77b4"  # Blue

# Distinct colors for multi-model comparison
MODEL_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#7f7f7f",  # gray
]


def list_result_files(output_dir: str = "output") -> list[str]:
    """Find all result JSON files in the output directory."""
    if not os.path.isdir(output_dir):
        return []
    files = sorted(glob.glob(os.path.join(output_dir, "*.json")), reverse=True)
    return [os.path.basename(f) for f in files]


def load_result(filename: str, output_dir: str = "output") -> dict | None:
    """Load a result JSON file."""
    path = os.path.join(output_dir, filename)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Moving average smoother."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _label_from_filename(filename: str) -> str:
    """Extract a short display label from a result filename.

    Strips the experiment prefix and timestamp suffix, leaving the
    model/run identifier. Falls back to the full filename.
    """
    name = filename.removesuffix(".json")
    for prefix in ("standard_", "self_conditioning_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Try to strip trailing timestamp like _20260316_123456
    parts = name.rsplit("_", 2)
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        name = "_".join(parts[:-2])
    if not name:
        return filename
    return name


# ---------------------------------------------------------------------------
# Standard experiment plots (Figures 3/4) — multi-model overlay
# ---------------------------------------------------------------------------

def plot_standard_multi(filenames: list[str], window: int):
    """Generate matplotlib figures overlaying multiple standard experiments."""
    if not filenames:
        return None, None, "Select one or more result files."

    fig_turn, ax_turn = plt.subplots(figsize=(8, 5))
    fig_task, ax_task = plt.subplots(figsize=(8, 5))

    summaries = []
    max_turns = 0

    for i, fname in enumerate(filenames):
        data = load_result(fname)
        if data is None or data.get("experiment") != "standard":
            continue

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        label = _label_from_filename(fname)

        metrics = data["data"]["metrics"]
        num_turns = metrics["num_turns"]
        max_turns = max(max_turns, num_turns)
        turns = np.arange(1, num_turns + 1)

        turn_acc = np.array(metrics["turn_accuracy"])
        task_acc = np.array(metrics["task_accuracy"])
        turn_acc_smooth = _smooth(turn_acc, window)
        task_acc_smooth = _smooth(task_acc, window)

        # Turn accuracy
        ax_turn.scatter(
            turns, turn_acc,
            s=10, alpha=0.25, color=color, zorder=2,
        )
        ax_turn.plot(
            turns, turn_acc_smooth,
            linewidth=2.5, color=color, zorder=3, label=label,
        )

        # Task accuracy
        ax_task.scatter(
            turns, task_acc,
            s=10, alpha=0.25, color=color, zorder=2,
        )
        ax_task.plot(
            turns, task_acc_smooth,
            linewidth=2.5, color=color, zorder=3, label=label,
        )

        # Summary line
        horizon = metrics["horizon_length_0.5"]
        n = metrics["num_samples"]
        summaries.append(
            f"**{label}** — {n} samples, {num_turns} turns, "
            f"H₀.₅={horizon}, mean acc={np.mean(turn_acc):.4f}"
        )

    if not summaries:
        plt.close(fig_turn)
        plt.close(fig_task)
        return None, None, "No valid standard experiment data found."

    # p=0.99 baseline on task accuracy plot
    baseline = np.array([0.99 ** t for t in range(1, max_turns + 1)])
    ax_task.plot(
        np.arange(1, max_turns + 1), baseline,
        linewidth=1.5, color="gray", linestyle="--", zorder=1,
        label="p=0.99 baseline",
    )

    for ax, ylabel in [(ax_turn, "Turn Accuracy"), (ax_task, "Task Accuracy")]:
        ax.set_xlabel("Task Length", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(1, max_turns)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left" if ylabel == "Turn Accuracy" else "upper right", fontsize=9)
        ax.grid(True, alpha=0.2)

    fig_turn.tight_layout()
    fig_task.tight_layout()

    summary = "\n\n".join(summaries)
    return fig_turn, fig_task, summary


# ---------------------------------------------------------------------------
# Self-conditioning experiment plots (Figure 5) — multi-model overlay
# ---------------------------------------------------------------------------

def plot_self_conditioning_multi(filenames: list[str]):
    """Generate matplotlib figure overlaying multiple self-conditioning experiments."""
    if not filenames:
        return None, "Select one or more result files."

    fig, ax = plt.subplots(figsize=(7, 5))
    model_rows = []

    for i, fname in enumerate(filenames):
        data = load_result(fname)
        if data is None or data.get("experiment") != "self_conditioning":
            continue

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        label = _label_from_filename(fname)
        results = data["data"]

        rates = []
        accuracies = []
        std_errs = []

        for rate_str in sorted(results.keys(), key=float):
            rate = float(rate_str)
            rate_data = results[rate_str]
            rates.append(rate)
            accuracies.append(rate_data["accuracy"])

            samples = rate_data.get("samples", [])
            if samples:
                correct = np.array([s["correct"] for s in samples], dtype=float)
                se = np.std(correct, ddof=1) / np.sqrt(len(correct)) if len(correct) > 1 else 0
                std_errs.append(se)
            else:
                std_errs.append(0)

        rates_arr = np.array(rates)
        acc_arr = np.array(accuracies)
        se_arr = np.array(std_errs)

        ax.errorbar(
            rates_arr, acc_arr,
            yerr=se_arr,
            fmt="o-",
            linewidth=2.5,
            markersize=8,
            color=color,
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            zorder=3 + i,
            label=label,
        )

        # Linear slope of accuracy vs error rate
        slope = float(np.polyfit(rates_arr, acc_arr, 1)[0]) if len(rates_arr) >= 2 else 0.0

        model_rows.append({
            "label": label,
            "accs": {r: a for r, a in zip(rates, accuracies)},
            "slope": slope,
        })

    if not model_rows:
        plt.close(fig)
        return None, "No valid self-conditioning experiment data found."

    ax.set_xlabel("Induced Error Rate", fontsize=12)
    ax.set_ylabel("Turn 100 Accuracy", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    # Build one combined table for all models, sorted by slope (least negative = most robust first)
    all_rates = sorted({r for row in model_rows for r in row["accs"]})
    model_rows.sort(key=lambda r: r["slope"], reverse=True)
    header_cols = " | ".join(f"{r:.0%}" for r in all_rates)
    summary = f"| Model | {header_cols} | Slope |\n"
    summary += "|---" * (len(all_rates) + 2) + "|\n"
    for row in model_rows:
        acc_cols = " | ".join(
            f"{row['accs'].get(r, 0):.4f}" for r in all_rates
        )
        summary += f"| {row['label']} | {acc_cols} | {row['slope']:+.4f} |\n"

    return fig, summary


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def _std_files():
    return [f for f in list_result_files() if f.startswith("standard_")]


def _sc_files():
    return [f for f in list_result_files() if f.startswith("self_conditioning_")]


def refresh_files():
    return (
        gr.update(choices=_std_files(), value=[]),
        gr.update(choices=_sc_files(), value=[]),
    )


def on_standard_select(filenames, window):
    if not filenames:
        return None, None, "Select one or more result files."
    return plot_standard_multi(filenames, window)


def on_sc_select(filenames):
    if not filenames:
        return None, "Select one or more result files."
    return plot_self_conditioning_multi(filenames)


def build_app():
    with gr.Blocks(title="Long Horizon Execution Benchmark") as app:
        gr.Markdown("# Long Horizon Execution Benchmark — Results Viewer")
        gr.Markdown(
            "Visualize results from the standard execution and "
            "self-conditioning experiments (Figures 3–5 from the paper).  \n"
            "**Select multiple files to compare models on the same plot.**"
        )

        refresh_btn = gr.Button("🔄 Refresh file list", size="sm")

        with gr.Tabs():
            # ---- Standard Experiment Tab ----
            with gr.Tab("Standard Experiment"):
                with gr.Row():
                    with gr.Column(scale=3):
                        std_files = gr.CheckboxGroup(
                            label="Result files (select multiple to compare)",
                            choices=_std_files(),
                            interactive=True,
                        )
                    with gr.Column(scale=1):
                        window_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Smoothing window",
                        )

                std_summary = gr.Markdown("Select one or more result files to view.")

                with gr.Row():
                    turn_plot = gr.Plot(label="Turn Accuracy (Figure 4d)")
                    task_plot = gr.Plot(label="Task Accuracy (Figure 4c)")

                std_files.change(
                    on_standard_select,
                    inputs=[std_files, window_slider],
                    outputs=[turn_plot, task_plot, std_summary],
                )
                window_slider.change(
                    on_standard_select,
                    inputs=[std_files, window_slider],
                    outputs=[turn_plot, task_plot, std_summary],
                )

            # ---- Self-Conditioning Tab ----
            with gr.Tab("Self-Conditioning Experiment"):
                sc_files = gr.CheckboxGroup(
                    label="Result files (select multiple to compare)",
                    choices=_sc_files(),
                    interactive=True,
                )

                sc_summary = gr.Markdown("Select one or more result files to view.")

                sc_plot = gr.Plot(
                    label="Accuracy vs Induced Error Rate (Figure 5b)"
                )

                sc_files.change(
                    on_sc_select,
                    inputs=[sc_files],
                    outputs=[sc_plot, sc_summary],
                )

        # Wire refresh button
        refresh_btn.click(
            refresh_files,
            outputs=[std_files, sc_files],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
