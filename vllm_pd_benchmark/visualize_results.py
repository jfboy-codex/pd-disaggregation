from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a vLLM PD benchmark result directory")
    parser.add_argument(
        "result_dir",
        nargs="?",
        help="Result directory containing summary.csv; defaults to the newest directory under ./results",
    )
    parser.add_argument(
        "--scope",
        choices=("dataset", "task"),
        default="dataset",
        help="Plot per-dataset or per-task summary rows",
    )
    parser.add_argument("--output-dir", help="Output directory; defaults to RESULT_DIR/figures")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--show", action="store_true", help="Open figures interactively after saving")
    return parser.parse_args()


def newest_result(root: Path) -> Path:
    candidates = [item for item in root.glob("pd_benchmark_*") if (item / "summary.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No benchmark result containing summary.csv found under {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw in ("", None):
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return math.nan


def maybe_log_axis(axis: Any, values: list[float]) -> None:
    valid = [value for value in values if math.isfinite(value) and value > 0]
    if valid and max(valid) / min(valid) >= 50:
        axis.set_yscale("log")


def style_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def plot_report(plt: Any, rows: list[dict[str, str]], output: Path, dpi: int, scope: str) -> None:
    names = [row["name"] for row in rows]
    positions = list(range(len(rows)))
    width = max(12.0, min(24.0, 5.0 + 0.65 * len(rows)))
    figure, axes = plt.subplots(2, 2, figsize=(width, 9.5), constrained_layout=True)
    figure.suptitle(f"vLLM PD benchmark — {scope} summary", fontsize=16)

    successful = [number(row, "successful_requests") for row in rows]
    failed = [number(row, "failed_requests") for row in rows]
    axes[0, 0].bar(positions, successful, label="Successful", color="#2a9d8f")
    axes[0, 0].bar(positions, failed, bottom=successful, label="Failed", color="#e76f51")
    axes[0, 0].set_title("Request outcomes")
    axes[0, 0].set_ylabel("Requests")
    axes[0, 0].legend(frameon=False)

    bar_width = 0.2
    latency_series = (
        ("TTFT p50", "ttft_s_p50", "#457b9d"),
        ("TTFT p95", "ttft_s_p95", "#1d3557"),
        ("E2E p50", "e2e_s_p50", "#f4a261"),
        ("E2E p95", "e2e_s_p95", "#e76f51"),
    )
    latency_values: list[float] = []
    for offset, (label, key, color) in enumerate(latency_series):
        values = [number(row, key) for row in rows]
        latency_values.extend(values)
        axes[0, 1].bar(
            [position + (offset - 1.5) * bar_width for position in positions],
            values,
            width=bar_width,
            label=label,
            color=color,
        )
    axes[0, 1].set_title("Latency percentiles")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].legend(frameon=False, ncol=2)
    maybe_log_axis(axes[0, 1], latency_values)

    request_tps = [number(row, "request_throughput_rps") for row in rows]
    output_tps = [number(row, "output_token_throughput_tps") for row in rows]
    axes[1, 0].bar(positions, output_tps, color="#6d597a")
    axes[1, 0].set_title("Output-token throughput")
    axes[1, 0].set_ylabel("Tokens/s")
    for position, request_rate, token_rate in zip(positions, request_tps, output_tps):
        if math.isfinite(token_rate):
            axes[1, 0].text(
                position,
                token_rate,
                f"{request_rate:.2f} req/s",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    input_mean = [number(row, "input_tokens_mean") for row in rows]
    output_mean = [number(row, "output_tokens_mean") for row in rows]
    axes[1, 1].bar(
        [position - 0.18 for position in positions],
        input_mean,
        width=0.36,
        label="Input",
        color="#277da1",
    )
    axes[1, 1].bar(
        [position + 0.18 for position in positions],
        output_mean,
        width=0.36,
        label="Output",
        color="#f8961e",
    )
    axes[1, 1].set_title("Mean token count per successful request")
    axes[1, 1].set_ylabel("Tokens/request")
    axes[1, 1].legend(frameon=False)
    maybe_log_axis(axes[1, 1], input_mean + output_mean)

    for axis in axes.flat:
        axis.set_xticks(positions, names, rotation=35, ha="right")
        style_axis(axis)

    figure.savefig(output, dpi=dpi, bbox_inches="tight")


def plot_percentiles(plt: Any, rows: list[dict[str, str]], output: Path, dpi: int) -> None:
    percentiles = (50, 90, 95, 99)
    metrics = (
        ("TTFT", "ttft_s", "Seconds"),
        ("TPOT", "tpot_s", "Seconds/token"),
        ("End-to-end latency", "e2e_s", "Seconds"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    figure.suptitle("Latency percentile profiles", fontsize=16)
    colors = plt.get_cmap("tab10")
    for axis, (title, prefix, unit) in zip(axes, metrics):
        all_values: list[float] = []
        for index, row in enumerate(rows):
            values = [number(row, f"{prefix}_p{percentile}") for percentile in percentiles]
            all_values.extend(values)
            axis.plot(percentiles, values, marker="o", label=row["name"], color=colors(index % 10))
        axis.set_title(title)
        axis.set_xlabel("Percentile")
        axis.set_ylabel(unit)
        axis.set_xticks(percentiles)
        maybe_log_axis(axis, all_values)
        style_axis(axis)
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.savefig(output, dpi=dpi, bbox_inches="tight")


def plot_timeseries(plt: Any, rows: list[dict[str, str]], output: Path, dpi: int) -> None:
    times = [number(row, "bucket_start_s") for row in rows]
    output_tps = [number(row, "output_token_throughput_tps") for row in rows]
    active = [number(row, "active_requests") for row in rows]
    failed = [number(row, "failed_requests") for row in rows]

    figure, first = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    second = first.twinx()
    first.plot(times, output_tps, color="#6d597a", linewidth=1.4, label="Output tokens/s")
    second.plot(times, active, color="#277da1", linewidth=1.2, label="Active requests")
    failure_times = [time for time, value in zip(times, failed) if value > 0]
    failure_values = [rate for rate, value in zip(output_tps, failed) if value > 0]
    if failure_times:
        first.scatter(failure_times, failure_values, color="#e76f51", marker="x", label="Bucket with failures")
    first.set_title("Measured-run timeline")
    first.set_xlabel("Elapsed time (s)")
    first.set_ylabel("Output tokens/s")
    second.set_ylabel("Active requests")
    style_axis(first)
    second.spines["top"].set_visible(False)
    lines = first.get_lines() + second.get_lines()
    labels = [line.get_label() for line in lines]
    handles, scatter_labels = first.get_legend_handles_labels()
    for handle, label in zip(handles, scatter_labels):
        if label not in labels:
            lines.append(handle)
            labels.append(label)
    first.legend(lines, labels, frameon=False, loc="upper right")
    figure.savefig(output, dpi=dpi, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required: python -m pip install matplotlib") from exc

    script_dir = Path(__file__).resolve().parent
    result_dir = Path(args.result_dir).expanduser().resolve() if args.result_dir else newest_result(script_dir / "results")
    summary_path = result_dir / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"summary.csv not found: {summary_path}")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csv(summary_path)
    rows = [row for row in summary if row.get("scope") == args.scope]
    if not rows:
        raise SystemExit(f"No scope={args.scope!r} rows found in {summary_path}")

    generated = []
    report_path = output_dir / f"{args.scope}_report.png"
    plot_report(plt, rows, report_path, args.dpi, args.scope)
    generated.append(report_path)

    percentile_path = output_dir / f"{args.scope}_latency_percentiles.png"
    plot_percentiles(plt, rows, percentile_path, args.dpi)
    generated.append(percentile_path)

    timeseries_path = result_dir / "timeseries.csv"
    if timeseries_path.exists():
        timeline_output = output_dir / "timeseries.png"
        plot_timeseries(plt, read_csv(timeseries_path), timeline_output, args.dpi)
        generated.append(timeline_output)

    print(f"Result directory: {result_dir}")
    for path in generated:
        print(f"Generated: {path}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()


