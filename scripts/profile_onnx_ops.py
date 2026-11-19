"""Show which operations take the time when an exported ONNX model runs on the CPU.

Usage:
    python scripts/profile_onnx_ops.py artifacts/onnx/dinov2_lora.onnx

This is how the per-call position embedding resize in the DINOv2 export was found: one
`Resize` node used about half of the node time. Run it on a model before trusting a latency
number that looks wrong. It feeds random pixels at batch size 1, and prints the seven
slowest operation types, as milliseconds per run summed over every node of that type.
"""

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime

WARMUP_RUNS = 5
PROFILED_RUNS = 20
SHOWN_OPERATIONS = 7


def profile_events(model_path: Path, profile_dir: Path) -> list[dict]:
    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_dir / "profile")
    session = onnxruntime.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )
    size = session.get_inputs()[0].shape[2]
    pixels = np.random.default_rng(0).standard_normal((1, 3, size, size)).astype(np.float32)
    for _ in range(WARMUP_RUNS + PROFILED_RUNS):
        session.run(None, {"pixel_values": pixels})
    return json.loads(Path(session.end_profiling()).read_text())


def milliseconds_per_operation(events: list[dict]) -> dict[str, tuple[float, int]]:
    """Per operation type: milliseconds per run, and the number of nodes of that type.

    The profile holds the warm-up runs too, so each node's events are sorted by time and
    only the last PROFILED_RUNS of them are counted.
    """
    durations_by_node = defaultdict(list)
    operation_of_node = {}
    for event in sorted(events, key=lambda e: e.get("ts", 0)):
        if event.get("cat") == "Node" and event["name"].endswith("_kernel_time"):
            durations_by_node[event["name"]].append(event["dur"])
            operation_of_node[event["name"]] = event["args"]["op_name"]

    milliseconds = defaultdict(float)
    nodes = defaultdict(int)
    for name, durations in durations_by_node.items():
        operation = operation_of_node[name]
        milliseconds[operation] += sum(durations[-PROFILED_RUNS:]) / PROFILED_RUNS / 1000
        nodes[operation] += 1
    return {operation: (milliseconds[operation], nodes[operation]) for operation in milliseconds}


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory() as profile_dir:
        events = profile_events(args.model, Path(profile_dir))
    by_operation = milliseconds_per_operation(events)
    total = sum(milliseconds for milliseconds, _ in by_operation.values())
    print(f"{args.model.name}: {total:.1f} ms of node time per run")
    slowest = sorted(by_operation.items(), key=lambda item: -item[1][0])[:SHOWN_OPERATIONS]
    for operation, (milliseconds, nodes) in slowest:
        print(
            f"  {operation:20s} {milliseconds:7.2f} ms  {100 * milliseconds / total:4.1f}%  nodes {nodes}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="an ONNX file made by visual_search.export")
    return parser.parse_args()


if __name__ == "__main__":
    main()
