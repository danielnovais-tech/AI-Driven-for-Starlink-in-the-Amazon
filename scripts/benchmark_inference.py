#!/usr/bin/env python3
"""
Benchmark inference latency of a BeamformingNetwork agent.

Tests three backends:
    1. PyTorch (CPU, float32)
    2. PyTorch with dynamic int8 quantisation
    3. ONNX Runtime (CPU)

The target latency for the online beam controller is < 500 ms; this script
verifies that the neural network inference step is well within that budget
(typically < 10 ms on modern hardware).

Usage:
    python scripts/benchmark_inference.py [--n-runs 1000] [--state-dim 7]

Requirements:
    pip install torch onnx onnxruntime
"""

import argparse
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make src importable when running from the repository root
# ---------------------------------------------------------------------------
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.networks import BeamformingNetwork


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def benchmark_pytorch(model: torch.nn.Module, input_tensor: torch.Tensor, n_runs: int = 1000) -> float:
    """
    Measure average per-inference latency (ms) for a PyTorch model.

    Args:
        model:        Trained (or randomly initialised) PyTorch model.
        input_tensor: Dummy input tensor.
        n_runs:       Number of timed inference runs.

    Returns:
        Average latency in ms.
    """
    model.eval()
    with torch.no_grad():
        # Warm-up runs to avoid cold-start effects
        for _ in range(50):
            _ = model(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) / n_runs * 1000.0  # ms


def quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply dynamic int8 quantisation to all Linear layers.

    Args:
        model: PyTorch model to quantise.

    Returns:
        Quantised copy of the model.
    """
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )


def export_onnx(model: torch.nn.Module, input_tensor: torch.Tensor, onnx_path: str) -> None:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model:        Trained PyTorch model.
        input_tensor: Representative input tensor.
        onnx_path:    Destination .onnx file path.
    """
    model.eval()
    torch.onnx.export(
        model,
        input_tensor,
        onnx_path,
        input_names=["state"],
        output_names=["mean", "value"],
        dynamic_axes={"state": {0: "batch_size"}},
        opset_version=17,
    )


def benchmark_onnx(onnx_path: str, input_numpy: np.ndarray, n_runs: int = 1000) -> float:
    """
    Measure average per-inference latency (ms) for an ONNX Runtime session.

    Args:
        onnx_path:    Path to the exported .onnx file.
        input_numpy:  Numpy input array.
        n_runs:       Number of timed inference runs.

    Returns:
        Average latency in ms, or float('nan') if onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not installed.")
        return float("nan")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(50):
        session.run(None, {input_name: input_numpy})

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {input_name: input_numpy})
    end = time.perf_counter()
    return (end - start) / n_runs * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark BeamformingNetwork inference latency.")
    parser.add_argument("--n-runs", type=int, default=1000, help="Number of timed iterations.")
    parser.add_argument("--state-dim", type=int, default=7, help="State vector dimension.")
    parser.add_argument("--action-dim", type=int, default=4, help="Action vector dimension.")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer width.")
    parser.add_argument("--onnx-path", default="/tmp/beamforming_network.onnx", help="ONNX export path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    model = BeamformingNetwork(args.state_dim, args.action_dim, args.hidden)
    model.eval()

    dummy_input = torch.randn(1, args.state_dim)
    dummy_numpy = dummy_input.numpy().astype(np.float32)

    print(f"{'Backend':<35} {'Avg latency (ms)':>18}")
    print("-" * 55)

    # --- PyTorch FP32 ---
    t_pt = benchmark_pytorch(model, dummy_input, args.n_runs)
    print(f"{'PyTorch CPU (float32)':<35} {t_pt:>17.3f}")

    # --- PyTorch INT8 dynamic quantisation ---
    model_q = quantize_dynamic(model)
    t_q = benchmark_pytorch(model_q, dummy_input, args.n_runs)
    print(f"{'PyTorch CPU (int8 dynamic quant)':<35} {t_q:>17.3f}")

    # --- ONNX Runtime ---
    try:
        export_onnx(model, dummy_input, args.onnx_path)
        t_onnx = benchmark_onnx(args.onnx_path, dummy_numpy, args.n_runs)
        if not np.isnan(t_onnx):
            print(f"{'ONNX Runtime CPU':<35} {t_onnx:>17.3f}")
    except Exception as exc:
        print(f"  [SKIP] ONNX export/benchmark skipped: {exc}")

    print()
    threshold_ms = 500.0
    all_ok = all(
        v < threshold_ms
        for v in [t_pt, t_q]
        if not np.isnan(v)
    )
    status = "PASS" if all_ok else "FAIL"
    print(f"Latency target < {threshold_ms} ms: [{status}]")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
