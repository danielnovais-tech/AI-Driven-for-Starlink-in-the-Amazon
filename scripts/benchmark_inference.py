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


def prune_model(
    model: torch.nn.Module, prune_ratio: float = 0.3
) -> torch.nn.Module:
    """
    Apply L1-unstructured pruning to all Linear layers and make it permanent.

    Removes ``prune_ratio`` fraction of the smallest-magnitude weights in
    each Linear layer, replacing them with zeros.  The pruning mask is then
    made permanent (weights are zeroed in-place and the mask hook is removed)
    so the pruned model behaves like a standard sparse model.

    Args:
        model:       PyTorch model to prune (modified in-place and returned).
        prune_ratio: Fraction of connections to prune per layer (0.0–1.0).

    Returns:
        The pruned model (same object, modified in-place).
    """
    try:
        import torch.nn.utils.prune as _prune
    except ImportError:
        print("  [SKIP] torch.nn.utils.prune not available; pruning skipped.")
        return model

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            _prune.l1_unstructured(module, name="weight", amount=prune_ratio)
            # Make the pruning permanent (remove the re-parametrisation hook)
            _prune.remove(module, "weight")
    return model


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


def export_tensorrt_stub(onnx_path: str, trt_path: str, fp16: bool = True) -> bool:
    """
    Convert an ONNX model to TensorRT engine (requires ``tensorrt`` package).

    This function attempts to build a TensorRT engine from the exported ONNX
    file using the Python ``tensorrt`` API.  If ``tensorrt`` is not installed
    (typical in CPU-only environments), a clear ``[SKIP]`` message is printed
    and the function returns ``False``.

    Hardware recommendations:
        - **NVIDIA Jetson Orin** (8 TOPS INT8): target for on-board satellite
          use; expect < 1 ms inference latency for the BeamformingNetwork.
          **Power budget** ≈ 10–15 W; **thermal envelope** limited to < 85 °C
          junction temperature in a passively-cooled satellite enclosure.
        - **Jetson AGX Xavier / Orin NX**: recommended for GNN agents
          (higher memory bandwidth required for graph operations).
          Draws 20–30 W at full load; suitable for larger LEO platforms.
        - **x86 server with T4/A10G**: use for ground-station deployments;
          full TensorRT FP16 or INT8 calibration possible.  No power/thermal
          constraints relevant for satellite deployment.

    Workflow::

        onnx_path = "/tmp/beamforming.onnx"
        trt_path  = "/tmp/beamforming.engine"
        export_onnx(model, dummy_input, onnx_path)
        ok = export_tensorrt_stub(onnx_path, trt_path, fp16=True)
        # Deploy trt_path on target NVIDIA hardware with trt.Runtime

    Args:
        onnx_path: Path to an ONNX model exported by :func:`export_onnx`.
        trt_path:  Destination path for the serialised TensorRT engine.
        fp16:      If ``True``, enable FP16 precision mode (recommended for
                   embedded NVIDIA GPUs; reduces latency by ~2×).

    Returns:
        ``True`` if the engine was built and saved; ``False`` if
        ``tensorrt`` is unavailable or the build fails.
    """
    try:
        import tensorrt as trt  # type: ignore[import]
    except ImportError:
        print(
            "  [SKIP] tensorrt not installed; skip TensorRT build.\n"
            "         Install on NVIDIA hardware: pip install tensorrt\n"
            "         or via JetPack SDK Manager on Jetson devices."
        )
        return False

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  TRT parse error: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("  [FAIL] TensorRT engine build failed.")
        return False

    with open(trt_path, "wb") as f:
        f.write(engine)

    print(f"  TensorRT engine saved to {trt_path}  (fp16={fp16})")
    return True


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

def benchmark_gnn_pytorch(n_sats: int = 5, hidden: int = 64, n_runs: int = 500) -> float:
    """
    Measure mean per-inference latency (ms) for :class:`~agents.gnn_ppo_agent.GNNPPOAgent`.

    Requires ``torch_geometric``.  Returns ``float('nan')`` if unavailable.

    Args:
        n_sats:  Number of satellite nodes in the synthetic graph.
        hidden:  Hidden dimension for the GAT layers.
        n_runs:  Number of timed forward passes.

    Returns:
        Average latency in ms, or ``nan`` if ``torch_geometric`` is absent.
    """
    try:
        import torch as _torch
        from torch_geometric.data import HeteroData
        from agents.gnn_ppo_agent import GNNPPOAgent
    except ImportError:
        print("  [SKIP] torch_geometric not installed; GNN benchmark skipped.")
        return float("nan")

    agent = GNNPPOAgent(node_features=4, hidden=hidden)
    agent.net.eval()

    # Build a synthetic graph
    data = HeteroData()
    data["sat"].x = _torch.randn(n_sats, 4)
    data["sat"].num_nodes = n_sats
    data["ground_station"].x = _torch.zeros(1, 4)
    data["ground_station"].num_nodes = 1
    src = _torch.arange(n_sats, dtype=_torch.long)
    dst = _torch.zeros(n_sats, dtype=_torch.long)
    data["sat", "to", "ground_station"].edge_index = _torch.stack([src, dst], dim=0)

    # Warm-up
    for _ in range(20):
        agent.get_action(data, deterministic=True)

    start = time.perf_counter()
    for _ in range(n_runs):
        agent.get_action(data, deterministic=True)
    end = time.perf_counter()
    return (end - start) / n_runs * 1000.0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark BeamformingNetwork inference latency.")
    parser.add_argument("--n-runs", type=int, default=1000, help="Number of timed iterations.")
    parser.add_argument("--state-dim", type=int, default=7, help="State vector dimension.")
    parser.add_argument("--action-dim", type=int, default=4, help="Action vector dimension.")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer width.")
    parser.add_argument("--onnx-path", default="/tmp/beamforming_network.onnx", help="ONNX export path.")
    parser.add_argument("--trt-path", default="/tmp/beamforming_network.engine",
                        help="TensorRT engine output path (requires tensorrt package).")
    parser.add_argument("--gnn-sats", type=int, default=5,
                        help="Number of satellite nodes for GNN benchmark.")
    parser.add_argument("--gnn-hidden", type=int, default=64,
                        help="Hidden dimension for GNN benchmark.")
    parser.add_argument("--prune-ratio", type=float, default=0.3,
                        help="L1 unstructured pruning ratio for Linear layers (0–1, default 0.3).")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    model = BeamformingNetwork(args.state_dim, args.action_dim, args.hidden)
    model.eval()

    dummy_input = torch.randn(1, args.state_dim)
    dummy_numpy = dummy_input.numpy().astype(np.float32)

    print(f"{'Backend':<40} {'Avg latency (ms)':>18}")
    print("-" * 60)

    # --- PyTorch FP32 ---
    t_pt = benchmark_pytorch(model, dummy_input, args.n_runs)
    print(f"{'PyTorch CPU (float32)':<40} {t_pt:>17.3f}")

    # --- L1 pruned (make a fresh copy so the original model is untouched) ---
    import copy as _copy
    model_pruned = prune_model(_copy.deepcopy(model), prune_ratio=args.prune_ratio)
    t_pruned = benchmark_pytorch(model_pruned, dummy_input, args.n_runs)
    print(f"{'PyTorch CPU (L1 pruned {:.0%})':<40} {t_pruned:>17.3f}".format(args.prune_ratio))

    # --- PyTorch INT8 dynamic quantisation ---
    model_q = quantize_dynamic(model)
    t_q = benchmark_pytorch(model_q, dummy_input, args.n_runs)
    print(f"{'PyTorch CPU (int8 dynamic quant)':<40} {t_q:>17.3f}")

    # --- ONNX Runtime ---
    t_onnx = float("nan")
    try:
        export_onnx(model, dummy_input, args.onnx_path)
        t_onnx = benchmark_onnx(args.onnx_path, dummy_numpy, args.n_runs)
        if not np.isnan(t_onnx):
            print(f"{'ONNX Runtime CPU':<40} {t_onnx:>17.3f}")
    except Exception as exc:
        print(f"  [SKIP] ONNX export/benchmark skipped: {exc}")

    # --- GNN agent ---
    t_gnn = benchmark_gnn_pytorch(
        n_sats=args.gnn_sats, hidden=args.gnn_hidden, n_runs=min(args.n_runs, 500)
    )
    if not np.isnan(t_gnn):
        print(f"{'GNN PPO (torch_geometric CPU)':<40} {t_gnn:>17.3f}")

    print()
    threshold_ms = 500.0
    all_ok = all(
        v < threshold_ms
        for v in [t_pt, t_pruned, t_q, t_gnn]
        if not np.isnan(v)
    )
    status = "PASS" if all_ok else "FAIL"
    print(f"Latency target < {threshold_ms} ms: [{status}]")

    # --- TensorRT export (NVIDIA hardware only) ---
    if not np.isnan(t_onnx):
        print()
        print("Attempting TensorRT engine build …")
        export_tensorrt_stub(args.onnx_path, args.trt_path, fp16=True)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
