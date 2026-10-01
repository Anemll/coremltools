#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
FP8 (E4M3) weights and activations for a PyTorch model, run on the Neural Engine.

Converts a transformer-style feed-forward block to Core ML, quantizes its activations and its
weights to FP8 with coremltools.optimize, saves the model, and runs it on the CPU and the
Neural Engine next to the PyTorch output.

Needs macOS 27+, ``pip install ml_dtypes``, and an M6 or later for the Neural Engine to run FP8.

    python fp8_pytorch_example.py --encoding blockwise
    python fp8_pytorch_example.py --encoding palette   # also compiles in Xcode
"""

import argparse
import shutil
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct
import coremltools.optimize as cto
from coremltools.models.compute_plan import MLComputePlan


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x):
        return x + self.down(nn.functional.gelu(self.up(x)))


def placement(compiled_path: str) -> dict:
    plan = MLComputePlan.load_from_path(compiled_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is not None:
            device = type(usage.preferred_compute_device).__name__
            counts[(op.operator_name, device.replace("ML", "").replace("ComputeDevice", ""))] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--encoding",
        choices=("blockwise", "palette"),
        default="blockwise",
        help="blockwise: FP8 constexpr weights (compiles through coremltools). "
        "palette: E4M3 table weights (also compiles in Xcode).",
    )
    parser.add_argument("--output", default="ffn_fp8.mlpackage")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = FeedForward(args.dim, 4 * args.dim).eval()
    example = torch.randn(1, args.tokens, args.dim)
    traced = torch.jit.trace(model, example)

    # 1. Convert. FP8 needs the iOS 26 / macOS 26 deployment target (Core ML 9).
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=example.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS26,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    # 2. FP8 activations: calibrate on sample data and insert quantize/dequantize pairs.
    sample_data = [{"x": torch.randn(example.shape).numpy()} for _ in range(8)]
    activation_config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="fp8e4m3fn")
    )
    mlmodel = cto.coreml.linear_quantize_activations(mlmodel, activation_config, sample_data)

    # 3. FP8 weights, per output channel. Codes stay within +-240 by default so that the Neural
    #    Engine reads them exactly (pass fp8_max=448 for CPU-only models).
    weight_config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="fp8e4m3fn",
            granularity="per_channel",
            fp8_encoding=args.encoding,
        )
    )
    mlmodel = cto.coreml.linear_quantize_weights(mlmodel, weight_config)
    mlmodel.save(args.output)
    print(f"saved {args.output}")

    # 4. Compile and run. compile_model / MLModel work around the Core ML compiler crash on FP8
    #    constexpr weights; the palette encoding compiles without it.
    compiled_path = args.output.replace(".mlpackage", ".mlmodelc")
    shutil.rmtree(compiled_path, ignore_errors=True)
    ct.models.utils.compile_model(args.output, compiled_path)
    print(f"compiled {compiled_path}")
    for (op, device), count in sorted(placement(compiled_path).items()):
        print(f"   {op:40s} {device:12s} x{count}")

    x = torch.randn(example.shape)
    with torch.no_grad():
        reference = model(x).numpy()
    for name, units in (("CPU", ct.ComputeUnit.CPU_ONLY), ("Neural Engine", ct.ComputeUnit.CPU_AND_NE)):
        compiled = ct.models.CompiledMLModel(compiled_path, compute_units=units)
        y = compiled.predict({"x": x.numpy()})["y"].astype(np.float32)
        error = np.abs(y - reference).max() / np.abs(reference).max()
        print(f"{name:14s} max|y - torch| / max|torch| = {error:.3e}")


if __name__ == "__main__":
    main()
