#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
FP8 (E4M3) weights and activations with the MIL builder, run on the Neural Engine.

Builds a chain of 1x1 convolutions with FP8 weights (``constexpr_blockwise_shift_scale``) and FP8
activations between the layers (``quantize`` -> ``dequantize``), next to the same chain in FP16.
Then compiles both, checks where each op runs, compares the Neural Engine output with the CPU, and
times both models.

Needs macOS 27+, ``pip install ml_dtypes``, and an M6 or later for the Neural Engine to run FP8.

    python fp8_mil_example.py --channels 512 --layers 32 --save-dir models
"""

import argparse
import os
import shutil
import time
from collections import Counter

import ml_dtypes
import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan

# The Neural Engine reads FP8 weights like IEEE E4M3, whose largest finite value is 240. Larger
# E4M3FN codes (up to 448) become infinity there, so the weights are scaled to +-240.
FP8_WEIGHT_MAX = 240.0


def fp8_weight(w: np.ndarray):
    """Symmetric per-output-channel FP8 quantization. Returns (FP8 data, fp16 scale)."""
    scale = (np.abs(w).reshape(w.shape[0], -1).max(axis=1) / FP8_WEIGHT_MAX).astype(np.float16)
    scale = scale.reshape((-1,) + (1,) * (w.ndim - 1))
    data = np.clip(w / scale.astype(np.float32), -FP8_WEIGHT_MAX, FP8_WEIGHT_MAX)
    return data.astype(ml_dtypes.float8_e4m3fn), scale


def activation_scales(x: np.ndarray, weights):
    """Per-tensor FP8 activation scales from a float pass: max(|a|) maps to 448."""
    scales = []
    for w in weights[:-1]:
        x = np.einsum("oi,bihw->bohw", w[:, :, 0, 0], x)
        scales.append(np.float16(np.abs(x).max() / 448.0))
    return scales


def build_program(weights, act_scales, channels, spatial, fp8):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, channels, spatial, spatial), dtype=types.fp16)],
        opset_version=ct.target.iOS26,  # FP8 needs the iOS 26 / macOS 26 opset (Core ML 9)
    )
    def prog(x):
        for i, w in enumerate(weights):
            if fp8:
                data, scale = fp8_weight(w)
                w = mb.constexpr_blockwise_shift_scale(data=data, scale=scale, name=f"w{i}")
            else:
                w = w.astype(np.float16)
            x = mb.conv(x=x, weight=w, name=f"conv{i}")
            if fp8 and i < len(weights) - 1:
                # FP8 activations between layers; FP8 has no zero point.
                x = mb.quantize(input=x, scale=act_scales[i], output_dtype="fp8e4m3fn", name=f"q{i}")
                x = mb.dequantize(input=x, scale=act_scales[i], name=f"dq{i}")
        return x

    return prog


def placement(compiled_path: str) -> dict:
    plan = MLComputePlan.load_from_path(compiled_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is not None:
            device = type(usage.preferred_compute_device).__name__
            counts[(op.operator_name, device.replace("ML", "").replace("ComputeDevice", ""))] += 1
    return dict(counts)


def median_ms(model, feed, iters):
    for _ in range(5):
        model.predict(feed)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        model.predict(feed)
        times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--spatial", type=int, default=64)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--save-dir", help="Also save each model as .mlpackage and compiled .mlmodelc in this folder."
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    c = args.channels
    weights = [
        (rng.standard_normal((c, c, 1, 1)) / np.sqrt(c)).astype(np.float32) for _ in range(args.layers)
    ]
    x = rng.standard_normal((1, c, args.spatial, args.spatial)).astype(np.float16)
    act_scales = activation_scales(x.astype(np.float32), weights)
    flops = 2.0 * c * c * args.spatial * args.spatial * args.layers

    for name, fp8 in (("fp16", False), ("fp8 W8A8", True)):
        prog = build_program(weights, act_scales, c, args.spatial, fp8)
        # ct.convert loads (compiles) the model. For FP8 weights, coremltools compiles around a crash
        # in the Core ML compiler; see coremltools.models._fp8_compile.
        mlmodel = ct.convert(
            prog, minimum_deployment_target=ct.target.iOS26, compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        compiled_path = mlmodel.get_compiled_model_path()
        print(f"\n== {name}")
        if args.save_dir:
            stem = os.path.join(args.save_dir, f"conv{c}x{args.layers}_{'fp8_w8a8' if fp8 else 'fp16'}")
            for path in (stem + ".mlpackage", stem + ".mlmodelc"):
                shutil.rmtree(path, ignore_errors=True)
            mlmodel.save(stem + ".mlpackage")
            ct.models.utils.compile_model(stem + ".mlpackage", stem + ".mlmodelc")
            print(f"   saved {stem}.mlpackage and {stem}.mlmodelc")
        for (op, device), count in sorted(placement(compiled_path).items()):
            print(f"   {op:40s} {device:12s} x{count}")

        feed = {"x": x}
        y_ane = list(mlmodel.predict(feed).values())[0].astype(np.float32)
        cpu = ct.models.CompiledMLModel(compiled_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        y_cpu = list(cpu.predict(feed).values())[0].astype(np.float32)
        ms = median_ms(mlmodel, feed, args.iters)
        print(f"   Neural Engine vs CPU: max|diff| / max|y| = {np.abs(y_ane - y_cpu).max() / np.abs(y_cpu).max():.3e}")
        print(f"   median {ms:.3f} ms per call ({flops / ms / 1e9:.2f} TFLOPS, including call overhead)")


if __name__ == "__main__":
    main()
