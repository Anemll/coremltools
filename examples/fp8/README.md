# FP8 on the Neural Engine

Core ML runs FP8 (E4M3) weights and activations on the Neural Engine of M6 and later. FP8 W8A8
halves the weight bytes and runs FP8×FP8 math, which on M6 is as fast as INT8 W8A8 for 1x1 convs
and linear layers, but slower than INT8 for 3x3 convs (see [ResNet50](#resnet50)).

| Script | What it shows |
|---|---|
| [`fp8_mil_example.py`](fp8_mil_example.py) | A 1x1 conv chain built with the MIL builder: FP8 weights (`constexpr_blockwise_shift_scale`), FP8 activations (`quantize` / `dequantize`), op placement, Neural Engine vs CPU, and timing against FP16. |
| [`fp8_pytorch_example.py`](fp8_pytorch_example.py) | A PyTorch feed-forward block: `ct.convert`, then `linear_quantize_activations` and `linear_quantize_weights` with `dtype="fp8e4m3fn"`, compiled and run on the CPU and the Neural Engine. |
| [`fp8_resnet50.py`](fp8_resnet50.py) | Apple's FP16 ResNet50 from the [quantization performance guide](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html), quantized to FP8 and INT8 and compared with Apple's INT8 models on ImageNetV2: accuracy, agreement with FP16, placement. |
| [`time_models.swift`](time_models.swift) | Times compiled models with plain Core ML on the Neural Engine, GPU or CPU, interleaved over several rounds so background load evens out. `swiftc -O -parse-as-library time_models.swift -o time_models`, then `./time_models --units ane,gpu *.mlmodelc`. |
| [`run_fp8_model.swift`](run_fp8_model.swift) | Runs a compiled `.mlmodelc` with plain Core ML (no coremltools): op placement from `MLComputePlan`, then CPU and Neural Engine timing. Build with `xcrun swiftc -parse-as-library -O run_fp8_model.swift -o run_fp8_model`. |

`fp8_mil_example.py --save-dir DIR` and `fp8_pytorch_example.py --output DIR/NAME.mlpackage` keep the
`.mlpackage` and its compiled `.mlmodelc`.

## Requirements

- macOS 27 or later to compile and run. The Neural Engine runs E4M3 on M6 and later; other Macs fall back to a much slower CPU path.
- The iOS 26 opset (Core ML 9), where FP8 lives; there is no newer target. Convert with `minimum_deployment_target=ct.target.iOS26`, or quantize an existing model: `linear_quantize_weights` / `linear_quantize_activations` upgrade older models to iOS 26 when the config uses FP8.
- `pip install ml_dtypes`, which provides the numpy FP8 dtypes coremltools uses.

## Things to know

- **Use `fp8e4m3fn` for the Neural Engine.** `fp8e5m2` compiles and runs on the CPU, but its `quantize` / `dequantize` are not placed on the Neural Engine. The GPU has no FP8 path either.
- **Keep FP8 weight codes within ±240.** The Neural Engine reads FP8 weights like IEEE E4M3, whose largest finite value is 240, so E4M3FN codes above 240 turn into infinity. `OpLinearQuantizerConfig` scales FP8 weights to ±240 by default; pass `fp8_max=448` only for CPU-only models. Activations use the full ±448 range.
- **Two weight encodings** (`OpLinearQuantizerConfig(fp8_encoding=...)`):
  - `"blockwise"` (default): `constexpr_blockwise_shift_scale` with FP8 data, at any granularity. The Core ML compiler in macOS 27 crashes on these weights while it collects compile analytics. `ct.models.utils.compile_model` and `MLModel` compile around it (see `coremltools/models/_fp8_compile.py`), but Xcode cannot compile the `.mlpackage`, and its model viewer shows "Cannot decode metadata": ship the compiled `.mlmodelc` instead.
  - `"palette"`: the FP8 codes index a table of the E4M3 values, with the scale applied around each `conv` / `linear` (`mul(x, a) -> op -> mul(scale / a)`, where `a` is the largest channel scale rounded down to a power of two). It compiles everywhere, including Xcode, and the Neural Engine runs it as native FP8 weights. Per-tensor or per-channel granularity, `fp8e4m3fn`, and weights used by `conv` or `linear` only. The two extra `mul`s per layer are cheap when weights dominate (an LLM FFN at 64 tokens runs as fast as blockwise), but with large activations they cost more than FP8 saves: a 32-layer 512-channel 64×64 conv chain runs at 2.29 ms with palette weights vs 1.47 ms blockwise and 2.14 ms FP16.
- **Rounding of exact ties differs by device.** Converting to FP8 rounds to nearest; for values exactly halfway between two FP8 values, the CPU rounds toward -inf, the Neural Engine away from zero, and coremltools (when it folds constants) to even. Out-of-range values saturate to ±448 everywhere.
- **Errors compound in deep random chains.** One FP8 W8A8 layer is within FP8 precision (a few percent). The 32-layer random chain in `fp8_mil_example.py` shows about 2e-1 between the Neural Engine and the CPU (2e-2 for FP16), because small rounding differences grow layer by layer.

## ResNet50

[`fp8_resnet50.py`](fp8_resnet50.py) starts from Apple's FP16 ResNet50 in the
[quantization performance guide](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)
(batch 8, iOS 16 target), quantizes it with `linear_quantize_weights` / `linear_quantize_activations`
(128 calibration images), and evaluates it on the other 9,872 images of ImageNetV2 (matched frequency).
Apple's INT8 models from the same page are evaluated alongside. Latency is from `time_models.swift`
(best median over interleaved rounds, batch 8) on an M6 with macOS 27.

| Model | Top-1 | Top-5 | Same top-1 as FP16 | Neural Engine | GPU |
|---|---:|---:|---:|---:|---:|
| FP16 (Apple) | 63.30 | 84.60 | – | 3.17 ms | 5.99 ms |
| INT8 weights (Apple, post-training) | 63.46 | 84.72 | 97.3% | 2.86 ms | 16.0 ms |
| INT8 W8A8 (Apple, trained with quantization) | 64.79 | 85.14 | 81.5% | 1.83 ms | 62.1 ms |
| INT8 weights (coremltools) | 63.45 | 84.67 | 97.2% | 2.86 ms | – |
| INT8 W8A8 (coremltools, calibrated) | 62.62 | 84.37 | 92.0% | 1.78 ms | 4.66 ms |
| **FP8 weights** | 62.90 | 84.51 | 94.6% | 3.00 ms | CPU only |
| **FP8 W8A8** | 61.78 | 83.96 | 86.5% | 2.64 ms | CPU only |
| FP8 weights, palette | 62.90 | 84.52 | 94.7% | 3.09 ms | CPU only |
| FP8 W8A8, palette | 62.19 | 84.23 | 86.5% | 3.05 ms | CPU only |

- **FP8 is correct.** Fake-quantizing torchvision's FP32 ResNet50 (the same weights) to per-channel
  E4M3 in PyTorch gives 63.00 top-1 (INT8: 63.39, FP32: 63.28), next to 62.90 from Core ML on the
  Neural Engine. On the same inputs, the Neural Engine and the CPU agree on every top-1 of the FP8
  W8A8 model (logit cosine 0.997; INT8 W8A8: 0.995).
- **FP8 costs a little more accuracy than INT8.** E4M3 keeps 3 mantissa bits, so per-channel FP8
  weights are coarser than per-channel INT8 for most weights of a layer.
- **FP8 W8A8 is 1.2x faster than FP16 in Core ML, INT8 W8A8 1.8x.** On M6 the Neural Engine runs
  FP8×FP8 as fast as INT8 for 1x1 convs and linear layers, but Core ML's 3x3 convs with 64–512
  channels at 28–56 pixels run close to FP16 speed in FP8 (INT8: about 1.8x). Core AI compiles the same
  PyTorch ResNet50 (same quantize/dequantize placement) to 1.97 ms in FP8 W8A8 against 2.64 ms in
  Core ML, with FP16 and INT8 equal in both, so part of the gap is Core ML's compile path rather
  than the hardware. Nothing in the model changed it: scale folding, per-tensor weights, explicit
  padding, shared dequantize ops, no scale normalization around the adds, `.fastPrediction`.
  (An FP8 `zero_point` crashes Core ML.) FP8 fits transformer layers (linear) better than 3x3
  conv nets today.
- **The GPU has no fast FP8.** With `.cpuAndGPU`, Core ML runs FP8 models on the CPU. Core AI runs
  them on the GPU but slowly (ResNet50: FP8 weights 30.6 ms and FP8 W8A8 46.2 ms vs 5.8 ms FP16),
  and its FP8 W8A8 GPU output is wrong (0.2% top-1). With `.all` they run on the
  Neural Engine. Also, on macOS 27 the iOS 26 opset runs `add` and `reshape` on the CPU when the GPU is
  requested (the FP16 ResNet50 re-targeted to iOS 26 takes 17.8 ms on the GPU against 5.99 ms), so
  keep GPU models on an older target.

## Checking placement

`MLComputePlan` reports where each op runs. In an FP8 model on M6, `conv` / `linear`, `quantize`, and `dequantize` should all show `NeuralEngine`:

```python
from coremltools.models.compute_plan import MLComputePlan

plan = MLComputePlan.load_from_path(compiled_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
for op in plan.model_structure.program.functions["main"].block.operations:
    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    if usage is not None:
        print(op.operator_name, type(usage.preferred_compute_device).__name__)
```
