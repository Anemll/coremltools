#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
FP8 (E4M3) ResNet50 from the Core ML quantization performance guide, checked on ImageNetV2.

Starts from Apple's FP16 ResNet50 (the "Float16" row of
https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html), quantizes it
to FP8 and INT8 with coremltools.optimize, and compares every variant with Apple's own INT8
models from the same page: top-1/top-5 accuracy, agreement with FP16 and placement. Time the
compiled models with time_models.swift (Python's predict() adds the copy of the 4.8 MB input).

Downloads (see README.md):
    ResNet50.mlpackage, ResNet50WeightOnlySymmetricQuantized.mlpackage and
    ResNet50SymmetricPerChannel.mlpackage from ml-assets.apple.com, into --apple-dir
    ImageNetV2 matched-frequency (10,000 labelled images), into --data-dir

    python fp8_resnet50.py --apple-dir apple --data-dir data/imagenetv2-matched-frequency-format-val

Needs macOS 27+ and ``pip install ml_dtypes pillow``. The Neural Engine runs FP8 on M6 and later.
"""

import argparse
import json
import os
import shutil
from collections import Counter

import numpy as np
from PIL import Image

import coremltools as ct
import coremltools.optimize as cto
from coremltools.models.compute_plan import MLComputePlan

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
BATCH = 8  # Apple's models take a fixed [8, 3, 224, 224] input named "image"
UNITS = {
    "ane": ct.ComputeUnit.CPU_AND_NE,
    "gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu": ct.ComputeUnit.CPU_ONLY,
}
APPLE_MODELS = {
    "apple_fp16": "ResNet50.mlpackage",
    "apple_int8_w8": "ResNet50WeightOnlySymmetricQuantized.mlpackage",
    "apple_int8_w8a8_qat": "ResNet50SymmetricPerChannel.mlpackage",
}
# name -> (activation dtype or None, weight dtype or None, fp8 weight encoding)
OUR_MODELS = {
    "fp16_ios26": (None, None, None),
    "int8_w8": (None, "int8", None),
    "int8_w8a8": ("int8", "int8", None),
    "fp8_w8": (None, "fp8e4m3fn", "blockwise"),
    "fp8_w8a8": ("fp8e4m3fn", "fp8e4m3fn", "blockwise"),
    "fp8_w8_palette": (None, "fp8e4m3fn", "palette"),
    "fp8_w8a8_palette": ("fp8e4m3fn", "fp8e4m3fn", "palette"),
}


# --------------------------------------------------------------------------- data


def center_crop(path: str) -> np.ndarray:
    """Resize the short side to 256 and take the central 224x224 crop (torchvision eval transform)."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = 256 / min(w, h)
    image = image.resize((max(224, int(w * scale)), max(224, int(h * scale))), Image.BILINEAR)
    w, h = image.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return np.asarray(image.crop((left, top, left + 224, top + 224)), dtype=np.uint8)


def load_imagenetv2(data_dir: str, cache: str):
    """Center crops (uint8 NHWC) and labels. Class folders are named by ImageNet class index."""
    if os.path.exists(cache):
        data = np.load(cache)
        return data["images"], data["labels"]
    files = sorted(
        (int(label), os.path.join(data_dir, label, name))
        for label in os.listdir(data_dir)
        if label.isdigit()
        for name in sorted(os.listdir(os.path.join(data_dir, label)))
    )
    images = np.stack([center_crop(path) for _, path in files])
    labels = np.array([label for label, _ in files], np.int32)
    np.savez(cache, images=images, labels=labels)
    return images, labels


def to_input(crops: np.ndarray) -> np.ndarray:
    """uint8 NHWC crops -> normalized fp32 NCHW batch, padded to BATCH."""
    x = crops.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    if len(x) < BATCH:
        x = np.concatenate([x, np.zeros((BATCH - len(x),) + x.shape[1:], np.float32)])
    return x


# --------------------------------------------------------------------------- models


def build(name: str, fp16: ct.models.MLModel, calibration: list) -> ct.models.MLModel:
    activation_dtype, weight_dtype, encoding = OUR_MODELS[name]
    if activation_dtype is None and weight_dtype is None:
        # No compression, just the iOS 26 opset, to separate the opset from the quantization.
        return ct.models.utils._apply_graph_pass(fp16, [], spec_version=ct._SPECIFICATION_VERSION_IOS_26)
    model = fp16
    if activation_dtype is not None:
        # Calibrate activation ranges on sample images; inserts quantize/dequantize pairs.
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype=activation_dtype
            )
        )
        model = cto.coreml.linear_quantize_activations(model, config, calibration)
    weight_config = dict(mode="linear_symmetric", dtype=weight_dtype, granularity="per_channel")
    if encoding is not None:
        weight_config["fp8_encoding"] = encoding
    config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(**weight_config)
    )
    return cto.coreml.linear_quantize_weights(model, config)


def placement(compiled: str, units: ct.ComputeUnit) -> dict:
    plan = MLComputePlan.load_from_path(compiled, compute_units=units)
    counts = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is not None:
            device = type(usage.preferred_compute_device).__name__
            counts[device.replace("ML", "").replace("ComputeDevice", "")] += 1
    return dict(counts)


def weight_bytes(package: str) -> int:
    weights = os.path.join(package, "Data", "com.apple.CoreML", "weights")
    return sum(os.path.getsize(os.path.join(weights, f)) for f in os.listdir(weights))


def evaluate(model, images, labels, reference=None):
    """Top-1/top-5 accuracy, and top-1 agreement with the reference predictions."""
    top1 = top5 = agree = 0
    predictions = []
    for start in range(0, len(images), BATCH):
        crops = images[start : start + BATCH]
        logits = model.predict({"image": to_input(crops)})["classLabelProbs"][: len(crops)]
        top = np.argsort(-logits, axis=1)[:, :5]
        truth = labels[start : start + BATCH]
        top1 += int((top[:, 0] == truth).sum())
        top5 += int((top == truth[:, None]).any(axis=1).sum())
        predictions.append(top[:, 0])
    predictions = np.concatenate(predictions)
    if reference is not None:
        agree = float((predictions == reference).mean())
    return 100 * top1 / len(images), 100 * top5 / len(images), 100 * agree, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--apple-dir", required=True, help="folder with Apple's three ResNet50 .mlpackages")
    parser.add_argument("--data-dir", required=True, help="ImageNetV2 folder (class-index subfolders)")
    parser.add_argument("--out-dir", default="resnet50_fp8")
    parser.add_argument("--variants", default=",".join(list(APPLE_MODELS) + list(OUR_MODELS)))
    parser.add_argument("--units", default="ane,gpu", help="compute units to evaluate: ane,gpu,cpu")
    parser.add_argument("--num-calib", type=int, default=128, help="calibration images (held out)")
    parser.add_argument("--num-eval", type=int, default=0, help="evaluation images (0 = all others)")
    parser.add_argument("--force", action="store_true", help="rebuild models that already exist")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    images, labels = load_imagenetv2(args.data_dir, os.path.join(args.out_dir, "imagenetv2_crops.npz"))
    order = np.random.default_rng(0).permutation(len(images))
    calib = order[: args.num_calib]
    held_out = order[args.num_calib :]
    if args.num_eval:
        held_out = held_out[: args.num_eval]
    eval_images, eval_labels = images[held_out], labels[held_out]
    calibration = [
        {"image": to_input(images[calib[i : i + BATCH]])} for i in range(0, len(calib), BATCH)
    ]
    print(f"{len(images)} images: {len(calib)} for calibration, {len(held_out)} for evaluation")

    fp16_package = os.path.join(args.apple_dir, APPLE_MODELS["apple_fp16"])
    fp16 = ct.models.MLModel(fp16_package, skip_model_load=True)
    fp16_size = weight_bytes(fp16_package)

    reference = {}
    results = []
    for name in args.variants.split(","):
        if name in APPLE_MODELS:
            package = os.path.join(args.apple_dir, APPLE_MODELS[name])
        else:
            package = os.path.join(args.out_dir, f"ResNet50_{name}.mlpackage")
            if args.force or not os.path.exists(package):
                shutil.rmtree(package, ignore_errors=True)
                build(name, fp16, calibration).save(package)
        compiled = os.path.join(args.out_dir, os.path.basename(package).replace(".mlpackage", ".mlmodelc"))
        if args.force or not os.path.exists(compiled):
            shutil.rmtree(compiled, ignore_errors=True)
            ct.models.utils.compile_model(package, compiled)
        spec_version = ct.models.MLModel(package, skip_model_load=True).get_spec().specificationVersion
        row = dict(name=name, spec=spec_version, ratio=round(fp16_size / weight_bytes(package), 2))
        for unit in args.units.split(","):
            where = placement(compiled, UNITS[unit])
            row[unit] = dict(placement=where)
            if unit != "cpu" and set(where) == {"CPU"}:
                # Nothing runs on the requested unit (e.g. FP8 on the GPU): skip the slow CPU run.
                continue
            model = ct.models.CompiledMLModel(compiled, compute_units=UNITS[unit])
            top1, top5, agree, predictions = evaluate(model, eval_images, eval_labels, reference.get(unit))
            if name == "apple_fp16":
                reference[unit] = predictions
            row[unit].update(top1=round(top1, 2), top5=round(top5, 2), agree_fp16=round(agree, 2))
        results.append(row)
        print(json.dumps(row), flush=True)

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    units = args.units.split(",")
    header = f"{'variant':20} {'spec':>4} {'ratio':>5} " + " ".join(
        f"{u + ' top1':>9} {u + ' top5':>9} {u + ' agree':>9} {u + ' placement':24}" for u in units
    )
    print("\n" + header + "\n" + "-" * len(header))
    for row in results:
        cells = []
        for u in units:
            r = row[u]
            where = ",".join(f"{k}:{v}" for k, v in sorted(r["placement"].items()))
            if "top1" in r:
                cells.append(f"{r['top1']:9.2f} {r['top5']:9.2f} {r['agree_fp16']:9.2f} {where:24}")
            else:
                cells.append(f"{'-':>9} {'-':>9} {'-':>9} {where:24}")
        print(f"{row['name']:20} {row['spec']:4d} {row['ratio']:5.2f} " + " ".join(cells))


if __name__ == "__main__":
    main()
