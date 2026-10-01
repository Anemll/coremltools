#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
import subprocess

import numpy as np
import pytest

import coremltools as ct
import coremltools.optimize as cto
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.testing_utils import macos_compatible_with_deployment_target
from coremltools.models import _fp8_compile

ml_dtypes = pytest.importorskip("ml_dtypes")

_CAN_RUN_IOS26 = ct.utils._is_macos() and macos_compatible_with_deployment_target(ct.target.iOS26)
_E4M3_VALUES = set(
    np.nan_to_num(np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32))
)


def _weights():
    rng = np.random.default_rng(0)
    return {
        "conv_w": (rng.standard_normal((128, 64, 1, 1)) * 0.05).astype(np.float32),
        "conv_b": (rng.standard_normal(128) * 0.1).astype(np.float32),
        "linear_w": (rng.standard_normal((32, 128)) * 0.1).astype(np.float32),
        "linear_b": (rng.standard_normal(32) * 0.1).astype(np.float32),
    }


def _conv_linear_model(opset_version=ct.target.iOS26, skip_model_load=True):
    w = _weights()

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 64, 4, 4))], opset_version=opset_version)
    def prog(x):
        x = mb.conv(x=x, weight=w["conv_w"], bias=w["conv_b"], name="conv")
        x = mb.relu(x=x)
        x = mb.transpose(x=mb.reshape(x=x, shape=(128, 16)), perm=[1, 0])
        return mb.linear(x=x, weight=w["linear_w"], bias=w["linear_b"], name="linear")

    return ct.convert(
        prog,
        minimum_deployment_target=opset_version,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=skip_model_load,
    )


def _predict(mlmodel, inputs):
    """Predict on the CPU through a saved copy (``mlmodel`` may have been built without loading)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.mlpackage")
        mlmodel.save(path)
        loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
        return list(loaded.predict(inputs).values())[0]


def _fp8_config(**kwargs):
    kwargs.setdefault("dtype", "fp8e4m3fn")
    return cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", **kwargs)
    )


def _ops(mlmodel, op_type):
    return mlmodel._mil_program.functions["main"].find_ops(op_type=op_type)


class TestFP8Config:
    @pytest.mark.parametrize(
        "dtype, expected",
        [
            ("fp8e4m3fn", types.fp8e4m3fn),
            (types.fp8e4m3fn, types.fp8e4m3fn),
            (ml_dtypes.float8_e4m3fn, types.fp8e4m3fn),
            ("fp8e5m2", types.fp8e5m2),
            (ml_dtypes.float8_e5m2, types.fp8e5m2),
        ],
    )
    def test_dtype(self, dtype, expected):
        assert cto.coreml.OpLinearQuantizerConfig(dtype=dtype).dtype == expected

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"mode": "linear"}, "symmetric"),
            ({"fp8_max": 500}, "fp8_max"),
            ({"dtype": "fp8e5m2", "fp8_encoding": "palette"}, "palette"),
            ({"granularity": "per_block", "fp8_encoding": "palette"}, "palette"),
        ],
    )
    def test_invalid(self, kwargs, message):
        kwargs.setdefault("dtype", "fp8e4m3fn")
        with pytest.raises(ValueError, match=message):
            cto.coreml.OpLinearQuantizerConfig(**kwargs)


class TestFP8Weights:
    @pytest.mark.parametrize("granularity", ["per_tensor", "per_channel", "per_block"])
    def test_blockwise(self, granularity):
        mlmodel = cto.coreml.linear_quantize_weights(
            _conv_linear_model(), _fp8_config(granularity=granularity, block_size=32)
        )
        ops = _ops(mlmodel, "constexpr_blockwise_shift_scale")
        assert len(ops) == 2
        originals = [_weights()["conv_w"], _weights()["linear_w"]]
        for op, original in zip(ops, originals):
            assert op.data.dtype == types.fp8e4m3fn
            codes = np.abs(op.data.val.astype(np.float32))
            # 240 is the default range, which the Neural Engine reads exactly.
            assert codes.max() == 240
            decompressed = op.materialized_val_inference().astype(np.float32)
            assert np.abs(decompressed - original).max() <= np.abs(original).max() * 2**-4

    def test_fp8_max(self):
        mlmodel = cto.coreml.linear_quantize_weights(_conv_linear_model(), _fp8_config(fp8_max=448.0))
        for op in _ops(mlmodel, "constexpr_blockwise_shift_scale"):
            assert np.abs(op.data.val.astype(np.float32)).max() == 448

    def test_upgrades_to_ios26(self):
        # FP8 ops exist from iOS26, so an older model (e.g. a downloaded one) is upgraded to it.
        mlmodel = cto.coreml.linear_quantize_weights(
            _conv_linear_model(opset_version=ct.target.iOS18), _fp8_config()
        )
        assert mlmodel.get_spec().specificationVersion == ct._SPECIFICATION_VERSION_IOS_26
        assert len(_ops(mlmodel, "constexpr_blockwise_shift_scale")) == 2

    def test_palette(self):
        mlmodel = cto.coreml.linear_quantize_weights(
            _conv_linear_model(), _fp8_config(fp8_encoding="palette")
        )
        assert len(_ops(mlmodel, "constexpr_blockwise_shift_scale")) == 0
        luts = _ops(mlmodel, "constexpr_lut_to_dense")
        assert len(luts) == 2
        for op in luts:
            # The table holds exactly the E4M3 values, and the used codes stay within 240.
            assert set(op.lut.val.astype(np.float32).ravel()) <= _E4M3_VALUES
            used = op.materialized_val_inference().astype(np.float32)
            assert np.abs(used).max() == 240
        # The scale moved around the consumers; the output name is unchanged.
        assert len(_ops(mlmodel, "mul")) == 4
        assert [o.name for o in mlmodel.get_spec().description.output] == ["linear"]
        assert not _fp8_compile.needs_fp8_compile_workaround(mlmodel.get_spec())

    def test_palette_wide_channel_scales(self):
        # Folded batch norms give channel scales that differ by orders of magnitude (up to 1e7 in
        # ResNet50). The prescaled activations must not underflow in fp16.
        rng = np.random.default_rng(3)
        channel_amax = np.logspace(-6, 0, 32).astype(np.float32)
        weight = rng.uniform(-1, 1, (32, 16, 3, 3)).astype(np.float32) * channel_amax[:, None, None, None]
        bias = rng.standard_normal(32).astype(np.float32)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 8, 8))], opset_version=ct.target.iOS26)
        def prog(x):
            return mb.conv(x=x, weight=weight, bias=bias, pad_type="same", name="conv")

        base = ct.convert(prog, minimum_deployment_target=ct.target.iOS26, skip_model_load=True)
        palette = cto.coreml.linear_quantize_weights(base, _fp8_config(fp8_encoding="palette"))
        prescale = [op for op in _ops(palette, "mul") if op.name.endswith("_fp8_prescale")]
        assert len(prescale) == 1
        # The prescale is the largest channel scale rounded down to a power of two.
        alpha = float(prescale[0].y.val)
        assert alpha == 2.0 ** np.floor(np.log2(channel_amax.max() / 240))
        if _CAN_RUN_IOS26:
            blockwise = cto.coreml.linear_quantize_weights(base, _fp8_config())
            x = {"x": rng.standard_normal((1, 16, 8, 8)).astype(np.float32)}
            expected = _predict(blockwise, x)
            np.testing.assert_allclose(_predict(palette, x), expected, rtol=1e-2, atol=1e-2)

    def test_palette_skips_other_consumers(self):
        w = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)

        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 64))], opset_version=ct.target.iOS26)
        def prog(x):
            return mb.matmul(x=x, y=w)

        mlmodel = ct.convert(prog, minimum_deployment_target=ct.target.iOS26, skip_model_load=True)
        mlmodel = cto.coreml.linear_quantize_weights(mlmodel, _fp8_config(fp8_encoding="palette"))
        assert len(_ops(mlmodel, "constexpr_lut_to_dense")) == 0

    @pytest.mark.skipif(not _CAN_RUN_IOS26, reason="Needs the macOS 26+ Core ML runtime.")
    def test_encodings_predict_the_same(self, tmp_path):
        x = np.random.default_rng(1).standard_normal((1, 64, 4, 4)).astype(np.float32)
        base = _conv_linear_model()
        outputs = {}
        for encoding in ("blockwise", "palette"):
            mlmodel = cto.coreml.linear_quantize_weights(base, _fp8_config(fp8_encoding=encoding))
            path = str(tmp_path / f"{encoding}.mlpackage")
            mlmodel.save(path)
            if encoding == "palette":
                # The stock compiler (as used by Xcode) accepts the palette encoding.
                out_dir = str(tmp_path / "xcrun")
                os.makedirs(out_dir)
                subprocess.run(["xcrun", "coremlcompiler", "compile", path, out_dir], check=True)
            loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
            outputs[encoding] = loaded.predict({"x": x})["linear"]
        np.testing.assert_allclose(outputs["palette"], outputs["blockwise"], rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _CAN_RUN_IOS26, reason="Needs the macOS 26+ Core ML runtime.")
class TestFP8ActivationsAndCompile:
    def test_activations(self):
        base = _conv_linear_model(skip_model_load=False)
        rng = np.random.default_rng(2)
        sample_data = [{"x": rng.standard_normal((1, 64, 4, 4)).astype(np.float32)} for _ in range(4)]
        mlmodel = cto.coreml.linear_quantize_activations(base, _fp8_config(), sample_data)
        quantize_ops = _ops(mlmodel, "quantize")
        assert len(quantize_ops) > 0
        for op in quantize_ops:
            assert op.outputs[0].dtype == types.fp8e4m3fn
            assert op.zero_point is None

        x = sample_data[0]["x"]
        y = mlmodel.predict({"x": x})["linear"]
        y_ref = base.predict({"x": x})["linear"]
        assert np.abs(y - y_ref).max() <= 0.1 * np.abs(y_ref).max()

    def test_compile_model_works_around_compiler_crash(self, tmp_path):
        mlmodel = cto.coreml.linear_quantize_weights(_conv_linear_model(), _fp8_config())
        assert _fp8_compile.needs_fp8_compile_workaround(mlmodel.get_spec())
        path = str(tmp_path / "blockwise.mlpackage")
        mlmodel.save(path)

        compiled = ct.models.utils.compile_model(path, str(tmp_path / "blockwise.mlmodelc"))
        with open(os.path.join(compiled, "model.mil")) as f:
            assert "constexpr_blockwise_shift_scale(data = tensor<fp8e4m3fn," in f.read()
        x = np.random.default_rng(3).standard_normal((1, 64, 4, 4)).astype(np.float32)
        y = ct.models.CompiledMLModel(compiled, compute_units=ct.ComputeUnit.CPU_ONLY).predict({"x": x})
        assert np.all(np.isfinite(y["linear"]))
