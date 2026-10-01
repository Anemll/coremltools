#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import os
import re
import subprocess

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.frontend.milproto.load import load as milproto_load
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.defs.iOS26 import _IOS26_TARGET
from coremltools.converters.mil.mil.ops.tests.iOS26 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import macos_compatible_with_deployment_target

ml_dtypes = pytest.importorskip("ml_dtypes")

# FP8 dtype name -> (numpy dtype, largest finite value)
_FP8_DTYPES = {
    "fp8e4m3fn": (ml_dtypes.float8_e4m3fn, 448.0),
    "fp8e5m2": (ml_dtypes.float8_e5m2, 57344.0),
}


def _fp8_quantize(x: np.ndarray, scale: float, dtype_name: str) -> np.ndarray:
    """Reference FP8 quantize, matching Core ML on the CPU: fp32 divide, saturate, round to nearest even."""
    np_dtype, fp8_max = _FP8_DTYPES[dtype_name]
    return np.clip(x.astype(np.float32) / np.float32(scale), -fp8_max, fp8_max).astype(np_dtype)


def _nudge_off_fp8_ties(x: np.ndarray, scale: float, dtype_name: str) -> np.ndarray:
    """
    Move values whose ``x / scale`` falls exactly halfway between two FP8 values up by one ulp.
    Core ML devices round such ties differently (the CPU toward -inf, the Neural Engine away from 0).
    """
    np_dtype = _FP8_DTYPES[dtype_name][0]
    x = x.copy()
    q = x.astype(np.float32) / np.float32(scale)
    nearest = q.astype(np_dtype).astype(np.float32)
    mirror = 2 * q - nearest  # the other neighbor if q is a tie
    is_tie = (mirror != nearest) & (mirror.astype(np_dtype).astype(np.float32) == mirror)
    x[is_tie] = np.nextafter(x[is_tie], np.array(np.inf, x.dtype))
    return x


def _neural_engine_supports_fp8() -> bool:
    """The Neural Engine runs E4M3 from the M6 generation on. COREMLTOOLS_TEST_ANE_FP8=0/1 overrides."""
    override = os.environ.get("COREMLTOOLS_TEST_ANE_FP8")
    if override is not None:
        return override == "1"
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout
    except OSError:
        return False
    match = re.search(r"Apple M(\d+)", brand)
    return match is not None and int(match.group(1)) >= 6


class TestQuantizeFP8:
    @pytest.mark.parametrize("dtype_name", _FP8_DTYPES)
    def test_builder_eval(self, dtype_name):
        fp8_max = _FP8_DTYPES[dtype_name][1]
        # Out-of-range values saturate, and 2.125 (1.0625 / 0.5) is a tie that rounds to even.
        x = np.array([2 * fp8_max, -2 * fp8_max, 1.0625, 0.1, -3.3, 0.0], dtype=np.float32)

        @mb.program(input_specs=[], opset_version=_IOS26_TARGET)
        def prog():
            return mb.quantize(input=x, scale=np.float32(0.5), output_dtype=dtype_name)

        output = prog.functions["main"].find_ops(op_type="quantize")[0].outputs[0]
        assert output.dtype == types.string_to_builtin(dtype_name)
        expected = _fp8_quantize(x, 0.5, dtype_name)
        np.testing.assert_array_equal(output.val.view(np.uint8), expected.view(np.uint8))
        assert np.abs(output.val.astype(np.float32)).max() == fp8_max
        assert not np.any(np.isnan(output.val.astype(np.float32)))

    def test_zero_point_is_rejected(self):
        with pytest.raises(ValueError, match="zero_point is not supported for FP8"):

            @mb.program(input_specs=[mb.TensorSpec(shape=(4,))], opset_version=_IOS26_TARGET)
            def prog(x):
                return mb.quantize(
                    input=x, scale=np.float32(1), zero_point=np.int8(0), output_dtype="fp8e4m3fn"
                )

    def test_fp8_needs_ios26(self):
        with pytest.raises(ValueError, match="unrecognized output dtype"):

            @mb.program(input_specs=[mb.TensorSpec(shape=(4,))], opset_version=ct.target.iOS18)
            def prog(x):
                return mb.quantize(input=x, scale=np.float32(1), output_dtype="fp8e4m3fn")

    @pytest.mark.parametrize(
        "compute_unit, backend, dtype_name",
        itertools.product(compute_units, backends, _FP8_DTYPES),
    )
    def test_builder_to_backend_quantize_dequantize(self, compute_unit, backend, dtype_name):
        fp8_max = _FP8_DTYPES[dtype_name][1]
        scale = 0.25  # a power of two, so fp16 and fp32 models see the same scale
        x = np.random.default_rng(0).standard_normal((2, 3, 4, 8)).astype(np.float32) * 8
        # Twice the FP8 range saturates (and stays within fp16), and 1.0625 / 0.25 is exact.
        x[0, 0, 0, :3] = [2 * fp8_max * scale, -2 * fp8_max * scale, 1.0625]
        x_model = x.astype(np.float16 if backend.precision == "fp16" else np.float32)
        x_model = _nudge_off_fp8_ties(x_model, scale, dtype_name)
        x = x_model.astype(np.float32)
        expected = _fp8_quantize(x_model, scale, dtype_name).astype(np.float32) * scale

        def build(x):
            q = mb.quantize(input=x, scale=np.float32(scale), output_dtype=dtype_name)
            return mb.dequantize(input=q, scale=np.float32(scale))

        run_compare_builder(
            build,
            {"x": mb.placeholder(shape=x.shape)},
            input_values={"x": x},
            expected_output_types=x.shape + (types.fp32,),
            expected_outputs=expected,
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-6,
            rtol=1e-3,
        )


class TestOpVersions:
    def test_versions_do_not_share_type_domains(self):
        """Building one op version must not change which dtypes another version accepts."""
        fp8_data = np.zeros((2, 4), ml_dtypes.float8_e4m3fn)
        for opset_version in (ct.target.iOS18, _IOS26_TARGET, ct.target.iOS18):

            @mb.program(input_specs=[mb.TensorSpec(shape=(4,))], opset_version=opset_version)
            def prog(x):
                q = mb.quantize(input=x, scale=np.float32(1), output_dtype="int8")
                return mb.dequantize(input=q, scale=np.float32(1))

            if opset_version == _IOS26_TARGET:

                @mb.program(input_specs=[mb.TensorSpec(shape=(4,))], opset_version=opset_version)
                def prog(x):
                    q = mb.quantize(input=x, scale=np.float32(1), output_dtype="fp8e4m3fn")
                    w = mb.constexpr_blockwise_shift_scale(
                        data=fp8_data, scale=np.ones((2, 1), np.float32)
                    )
                    return mb.dequantize(input=q, scale=np.float32(1)), w

            else:
                with pytest.raises(ValueError, match="type domain"):

                    @mb.program(input_specs=[], opset_version=opset_version)
                    def prog():
                        return mb.constexpr_blockwise_shift_scale(
                            data=fp8_data, scale=np.ones((2, 1), np.float32)
                        )


class TestDequantizeFP8:
    @pytest.mark.parametrize("dtype_name", _FP8_DTYPES)
    def test_builder_eval(self, dtype_name):
        np_dtype = _FP8_DTYPES[dtype_name][0]
        data = np.array([-2.0, 0.5, 3.0, 0.0], np.float32).astype(np_dtype)

        @mb.program(input_specs=[], opset_version=_IOS26_TARGET)
        def prog():
            return mb.dequantize(input=data, scale=np.float32(0.25))

        op = prog.functions["main"].find_ops(op_type="dequantize")[0]
        assert op.outputs[0].dtype == types.fp32
        np.testing.assert_array_equal(op.materialized_val_inference(), [-0.5, 0.125, 0.75, 0.0])

    def test_zero_point_is_rejected(self):
        with pytest.raises(ValueError, match="zero_point is not supported for FP8"):

            @mb.program(input_specs=[], opset_version=_IOS26_TARGET)
            def prog():
                data = np.zeros(4, ml_dtypes.float8_e4m3fn)
                zero_point = np.zeros((), ml_dtypes.float8_e4m3fn)
                return mb.dequantize(input=data, scale=np.float32(1), zero_point=zero_point)


class TestConstexprBlockwiseShiftScaleFP8:
    @pytest.mark.parametrize("dtype_name", _FP8_DTYPES)
    def test_builder_eval(self, dtype_name):
        np_dtype = _FP8_DTYPES[dtype_name][0]
        data = np.array([0.5, -1.0, 2.0, 240.0, 3.5, 0.0, -0.25, 16.0], np.float32)
        data = data.astype(np_dtype).reshape((1, 2, 4))
        scale = np.array([0.25, 2.0], np.float16).reshape((1, 2, 1))

        @mb.program(input_specs=[], opset_version=_IOS26_TARGET)
        def prog():
            return mb.constexpr_blockwise_shift_scale(data=data, scale=scale)

        op = prog.functions["main"].find_ops(op_type="constexpr_blockwise_shift_scale")[0]
        assert op.outputs[0].dtype == types.fp16
        np.testing.assert_array_equal(
            op.materialized_val_inference(), (data.astype(np.float32) * scale).astype(np.float16)
        )

    def test_offset_is_rejected(self):
        with pytest.raises(ValueError, match="not supported when 'data' is FP8"):

            @mb.program(input_specs=[], opset_version=_IOS26_TARGET)
            def prog():
                return mb.constexpr_blockwise_shift_scale(
                    data=np.zeros((2, 4), ml_dtypes.float8_e4m3fn),
                    scale=np.ones((2, 1), np.float16),
                    offset=np.zeros((2, 1), np.float16),
                )

    def test_fp8_needs_ios26(self):
        with pytest.raises(ValueError, match="data"):

            @mb.program(input_specs=[], opset_version=ct.target.iOS18)
            def prog():
                return mb.constexpr_blockwise_shift_scale(
                    data=np.zeros((2, 4), ml_dtypes.float8_e4m3fn), scale=np.ones((2, 1), np.float16)
                )

    @pytest.mark.skipif(
        not macos_compatible_with_deployment_target(ct.target.iOS26),
        reason="Needs the macOS 26+ Core ML runtime.",
    )
    def test_save_and_load_round_trip(self, tmp_path):
        data = np.linspace(-240, 240, 64, dtype=np.float32).astype(ml_dtypes.float8_e4m3fn)
        data = data.reshape((16, 4))

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))], opset_version=_IOS26_TARGET)
        def prog(x):
            w = mb.constexpr_blockwise_shift_scale(data=data, scale=np.full((16, 1), 0.5, np.float32))
            return mb.linear(x=x, weight=w)

        mlmodel = ct.convert(
            prog, minimum_deployment_target=ct.target.iOS26, skip_model_load=True
        )
        package_path = str(tmp_path / "fp8.mlpackage")
        mlmodel.save(package_path)
        spec = ct.utils.load_spec(package_path)
        weights_dir = ct.models.utils._try_get_weights_dir_path(package_path)
        loaded = milproto_load(spec, spec.specificationVersion, weights_dir)
        op = loaded.functions["main"].find_ops(op_type="constexpr_blockwise_shift_scale")[0]
        assert op.data.dtype == types.fp8e4m3fn
        np.testing.assert_array_equal(op.data.val.view(np.uint8), data.view(np.uint8))

    @pytest.mark.parametrize(
        "compute_unit, backend, dtype_name",
        itertools.product(compute_units, backends, _FP8_DTYPES),
    )
    def test_builder_to_backend(self, compute_unit, backend, dtype_name):
        """Runs through ct.convert, which compiles around the Core ML compiler's FP8 crash."""
        np_dtype, fp8_max = _FP8_DTYPES[dtype_name]
        rng = np.random.default_rng(1)
        data = (rng.uniform(-1, 1, (8, 16)) * min(fp8_max, 240)).astype(np_dtype)
        scale = rng.uniform(0.01, 0.1, (8, 2)).astype(np.float32)
        x = rng.standard_normal((8, 16)).astype(np.float32)
        weight = data.astype(np.float32) * np.repeat(scale, 8, axis=1)

        def build(x):
            return mb.add(x=x, y=mb.constexpr_blockwise_shift_scale(data=data, scale=scale))

        run_compare_builder(
            build,
            {"x": mb.placeholder(shape=x.shape)},
            input_values={"x": x},
            expected_output_types=x.shape + (types.fp32,),
            expected_outputs=x + weight,
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.skipif(
    not macos_compatible_with_deployment_target(ct.target.iOS26) or not _neural_engine_supports_fp8(),
    reason="Needs a Neural Engine that runs FP8 (E4M3), M6 or later.",
)
class TestFP8OnNeuralEngine:
    def test_w8a8_conv_chain(self):
        """FP8 weights and activations: every compute op is placed on the Neural Engine."""
        from coremltools.models.compute_plan import MLComputePlan

        channels, num_layers = 512, 4
        rng = np.random.default_rng(0)
        layers = []
        for _ in range(num_layers):
            w = rng.standard_normal((channels, channels, 1, 1)).astype(np.float32) / np.sqrt(channels)
            # Codes up to 240: the Neural Engine reads FP8 weights like IEEE E4M3.
            scale = (np.abs(w).reshape(channels, -1).max(1) / 240).astype(np.float16)
            data = (w / scale.astype(np.float32).reshape(-1, 1, 1, 1)).astype(ml_dtypes.float8_e4m3fn)
            layers.append((data, scale.reshape(-1, 1, 1, 1)))

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, channels, 32, 32), dtype=types.fp16)],
            opset_version=_IOS26_TARGET,
        )
        def prog(x):
            for i, (data, scale) in enumerate(layers):
                w = mb.constexpr_blockwise_shift_scale(data=data, scale=scale)
                x = mb.conv(x=x, weight=w)
                if i < num_layers - 1:
                    q = mb.quantize(input=x, scale=np.float16(0.05), output_dtype="fp8e4m3fn")
                    x = mb.dequantize(input=q, scale=np.float16(0.05))
            return x

        mlmodel = ct.convert(
            prog, minimum_deployment_target=ct.target.iOS26, compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        plan = MLComputePlan.load_from_path(
            mlmodel.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        for op in plan.model_structure.program.functions["main"].block.operations:
            if op.operator_name.split(".")[-1] in ("conv", "quantize", "dequantize"):
                device = plan.get_compute_device_usage_for_mlprogram_operation(op).preferred_compute_device
                assert type(device).__name__ == "MLNeuralEngineComputeDevice", op.operator_name

        x = rng.standard_normal((1, channels, 32, 32)).astype(np.float16)
        y_ane = list(mlmodel.predict({"x": x}).values())[0]
        cpu_model = ct.models.CompiledMLModel(
            mlmodel.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_ONLY
        )
        y_cpu = list(cpu_model.predict({"x": x}).values())[0]
        assert np.abs(y_ane - y_cpu).max() / np.abs(y_cpu).max() < 0.05
