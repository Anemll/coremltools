#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS17.quantization_ops import (
    dequantize as _dequantize_iOS17,
)
from coremltools.converters.mil.mil.ops.defs.iOS17.quantization_ops import (
    quantize as _quantize_iOS17,
)
from coremltools.converters.mil.mil.ops.defs.iOS26 import _IOS26_TARGET


@register_op(opset_version=_IOS26_TARGET)
class quantize(_quantize_iOS17):
    """
    Performs affine/linear quantization on an input tensor.

    The only difference between this version and the iOS 17
    :py:class:`~.iOS17.quantization_ops.quantize` is that ``output_dtype`` can also be
    ``"fp8e4m3fn"`` or ``"fp8e5m2"``. For an FP8 ``output_dtype``::

        quantized_data = round_to_nearest(clip(input / scale, -FP8_MAX, FP8_MAX))

    where ``FP8_MAX`` is 448 for ``fp8e4m3fn`` and 57344 for ``fp8e5m2``. Out-of-range values
    saturate rather than becoming NaN, and ``zero_point`` must not be set.

    Exact ties are rounded to even when the value is computed at conversion time. Core ML
    devices can round exact ties differently (on macOS 27 the CPU rounds them toward -inf and the
    Neural Engine away from zero).

    Attributes
    ----------
    SrcT: fp16, fp32
    DstT: uint8, int8, fp8e4m3fn, fp8e5m2
    """

    # Operation.__init__ resolves type domains onto the input types, so each version needs its own spec.
    input_spec = InputSpec(
        input=TensorInputType(type_domain="SrcT"),
        zero_point=TensorInputType(const=True, optional=True, type_domain="DstT"),
        scale=TensorInputType(const=True, type_domain="SrcT"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        output_dtype=TensorInputType(const=True, type_domain=types.str),
    )

    type_domains = {
        "SrcT": (types.fp16, types.fp32),
        "DstT": (types.uint8, types.int8, types.fp8e4m3fn, types.fp8e5m2),
    }

    def type_inference(self):
        out_dtype = types.string_to_builtin(self.output_dtype.val)
        if types.is_fp8(out_dtype) and self.zero_point is not None:
            raise ValueError(
                f'"quantize" op: zero_point is not supported for FP8 output dtype '
                f'"{self.output_dtype.val}".'
            )
        return super().type_inference()


@register_op(opset_version=_IOS26_TARGET)
class dequantize(_dequantize_iOS17):
    """
    Performs dequantization on an input tensor with affine/linear quantization.

    The only difference between this version and the iOS 17
    :py:class:`~.iOS17.quantization_ops.dequantize` is that ``input`` can also be
    ``fp8e4m3fn`` or ``fp8e5m2``, in which case ``zero_point`` must not be set.

    Attributes
    ----------
    SrcT: uint8, int8, fp8e4m3fn, fp8e5m2
    DstT: fp16, fp32
    """

    input_spec = InputSpec(
        input=TensorInputType(type_domain="SrcT"),
        zero_point=TensorInputType(const=True, optional=True, type_domain="SrcT"),
        scale=TensorInputType(const=True, type_domain="DstT"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "DstT": (types.fp16, types.fp32),
        "SrcT": (types.uint8, types.int8, types.fp8e4m3fn, types.fp8e5m2),
    }

    def type_inference(self):
        if types.is_fp8(self.input.dtype) and self.zero_point is not None:
            raise ValueError('"dequantize" op: zero_point is not supported for FP8 input.')
        return super().type_inference()
