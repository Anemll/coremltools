#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS18.compression import (
    constexpr_blockwise_shift_scale as _constexpr_blockwise_shift_scale_iOS18,
)
from coremltools.converters.mil.mil.ops.defs.iOS26 import _IOS26_TARGET


@register_op(opset_version=_IOS26_TARGET)
class constexpr_blockwise_shift_scale(_constexpr_blockwise_shift_scale_iOS18):
    """
    A compression op that returns ``(data - offset) * scale``, with ``data`` quantized
    blockwise.

    The only difference between this version and the iOS 18
    :py:class:`~.iOS18.compression.constexpr_blockwise_shift_scale` is that ``data`` can also
    be ``fp8e4m3fn`` or ``fp8e5m2``. FP8 data is symmetric, so ``offset`` must not be set.

    Note: the Core ML compiler in iOS 26 / macOS 27 crashes on FP8 ``data`` while collecting
    compile analytics. ``coremltools.models.utils.compile_model`` works around it; see
    ``coremltools.models._fp8_compile``.

    Attributes
    ----------
    SrcT: int4, uint4, int8, uint8, fp8e4m3fn, fp8e5m2, fp16, fp32
    DstT: fp16, fp32
    OffsetT: int4, uint4, int8, uint8, fp16, fp32
    """

    # Operation.__init__ resolves type domains onto the input types, so each version needs its own spec.
    input_spec = InputSpec(
        data=TensorInputType(const=True, type_domain="SrcT"),
        scale=TensorInputType(const=True, type_domain="DstT"),
        offset=TensorInputType(const=True, optional=True, type_domain="OffsetT"),
    )

    type_domains = {
        "SrcT": (
            types.int4,
            types.uint4,
            types.int8,
            types.uint8,
            types.fp8e4m3fn,
            types.fp8e5m2,
            types.fp16,
            types.fp32,
        ),
        "DstT": (types.fp16, types.fp32),
        "OffsetT": (types.int4, types.uint4, types.int8, types.uint8, types.fp16, types.fp32),
    }

    def _validate_inputs(self):
        if types.is_fp8(self.data.dtype) and self.offset is not None:
            raise ValueError(
                "Invalid parameter 'offset'; it is not supported when 'data' is FP8."
            )
        super()._validate_inputs()
