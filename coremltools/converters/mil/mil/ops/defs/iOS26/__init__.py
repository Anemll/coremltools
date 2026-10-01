#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target

# Ensure op registrations recognize the new opset.
_IOS26_TARGET = target.iOS26

from .compression import constexpr_blockwise_shift_scale
from .quantization_ops import dequantize, quantize
