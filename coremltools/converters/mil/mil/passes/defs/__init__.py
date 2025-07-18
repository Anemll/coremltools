#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Missing module - commented out for development
# from .optimize_deployment_target import (
#     divide_to_multiply,
#     lower_const_op_to_const_tensor,
#     update_output_dtypes,
# )
from .optimize_elementwise_binary import select_optimization
from .optimize_normalization import fuse_layernorm_or_instancenorm
from .ane_rms_norm_to_layer_norm import lower_ane_rms_norm_to_layer_norm
from .fuse_rms_norm import fuse_rms_norm
from .optimize_quantization import (
    canonicalize_quantized_lut_pattern,
    dequantize_quantize_pair_elimination,
)
