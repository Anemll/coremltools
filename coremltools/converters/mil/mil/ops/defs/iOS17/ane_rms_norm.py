#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.ops.defs.iOS15.ane_rms_norm import (
    ane_rms_norm as _ane_rms_norm_iOS15,
)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class ane_rms_norm(_ane_rms_norm_iOS15):
    pass 