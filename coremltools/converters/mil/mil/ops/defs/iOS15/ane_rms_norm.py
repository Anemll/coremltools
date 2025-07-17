#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import numpy as np

from coremltools.converters.mil.mil import (
    DefaultInputs,
    InputSpec,
    Operation,
    TensorInputType,
    precondition,
    types,
)
from coremltools.converters.mil.mil.operation import VALUE
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.types.symbolic import any_symbolic


@register_op
class ane_rms_norm(Operation):
    """
    Apply Root Mean Square Normalization to the n-dimensional input tensor.
    This operation can be optimized for Apple Neural Engine (ANE) by lowering
    it to a sequence of other operations.

    .. math::
       out = x / sqrt(mean(x**2) + epsilon) * gamma

    Parameters
    ----------
    x: tensor<*?, T> (Required)
        * Input tensor.
    axes: const<[K], i32> (Optional)
        * Dimensions to perform normalization over.
        * Default is ``[-1]``.
    gamma: const tensor<*?, T> (Optional)
        * The learnable gain (weight). The shape must be ``x.shape[axes]``.
        * Default is all ones.
    epsilon: const T (Optional)
        * Small constant to avoid division by 0.
        * Default is 1e-5.

    Returns
    -------
    tensor<*?, T>:
     * Tensor with same shape and type as the input tensor x.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axes=TensorInputType(const=True, optional=True, type_domain=types.int32),
        gamma=TensorInputType(const=True, optional=True, type_domain="T"),
        epsilon=TensorInputType(const=True, optional=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        return DefaultInputs(
            axes=[-1],
            gamma=None,
            epsilon=1e-5,
        )

    @staticmethod
    def _is_compatible_shape(shapea, shapeb):
        if not len(shapea) == len(shapeb):
            return False
        for a, b in zip(shapea, shapeb):
            if any_symbolic([a, b]):
                continue
            if a != b:
                return False
        return True

    def type_inference(self):
        rank = self.x.rank

        # check valid axes
        positive_axes = [axis + rank if axis < 0 else axis for axis in self.axes.val]
        if not all(axis >= 0 and axis < rank for axis in positive_axes):
            raise ValueError("axes must in the range of [-x.rank, x.rank-1].")

        # check shape of gamma
        if self.gamma is not None:
            normalized_shape = [self.x.shape[i] for i in range(rank) if i in positive_axes]
            if not ane_rms_norm._is_compatible_shape(list(self.gamma.shape), normalized_shape):
                raise ValueError(
                    "Expect shape {} for gamma, but get shape {} instead".format(
                        normalized_shape, list(self.gamma.shape)
                    )
                )

        x_shape = self.x.shape
        return types.tensor(self.x.dtype, tuple(x_shape))

    @precondition(allow=VALUE)
    def value_inference(self):
        def np_rms_norm(x, axes, gamma, epsilon):
            mean_square = np.mean(np.square(x), axis=tuple(axes), keepdims=True)
            normalized_x = x * np.rsqrt(mean_square + epsilon)
            if gamma is not None:
                return normalized_x * gamma
            return normalized_x

        _axes = self.axes.val
        _gamma = None if self.gamma is None else self.gamma.val
        return np_rms_norm(self.x.val, _axes, _gamma, self.epsilon.val) 