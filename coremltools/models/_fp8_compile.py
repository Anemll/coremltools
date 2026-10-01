#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Compile ML programs whose constexpr ops take FP8 data.

The Core ML compiler in iOS 26 / macOS 27 accepts FP8 ``data`` in
``constexpr_blockwise_shift_scale`` but then crashes while collecting compile analytics
(``-[__NSSetM addObject:]: object cannot be nil``): its dtype-name table has no FP8 entry.
Both ``coremlcompiler`` and ``MLModel.compileModel`` are affected. The runtime loads and runs
these weights correctly on the CPU and on the Neural Engine.

The workaround compiles a copy of the package in which those FP8 tensors are labeled int8
(both are one byte per element), then restores the FP8 dtype in the compiled ``model.mil`` and
in the blob metadata of the compiled weight file.
"""

import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Iterator, List, NamedTuple, Optional, Tuple

from coremltools import proto

from .utils import _ModelPackage, _try_get_weights_dir_path
from .utils import load_spec as _load_spec

# MIL proto dtype -> (dtype name in the compiled model.mil, MILBlob BlobDataType)
_FP8_DTYPES = {
    proto.MIL_pb2.FLOAT8E4M3FN: ("fp8e4m3fn", 16),
    proto.MIL_pb2.FLOAT8E5M2: ("fp8e5m2", 17),
}
_BLOB_DTYPE_INT8 = 4
_BLOB_METADATA_SENTINEL = 0xDEADBEEF

# Constexpr inputs for which int8 is a valid stand-in for FP8 while compiling.
_RELABEL_INPUTS = {
    "constexpr_blockwise_shift_scale": "data",
    "constexpr_sparse_blockwise_shift_scale": "nonzero_data",
}

_FUNC_RE = re.compile(r"^\s*func\s+([^<\s(]+)")
_NAME_ATTR_RE = re.compile(r'\[name = string\("([^"]+)"\)')
_BLOBFILE = r'BLOBFILE\(path = string\("(?P<path>[^"]+)"\), offset = uint64\((?P<offset>\d+)\)\)'


class _Fp8Tensor(NamedTuple):
    function: str
    op_name: str  # the constexpr op, or the const op when the data is a named const
    input_name: Optional[str]  # None when op_name is a const op
    mil_dtype: str
    blob_dtype: int


def _op_name(op: proto.MIL_pb2.Operation) -> str:
    return op.attributes["name"].immediateValue.tensor.strings.values[0]


def _fp8_constexpr_inputs(spec: proto.Model_pb2.Model) -> Iterator[Tuple[_Fp8Tensor, list]]:
    """
    Yield ``(tensor, messages)`` for every FP8 tensor that feeds a constexpr op. ``messages`` are the
    proto messages whose tensor dtype must be relabeled (the value, plus the const output if any).
    """
    if spec.WhichOneof("Type") != "mlProgram":
        return
    for fn_name, fn in spec.mlProgram.functions.items():
        for block in fn.block_specializations.values():
            consts = {op.outputs[0].name: op for op in block.operations if op.type == "const"}
            for op in block.operations:
                if not op.type.startswith("constexpr_"):
                    continue
                for input_name, arg in op.inputs.items():
                    for binding in arg.arguments:
                        if binding.HasField("value"):
                            messages = [binding.value]
                            owner, owner_input = _op_name(op), input_name
                        elif binding.name in consts:
                            const_op = consts[binding.name]
                            messages = [const_op.attributes["val"], const_op.outputs[0]]
                            owner, owner_input = _op_name(const_op), None
                        else:
                            continue
                        dtype = messages[0].type.tensorType.dataType
                        if dtype not in _FP8_DTYPES:
                            continue
                        if _RELABEL_INPUTS.get(op.type) != input_name:
                            raise NotImplementedError(
                                f"Cannot compile FP8 input '{input_name}' of '{op.type}': the Core ML "
                                "compiler crashes on FP8 constexpr inputs, and the workaround only "
                                "supports " + ", ".join(f"{k}.{v}" for k, v in _RELABEL_INPUTS.items())
                            )
                        if not messages[0].HasField("blobFileValue"):
                            raise NotImplementedError(
                                f"FP8 input '{input_name}' of '{op.type}' must be stored in the weight "
                                "file to be compiled."
                            )
                        mil_dtype, blob_dtype = _FP8_DTYPES[dtype]
                        yield _Fp8Tensor(fn_name, owner, owner_input, mil_dtype, blob_dtype), messages


def needs_fp8_compile_workaround(spec: proto.Model_pb2.Model) -> bool:
    """Return True if compiling ``spec`` would hit the FP8 constexpr crash in the Core ML compiler."""
    return next(_fp8_constexpr_inputs(spec), None) is not None


def _set_blob_dtype(weight_path: str, offset: int, from_dtype: int, to_dtype: int) -> None:
    with open(weight_path, "r+b") as f:
        f.seek(offset)
        sentinel, dtype = struct.unpack("<II", f.read(8))
        if sentinel != _BLOB_METADATA_SENTINEL:
            raise ValueError(f"No blob metadata at offset {offset} of {weight_path}.")
        if dtype == to_dtype:
            return  # a blob shared by several ops, already relabeled
        if dtype != from_dtype:
            raise ValueError(
                f"Blob at offset {offset} of {weight_path} has dtype {dtype}, expected {from_dtype}."
            )
        f.seek(offset + 4)
        f.write(struct.pack("<I", to_dtype))


def _clone_tree(src: str, dst: str) -> None:
    """Copy a directory, with APFS copy-on-write clones on macOS so large weights are not duplicated."""
    if sys.platform == "darwin":
        if subprocess.run(["cp", "-c", "-R", src, dst], capture_output=True).returncode == 0:
            return
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)


def _relabel_package_to_int8(package_path: str) -> List[_Fp8Tensor]:
    """Relabel the FP8 constexpr inputs of a (copied) package as int8, in the spec and in the blobs."""
    spec = _load_spec(package_path)
    weights_dir = _try_get_weights_dir_path(package_path)
    tensors = []
    for tensor, messages in _fp8_constexpr_inputs(spec):
        blob = messages[0].blobFileValue
        weight_path = os.path.join(weights_dir, os.path.basename(blob.fileName))
        _set_blob_dtype(weight_path, blob.offset, tensor.blob_dtype, _BLOB_DTYPE_INT8)
        for message in messages:
            message.type.tensorType.dataType = proto.MIL_pb2.INT8
        tensors.append(tensor)

    with open(_ModelPackage(package_path).getRootModel().path(), "wb") as f:
        f.write(spec.SerializeToString())
    return tensors


def _restore_fp8_in_compiled_model(compiled_path: str, tensors: List[_Fp8Tensor]) -> None:
    """Turn the int8 stand-ins back into FP8 in the compiled model.mil and weight file."""
    mil_path = os.path.join(compiled_path, "model.mil")
    with open(mil_path) as f:
        lines = f.read().splitlines(keepends=True)

    wanted = {(t.function, t.op_name): t for t in tensors}
    blobs = {}
    function = None
    for i, line in enumerate(lines):
        func_match = _FUNC_RE.match(line)
        if func_match:
            function = func_match.group(1)
            continue
        name_match = _NAME_ATTR_RE.search(line)
        tensor = wanted.pop((function, name_match.group(1)), None) if name_match else None
        if tensor is None:
            continue

        if tensor.input_name is None:
            # tensor<int8, [..]> x = const()[name = string("x"), val = tensor<int8, [..]>(BLOBFILE(..))];
            pattern = re.compile(r"(?P<head>tensor<)int8(?P<tail>, )")
        else:
            # y = constexpr_blockwise_shift_scale(data = tensor<int8, [..]>(BLOBFILE(..)), ..)
            pattern = re.compile(
                r"(?P<head>\b" + re.escape(tensor.input_name) + r" = tensor<)int8"
                r"(?P<tail>, \[[^\]]*\]>\(" + _BLOBFILE + r")"
            )
        new_line, count = pattern.subn(r"\g<head>" + tensor.mil_dtype + r"\g<tail>", line)
        blob_match = re.search(_BLOBFILE, new_line)
        if count == 0 or blob_match is None:
            raise RuntimeError(
                f"Could not find the int8 stand-in for '{tensor.op_name}' in the compiled model.mil."
            )
        lines[i] = new_line
        blob_path = blob_match.group("path").replace("@model_path", compiled_path)
        blobs[(blob_path, int(blob_match.group("offset")))] = tensor.blob_dtype

    if wanted:
        raise RuntimeError(f"FP8 tensors not found in the compiled model.mil: {sorted(wanted)}")

    for (blob_path, offset), blob_dtype in blobs.items():
        _set_blob_dtype(blob_path, offset, _BLOB_DTYPE_INT8, blob_dtype)
    with open(mil_path, "w") as f:
        f.write("".join(lines))


def compile_with_fp8_workaround(package_path: str, destination_path: Optional[str] = None) -> str:
    """
    Compile an ``.mlpackage`` whose constexpr ops take FP8 data, working around the compiler crash.

    Returns the path of the compiled ``.mlmodelc`` (``destination_path`` if given).
    """
    from ..libcoremlpython import _MLModelProxy

    with tempfile.TemporaryDirectory() as tmp_dir:
        stand_in = os.path.join(tmp_dir, os.path.basename(package_path.rstrip("/")))
        _clone_tree(package_path, stand_in)
        tensors = _relabel_package_to_int8(stand_in)
        compiled_path = _MLModelProxy.compileModel(stand_in)

    if destination_path is not None:
        shutil.move(compiled_path, destination_path)
        compiled_path = destination_path
    _restore_fp8_in_compiled_model(compiled_path, tensors)
    return compiled_path
