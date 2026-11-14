# Exercise the installed onnxruntime.tools package (wheel, optionally with the repo file copied over).
import os
import subprocess
import sys
import tempfile

import onnx
from onnx import TensorProto, helper

import onnxruntime
from onnxruntime.tools.mobile_helpers.usability_checker import check_shapes
from onnxruntime.tools.onnx_model_utils import is_fixed_size_tensor

print("onnxruntime", onnxruntime.__version__, "onnx", onnx.__version__)
ok = True


def expect(name, got, want):
    global ok
    status = "OK " if got == want else "BAD"
    ok &= got == want
    print(f"{status} {name}: got={got} want={want}")


expect("shapeless fixed", is_fixed_size_tensor(helper.make_tensor_value_info("x", TensorProto.FLOAT, None)), False)
expect("scalar fixed", is_fixed_size_tensor(helper.make_tensor_value_info("x", TensorProto.FLOAT, [])), True)
expect("[2,3] fixed", is_fixed_size_tensor(helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])), True)
expect("[N,3] fixed", is_fixed_size_tensor(helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])), False)

graph = helper.make_graph(
    [helper.make_node("Relu", ["X"], ["Y"])],
    "g",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
)
model = helper.make_model(graph)
with tempfile.TemporaryDirectory() as d:
    src, dst = os.path.join(d, "in.onnx"), os.path.join(d, "out.onnx")
    onnx.save(model, src)
    subprocess.run(
        [sys.executable, "-m", "onnxruntime.tools.make_dynamic_shape_fixed", "--dim_param", "N", "--dim_value", "2",
         src, dst],
        check=True,
    )
    out = onnx.load(dst)
    y = out.graph.output[0].type.tensor_type
    expect("make_dynamic_shape_fixed output dims",
           [dim.dim_value for dim in y.shape.dim] if y.HasField("shape") else None, [2, 3])

    fixed_in = onnx.load(src)
    fixed_in.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 2
    expect("check_shapes dynamic count (shapeless output)", check_shapes(fixed_in.graph)[1], 1)

print("REPRO", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
