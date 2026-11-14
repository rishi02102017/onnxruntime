import sys, logging, traceback, onnx, numpy as np
from onnx import helper, TensorProto
import onnxruntime as ort
print("ort", ort.__version__, ort.__file__, "onnx", onnx.__version__, "protobuf", __import__("google.protobuf").protobuf.__version__)

def run(name, fn):
    print(f"\n=== {name}")
    try:
        r = fn(); print("RESULT:", r)
    except Exception as e:
        print("EXC:", type(e).__name__, e)

from onnxruntime.tools.onnx_model_utils import is_fixed_size_tensor, fix_output_shapes, make_input_shape_fixed
from onnxruntime.tools.mobile_helpers import usability_checker as uc

# A: value info with elem_type but no shape (unknown rank)
vi_noshape = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
vi_dyn = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 3])
vi_fixed = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
vi_scalar = helper.make_tensor_value_info("y", TensorProto.FLOAT, [])
run("A1 is_fixed_size_tensor(no shape / unknown rank) expect False", lambda: (vi_noshape.type.tensor_type.HasField("shape"), bool(vi_noshape.type.tensor_type.shape), is_fixed_size_tensor(vi_noshape)))
run("A2 dyn expect False", lambda: is_fixed_size_tensor(vi_dyn))
run("A3 fixed expect True", lambda: is_fixed_size_tensor(vi_fixed))
run("A4 scalar expect True", lambda: (vi_scalar.type.tensor_type.HasField("shape"), is_fixed_size_tensor(vi_scalar)))

# A5: fix_output_shapes on model whose output has no shape
def a5():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    g = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [x], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    make_input_shape_fixed(m.graph, "x", [2, 3])
    fix_output_shapes(m)
    inferred = onnx.shape_inference.infer_shapes(m).graph.output[0]
    return {"output_after_fix": str(m.graph.output[0].type.tensor_type), "inferred": str(inferred.type.tensor_type)}
run("A5 fix_output_shapes fills unknown-rank output? (expect dims 2,3)", a5)

# A6: check_shapes counts unknown-rank value_info as fixed
def a6():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    s = helper.make_tensor_value_info("s", TensorProto.INT64, [None])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    nodes = [helper.make_node("Relu", ["x"], ["r"]), helper.make_node("Reshape", ["r", "s"], ["y"])]
    g = helper.make_graph(nodes, "g", [x, s], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m = onnx.shape_inference.infer_shapes(m)
    lg = logging.getLogger("t"); logging.basicConfig(level=logging.INFO)
    print("output y type:", m.graph.output[0].type.tensor_type)
    di, nd = uc.check_shapes(m.graph, lg)
    return {"dynamic_inputs": [i.name for i in di], "num_dynamic": nd}
run("A6 check_shapes: y has unknown rank (expect counted dynamic)", a6)

# B: check_shapes with logger None and no value_info on multi-node graph
def b():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    g = helper.make_graph([helper.make_node("Relu", ["x"], ["r"]), helper.make_node("Relu", ["r"], ["y"])], "g", [x], [y])
    return uc.check_shapes(g)
run("B check_shapes(graph) logger=None", b)

# D: TensorsData bad tuple
def d():
    from onnxruntime.quantization.calibrate import TensorsData, CalibrationMethod
    TensorsData(CalibrationMethod.MinMax, {"x": (np.float32(1), np.float32(2), np.float32(3))})
run("D TensorsData 3-tuple (expect TypeError)", d)

# C: optimize_model use_external_initializers
def c():
    import pathlib, tempfile
    from onnxruntime.tools.onnx_model_utils import optimize_model
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]); y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    m = helper.make_model(helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [x], [y]), opset_imports=[helper.make_opsetid("", 17)])
    d = pathlib.Path(tempfile.mkdtemp()); onnx.save(m, d / "m.onnx")
    optimize_model(d / "m.onnx", d / "o.onnx", use_external_initializers=True)
    return sorted(p.name for p in d.iterdir())
run("C optimize_model(use_external_initializers=True)", c)
