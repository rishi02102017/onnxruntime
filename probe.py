import subprocess, sys, tempfile, pathlib, onnx
from onnx import helper, TensorProto
import onnxruntime as ort
print("ort", ort.__version__)
d = pathlib.Path(tempfile.mkdtemp())
# Model: dynamic input x[N,3] -> Relu -> y (output declared without shape, as some exporters emit)
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
m = helper.make_model(helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [x], [y]), opset_imports=[helper.make_opsetid("", 17)])
onnx.save(m, d / "in.onnx")
r = subprocess.run([sys.executable, "-m", "onnxruntime.tools.make_dynamic_shape_fixed", "--input_name", "x", "--input_shape", "2,3", str(d / "in.onnx"), str(d / "out.onnx")], capture_output=True, text=True)
print("rc", r.returncode, r.stdout[-500:], r.stderr[-500:])
out = onnx.load(d / "out.onnx")
print("CLI output y type:", repr(str(out.graph.output[0].type.tensor_type)))
r = subprocess.run([sys.executable, "-m", "onnxruntime.tools.make_dynamic_shape_fixed", "--dim_param", "N", "--dim_value", "2", str(d / "in.onnx"), str(d / "out2.onnx")], capture_output=True, text=True)
print("dim_param rc", r.returncode, "y:", repr(str(onnx.load(d / "out2.onnx").graph.output[0].type.tensor_type)))

# Mobile usability checker end-to-end: Reshape with runtime shape -> unknown-rank value feeding Relu
x2 = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
s = helper.make_tensor_value_info("s", TensorProto.INT64, [2])
o = helper.make_tensor_value_info("o", TensorProto.FLOAT, None)
nodes = [helper.make_node("Reshape", ["x", "s"], ["r"]), helper.make_node("Relu", ["r"], ["o"])]
m2 = helper.make_model(helper.make_graph(nodes, "g", [x2, s], [o]), opset_imports=[helper.make_opsetid("", 17)])
inf = onnx.shape_inference.infer_shapes(m2)
print("value_info after shape inference:", [(v.name, str(v.type.tensor_type).replace("\n"," ")) for v in inf.graph.value_info])
print("output:", str(inf.graph.output[0].type.tensor_type).replace("\n"," "))
onnx.save(m2, d / "m2.onnx")
r = subprocess.run([sys.executable, "-m", "onnxruntime.tools.check_onnx_model_mobile_usability", str(d / "m2.onnx")], capture_output=True, text=True)
print("usability rc", r.returncode); print(r.stdout[-3000:]); print(r.stderr[-3000:])

# Patched variant
from onnxruntime.tools import onnx_model_utils as omu
def fixed(value):
    is_fixed = False
    if value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape"):
        is_fixed = all(d.HasField("dim_value") and d.dim_value > 0 for d in value.type.tensor_type.shape.dim)
    return is_fixed
for vi in [y, x, helper.make_tensor_value_info("a", TensorProto.FLOAT, [2,3]), helper.make_tensor_value_info("b", TensorProto.FLOAT, [])]:
    print("orig", omu.is_fixed_size_tensor(vi), "patched", fixed(vi), vi.name)
