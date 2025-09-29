import onnx

# Load the ONNX model
model = onnx.load("receiver_model.onnx")

# Check it's valid
onnx.checker.check_model(model)

# Print input/output info
print("== Model Inputs ==")
for inp in model.graph.input:
    name = inp.name
    dims = [d.dim_value if (d.dim_value > 0) else -1 for d in inp.type.tensor_type.shape.dim]
    dtype = inp.type.tensor_type.elem_type
    print(f"Name: {name}, Shape: {dims}, Dtype: {dtype}")

print("\n== Model Outputs ==")
for out in model.graph.output:
    name = out.name
    dims = [d.dim_value if (d.dim_value > 0) else -1 for d in out.type.tensor_type.shape.dim]
    dtype = out.type.tensor_type.elem_type
    print(f"Name: {name}, Shape: {dims}, Dtype: {dtype}")
