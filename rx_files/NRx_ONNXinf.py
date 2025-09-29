import onnxruntime as ort
import numpy as np

# Load ONNX
sess = ort.InferenceSession("receiver_model.onnx")

# Print inputs for clarity
for i in sess.get_inputs():
    print(i.name, i.shape, i.type)

# Create dummy inputs
batch_size = 1
y_dummy = np.random.rand(batch_size, 14, 96, 2).astype(np.float32)  # Y_real_imag
hr_dummy = np.random.rand(batch_size, 14, 96, 2).astype(np.float32) # Hr_real_imag

# Run inference with both inputs
outputs = sess.run(None, {
    'Y_real_imag': y_dummy,
    'Hr_real_imag': hr_dummy
})

# Inspect output
print("Output shape:", outputs[0].shape)  # should be [1, 14, 96, 2]
