import onnxruntime as ort
import numpy as np

# Load ONNX
sess = ort.InferenceSession("neural_receiver.onnx")

# Print inputs for clarity
for i in sess.get_inputs():
    print(i.name, i.shape, i.type)

# Create dummy inputs
batch_size = 1
y_realimag_dummy = np.random.rand(batch_size, 14, 76, 4).astype(np.float32)
no_dummy = np.array([1.0], dtype=np.float32)  # same batch size

# Run inference with **both inputs**
outputs = sess.run(None, {
    'y_realimag': y_realimag_dummy,
    'no': no_dummy
})

# Inspect output
print("Output shape:", outputs[0].shape)
