import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# --- Load TensorRT engine ---
def load_engine(trt_file_path):
    with open(trt_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("model_neuralrx.trt")
context = engine.create_execution_context()

# --- Allocate device memory for inputs and outputs ---
bindings = []
input_bindings = []
output_bindings = []

for i in range(engine.num_bindings):
    shape = tuple(context.get_binding_shape(i))
    size = np.prod(shape) * np.float32().nbytes
    mem = cuda.mem_alloc(size)
    bindings.append(mem)
    if engine.binding_is_input(i):
        input_bindings.append(i)
    else:
        output_bindings.append(i)

# --- Create dummy inputs (random floats) based on engine input shapes ---
for i in input_bindings:
    shape = tuple(context.get_binding_shape(i))
    dummy = np.random.rand(*shape).astype(np.float32)
    cuda.memcpy_htod(bindings[i], dummy)

# --- Warmup passes (to initialize GPU and optimize) ---
for _ in range(10):
    context.execute_v2(bindings)

# --- Timed inference ---
stream = cuda.Stream()
start_evt, end_evt = cuda.Event(), cuda.Event()
num_runs = 100
times = []

for _ in range(num_runs):
    start_evt.record(stream)
    context.execute_v2(bindings)
    end_evt.record(stream)
    end_evt.synchronize()
    times.append(start_evt.time_till(end_evt))

print(f"Inference average latency: {np.mean(times):.3f} ms")
print(f"Inference std deviation: {np.std(times):.3f} ms")

# --- Optional: retrieve output ---
outputs = []
for i in output_bindings:
    shape = tuple(context.get_binding_shape(i))
    out_host = np.empty(shape, dtype=np.float32)
    cuda.memcpy_dtoh(out_host, bindings[i])
    outputs.append(out_host)

print("Output shapes:", [o.shape for o in outputs])
