import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load engine
def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("model_neuralrx.plan")
context = engine.create_execution_context()

# Input shapes (from ONNX inspection)
batch_size = 1
y_shape = (batch_size, 14, 76, 4)  # y_realimag
no_shape = (batch_size,)            # no

# Output shape
out_shape = tuple(context.get_binding_shape(2-1))  # only 1 output, index 2-1=1

# Allocate device memory
d_y = cuda.mem_alloc(np.prod(y_shape) * np.dtype(np.float32).itemsize)
d_no = cuda.mem_alloc(np.prod(no_shape) * np.dtype(np.float32).itemsize)
d_out = cuda.mem_alloc(np.prod(out_shape) * np.dtype(np.float32).itemsize)

bindings = [int(d_y), int(d_no), int(d_out)]  # inputs then output
stream = cuda.Stream()

# Create dummy inputs
y_dummy = np.random.rand(*y_shape).astype(np.float32)
no_dummy = np.ones(no_shape, dtype=np.float32)  # can be 1.0

# Copy inputs to device
cuda.memcpy_htod_async(d_y, y_dummy, stream)
cuda.memcpy_htod_async(d_no, no_dummy, stream)

# Warmup
for _ in range(10):
    context.execute_v2(bindings)
    stream.synchronize()

# Timed runs
start, end = cuda.Event(), cuda.Event()
times = []
for _ in range(100):
    start.record(stream)
    context.execute_v2(bindings)
    end.record(stream)
    end.synchronize()
    times.append(start.time_till(end))

print(f"Average latency: {np.mean(times):.3f} ms")
print(f"Std dev: {np.std(times):.3f} ms")
