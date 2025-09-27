# neuralrx_testing_bench
software to test timing of neural receiver TensorRT engine 

Timing inference is executed on NRx_Inference_Timer.py

Code Breakdown

(1) **Imports**: Keep these the same

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import matplotlib.pyplot as plt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


(2) **Loading Engine**: change the name of the file used in the load_engine function. Make sure you use correct version. I ran this command to make sure Orin's TensorRT version matched with the system:

/usr/src/tensorrt/bin/trtexec \
  --onnx=receiver_model.onnx \
  --saveEngine=receiver_model_orin.trt \
  --fp16 \
  --workspace=4096


# --- Load TensorRT engine Python Code---
def load_engine(trt_file_path):
    with open(trt_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("receiver_model_orin.trt")
context = engine.create_execution_context()

(3) **# --- Allocate device memory for inputs and outputs ---** --> likely need to change these properties if attempting to use another trt file for inference testing
bindings = {}
input_bindings = []
output_bindings = []

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = tuple(engine.get_tensor_shape(name))
    size = int(np.prod(shape) * np.float32().nbytes)
    mem = cuda.mem_alloc(size)
    bindings[name] = mem
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        input_bindings.append(name)
    else:
        output_bindings.append(name)

(4) **Dummy testing + warmup**

# --- Create dummy inputs ---
for name in input_bindings:
    shape = tuple(engine.get_tensor_shape(name))
    dummy = np.random.rand(*shape).astype(np.float32)
    cuda.memcpy_htod(bindings[name], dummy)

# --- Build ordered list of device pointers ---
binding_addrs = [int(bindings[engine.get_tensor_name(i)]) for i in range(engine.num_io_tensors)]

# --- Warmup (discarded from stats) ---
for _ in range(20):
    context.execute_v2(binding_addrs)


(5) **Timing** --> likely does not need to change

# --- Timed inference ---
stream = cuda.Stream()
start_evt, end_evt = cuda.Event(), cuda.Event()
num_runs = 100
times = []

for _ in range(num_runs):
    start_evt.record(stream)
    context.execute_v2(binding_addrs)
    end_evt.record(stream)
    end_evt.synchronize()
    times.append(start_evt.time_till(end_evt))

# --- Stats (discard warmup bias) ---
valid_times = times[20:]  # skip first 20 runs
mean_lat = np.mean(valid_times)
std_lat = np.std(valid_times)
p95_lat = np.percentile(valid_times, 95)
max_lat = np.max(valid_times)

Everything else in the code is just data and visualizations using matplotlib.
