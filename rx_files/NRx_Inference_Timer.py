import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import matplotlib.pyplot as plt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) #for warnings

# --- Load TensorRT engine ---
def load_engine(trt_file_path):
    with open(trt_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("receiver_model_orin.trt") #needs to be changed if using different TRT model
context = engine.create_execution_context()

# --- Allocate device GPU memory for inputs and outputs ---
bindings = {}
input_bindings = []
output_bindings = []

for i in range(engine.num_io_tensors):  #these shapes/sizes will need to be changed based on models inputs/outputs
    name = engine.get_tensor_name(i)
    shape = tuple(engine.get_tensor_shape(name))
    size = int(np.prod(shape) * np.float32().nbytes)
    mem = cuda.mem_alloc(size)
    bindings[name] = mem
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        input_bindings.append(name)
    else:
        output_bindings.append(name)

# --- Build ordered list of device pointers for testing---
binding_addrs = [int(bindings[engine.get_tensor_name(i)]) for i in range(engine.num_io_tensors)]

# --- Warmup (discarded from stats) ---
for _ in range(20):
    # generate new dummy inputs for warmup too
    for name in input_bindings:
        shape = tuple(engine.get_tensor_shape(name))
        dummy = np.random.rand(*shape).astype(np.float32)
        cuda.memcpy_htod(bindings[name], dummy)
    context.execute_v2(binding_addrs)

# --- Timed inference ---
stream = cuda.Stream()
start_evt, end_evt = cuda.Event(), cuda.Event()
num_runs = 100
times = []

for _ in range(num_runs):
    # feed new dummy input every run
    for name in input_bindings:
        shape = tuple(engine.get_tensor_shape(name))
        dummy = np.random.rand(*shape).astype(np.float32)
        cuda.memcpy_htod(bindings[name], dummy)

    start_evt.record(stream)
    context.execute_v2(binding_addrs)
    end_evt.record(stream)
    end_evt.synchronize()
    times.append(start_evt.time_till(end_evt))

# --- Stats (I discarded warmup) ---
valid_times = times[20:] if len(times) > 20 else times
mean_lat = np.mean(valid_times)
std_lat = np.std(valid_times)
p95_lat = np.percentile(valid_times, 95)
max_lat = np.max(valid_times)

print("\n==== Inference Latency Report ====")
print(f"Mean latency     : {mean_lat:.3f} ms")
print(f"Std deviation    : {std_lat:.3f} ms")
print(f"95th percentile  : {p95_lat:.3f} ms")
print(f"Max latency      : {max_lat:.3f} ms")
print("=================================\n")

# --- Retrieve outputs (from last run) ---
outputs = []
for name in output_bindings:
    shape = tuple(engine.get_tensor_shape(name))
    out_host = np.empty(shape, dtype=np.float32)
    cuda.memcpy_dtoh(out_host, bindings[name])
    outputs.append(out_host)

print("Output shapes:", [o.shape for o in outputs])

# --- Latency Histogram ---
plt.figure(figsize=(6,4))
plt.hist(valid_times, bins=15, color="steelblue", edgecolor="black", alpha=0.7)
plt.axvline(mean_lat, color="red", linestyle="--", label=f"Mean: {mean_lat:.2f} ms")
plt.axvline(p95_lat, color="green", linestyle="--", label=f"95% < {p95_lat:.2f} ms")
plt.xlabel("Inference Latency (ms)")
plt.ylabel("Frequency")
plt.title("TensorRT Inference Latency Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# --- Latency Trend Plot ---
plt.figure(figsize=(6,4))
plt.plot(valid_times, marker="o", markersize=3, linewidth=1, color="darkorange")
plt.axhline(mean_lat, color="red", linestyle="--", label=f"Mean: {mean_lat:.2f} ms")
plt.axhline(p95_lat, color="green", linestyle="--", label=f"95% < {p95_lat:.2f} ms")
plt.xlabel("Run Index (post-warmup)")
plt.ylabel("Latency (ms)")
plt.title("Inference Latency Over Runs")
plt.legend()
plt.tight_layout()
plt.show()
