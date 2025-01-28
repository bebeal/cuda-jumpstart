import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void add_vec(float *a, float *b, float *c, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}
""")

add_vec = mod.get_function("add_vec")

n = 1024
a = np.ones(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)*2
c = np.empty_like(a)

add_vec(
    drv.In(a), drv.In(b), drv.Out(c), np.int32(n),
    block=(256,1,1), grid=((n+255)//256,1,1)
)

print(c[0])  # Expect 3.0
