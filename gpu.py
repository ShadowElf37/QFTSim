current = 5

import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools
from gpyfft.fft import FFT

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

field_ = np.zeros((50,50), np.complex64)

field_frames_ = np.array([np.empty_like(field_) for _ in range(50)])

field1 = pycl_array.to_device(queue, field_)
field2 = pycl_array.empty_like(field_)  # Create an empty pyopencl destination array

transform = FFT(context, queue, field1, axes = (2, 1))

program = cl.Program(context, """
#include <pyopencl-complex.h>
__kernel void advance(__global const cfloat64 *a, __global const cfloat64 *b)
{
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
}""").build()  # Create the OpenCL program

program.advance(queue, a.shape, None, a.data, b.data, c.data)  # Enqueue the program for execution and store the result in c

print("a: {}".format(a))
print("b: {}".format(b))
print("c: {}".format(c))
# Print all three arrays, to show sum() worked

gpu_data = cl_array.to_device(queue, tData2D)
plan.execute(gpu_data.data)
eData2D = gpu_data.get()


ctx = cl.Context([cl.get_platforms()[0].get_devices()[0]])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
eData2D_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=eData2D)
eData2D_dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, eData2D.nbytes)
prg = cl.Program(ctx, """
        //#define PYOPENCL_DEFINE_CDOUBLE     // uncomment for double support.
        #include "pyopencl-complex.h"    
        __kernel void sum(const unsigned int ySize,
                              __global cfloat_t *a,
                              __global cfloat_t *b)
        {
          int gid0 = get_global_id(0);
          int gid1 = get_global_id(1);

          b[gid1 + ySize*gid0] = a[gid1 + ySize*gid0]+a[gid1 + ySize*gid0];
        }
        """).build()

prg.sum(queue, eData2D.shape, None, np.int32(Ny), eData2D_buf, eData2D_dest_buf)
cl.enqueue_copy(queue, eData2Dresult, eData2D_dest_buf)