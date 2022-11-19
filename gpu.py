current = 5

import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

data_ = np.arange(0, 5, 0.01)**2
#integrated_data = np.empty_like(data)

data = pycl_array.to_device(queue, data_)
integrated_data = pycl_array.empty() # Create an empty pyopencl destination array

program = cl.Program(context, """
#include <pyopencl-complex.h>

int[3] black_red(float x, float min, float max) {
  int R;
  if (x > max){
    R = 255;
  } else if (x < min){
    R = 0;
  } else {
    R = (int) 255 * (x - max) / (max-min);
  }
  return [R, 0, 0];
}

__kernel void cmap(__global const float* data, __global float width, __global float* output)
{
  int i = get_global_id(0);
  output[i][i] = black_red(data[i][i]);
}""").build()  # Create the OpenCL program

program.integrate(queue, (data.shape[0]-1,), None, data.data, integrated_data.data)  # Enqueue the program for execution and store the result in c

print("a: {}".format(data))
print("b: {}".format(integrated_data))
print(np.sum(integrated_data))

print(np.sum(data*0.01))
print(125/3)
# Print all three arrays, to show sum() worked