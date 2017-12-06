/**
* \brief just passes the input
* \param[in] gInput Input Buffer
* \param[out] outputArr Output Image
* \author
*/

/*

  Summary of definitions adapted from http://opencl.codeplex.com/wikipage?title=OpenCL%20Tutorials%20-%201.

  SIMT: Single instruction, multiple thread. The same code is executed in parallel by a different threads, each thread executes the code with different data.

  WORK ITEM: Smallest execution entity.
  When a kernel is launched, lots of work items are launched, each executing the same code.
  Each work item has an ID, it is used to distinguish the data to be processed by each work item.
  Start as many work items as there are elements in array.

  WORK GROUP: They exist to allow communication and cooperation between work-items.
  They reflect how work-items are organized (can be N-dimensional grid of work-groups, with N = 1, 2 or 3).
  Work-groups also have a unique ID.
  There is no specified order on how work-items inside a group are processed. Work-group execution order is also undefined.

  ND-RANGE: Next organization level, specifying how work-groups are organized.

  KERNELS are functions called from the host, and are executed on a device. They are compiled at run-time.

  OPENCL PROGRAM is formed by kernels, functions, declarations, etc.

  GLOBAL ID: get_global_id(0) returns the id of the current work item in the first dimension. Returns the ID of the thread in execution.
  Many threads are launched at the same time and are executing the same kernel.
  Each thread receives a different ID, and will consequently perform a different computation.


  Allocate device memory, copy the data from the host to the device, set up a kernel, and copy the results back

  __global: memory allocated on the device

  OpenCL requires the image to be stored in RGBA

*/

/* Test function: copies the input and returns it */
__kernel void fft(
    const __global float * gInput,
    __global float * outputArr
    )
{
    /* Each work item has a three dimensional identifier */
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));

    outputArr[gid] = gInput[gid];
}
