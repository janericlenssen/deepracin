#ifndef DR_CLWRAP_H
#define DR_CLWRAP_H
#include "dR_base.h"

#ifdef __cplusplus
extern "C"{
#endif

/** \brief Upload an array from host memory to OpenCL memory
*
* \author  pascal
*
* \param filter Pointer to the filter class.
* \param[in] name Name for the array, is used for erro handling
* \param[in, out] hostMem The host memory to upload.
* \param[in] offset Memory offset
* \param[in] numBytes Number of bytes to download
* \param[in] deviceMem The OpenCL memory.
*/
gboolean dR_uploadArray(dR_Graph* net, gchar* name, void * srcHostMem, size_t offset, size_t numBytes, cl_mem destDeviceMem);


/** \brief Download an array from OpenCL memory to host memory
*
* \author  pascal libuschewski
*
* \param filter Pointer to the filter class.
* \param[in] name Name for the array, is used for error handling
* \param[in] deviceMem The OpenCL memory to download
* \param[in] byteOffset The offset in bytes in the buffer object to read from.
* \param[in] numBytes The size in bytes of data being read.
* \param[in, out] hostMem The pointer to buffer in host memory where data is to be read into.
*/
void dR_downloadArray(dR_Graph* net, gchar* name, cl_mem* srcDeviceMem, size_t byteOffset, size_t numBytes, void * destHostMem);

/**
*\brief A method for creating an OpenCL float buffer.
*
* \author  jan eric lenssen
*
* \param[in,out] net Pointer to the NN struct.
* \param[out] mem An OpenCL Memory Object into wich the newly created kernel is stored.
* \param[in] size Size of the Floatbuffer.
*/
gboolean dR_createFloatBuffer(dR_Graph* net, cl_mem* mem, gint size, int buffertype);

#ifdef __cplusplus
}
#endif

/**
* \brief Creates a cl context with the device type set in the property clDeviceType
* (stored in filter->clDeviceType)
*/
gboolean dR_clCreateCLContext(dR_Graph* net);


/**
*\brief A method for creating a single OpenCL kernel, given by the kernelName.
*
* \author  pascal
*
* \param[in,out] filter Pointer to the filter class.
* \param[out] kernel A OpenCL Kernel into wich the newly created kernel is stored.
* \param[in] kernelName The name of the kernel (is used for creating the kernel by name).
*/
gboolean dR_createKernel(dR_Graph* net, char * kernelName, cl_kernel* kernel);





/**
* \brief Creates clContext, clCommandQueue, clProgram and compiles the different clKernels
* Displays a device info if filter->silent is false
*/
gboolean dR_clInit(dR_Graph* net);


/**
* \brief Builds the CL program using given source files
*/
gboolean dR_clBuildWithSource(dR_Graph* net, const gchar * const filenames[], gint numberoffiles);

/**
* \brief Loads ptx code and tries to build program
*/
gboolean dR_clTryToBuildWithBinaries(dR_Graph* net);

/**
* \brief Creates ptx code for the kernels
* \details \see http://forums.nvidia.com/index.php?showtopic=171016
* and \see http://www.ros.org/doc/api/parallel_quickstep/html/opencl__utils_8cpp_source.html
*/
gboolean dR_clWriteBinaries(dR_Graph* net);

/** \brief A method for finishing up a kernel. Is called after each pass of the kernel.
*
* \author  pascal
*
* \param[in,out] filter Pointer to the filter class.
* \param[in] kernelName The name of the kernel (is used for the error report).
*/
gboolean dR_finishCLKernel(dR_Graph* net, char * kernelName);

/**
* \brief Checks for a OpenCL Error stored in net->clConfig->clError (if net->clConfig->clError != CL_SUCCESS)
*
*/
gboolean dR_openCLError(dR_Graph * net, char * message, char * callerName);

/**
* \brief Prints some runtime information of the kernel given by a cl_kernel. Kernel name is optional.
*
* \author  pascal
*
* \param[in,out] filter Pointer to the filter class.
* \param[in] kernel A OpenCL Kernel.
* \param[in] kernelName The name of the kernel (is used only for console output).
*/
void dR_kernelInfo(dR_Graph* net, cl_kernel kernel, char* kernelName);


gboolean dR_openCLErrorWithoutCleanup(dR_Graph* net, char * message, char * callerName);


/** \brief Releases the CPU and the GPU memory.
*
* \author  pascal
*
* \param filter Pointer to the filter class.
* \return TRUE or FALSE.
*/
gboolean dR_clMemoryBufferCleanup(dR_Graph* net, cl_mem mem);

gboolean dR_cleanupKernel(cl_kernel kernel);

gboolean dR_cleanupProgram(cl_program program);

gboolean dR_cleanupCL(dR_Graph* net);

gboolean dR_exportFloatArrayToCSV(dR_Graph* net, gchar * filepath, gchar * filename, cl_mem deviceMem, cl_float * hostMem);

#endif // DR_CLWRAP_H
