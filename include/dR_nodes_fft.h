#ifndef DR_LAYERS_FFT_H
#define DR_LAYERS_FFT_H
#include "dR_types.h"

// /////////////////////////////
// Fast Fourier Transform     //
// /////////////////////////////


// Struct that contains all operation specific parameters
typedef struct dR_FFT_Data dR_FFT_Data;

struct dR_FFT_Data {
    dR_Shape3                 ishape;
    cl_kernel                 inverseKernel;
    cl_kernel                 transposeKernel;
    cl_kernel                 copyKernel;
    cl_kernel                 normalizeKernel;
    cl_mem                    intermedBuf;
    gboolean                  inv;
    gint32                    real_width;
    gint32                    real_height;
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Appends an FFT layer that computes the fast fourier transform on input image
*
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputnode A node of the graph to which the layer should be appended.
*
*
* \returns The appended graph node

*/
dR_Node* dR_FFT(dR_Graph* net, dR_Node* inputNode1, gboolean inverse);
#ifdef __cplusplus
}
#endif
/**
* \brief The compute function. All non-initialization compute functionality has to be called here. Set kernel parameters and enqueue kernels.

*/
gboolean dR_fft_compute(dR_Graph* net, dR_Node* layer);

/**
* \brief The schedule function. All platform dependent scheduling and selection of parameters should be done here. Is executed in dR_prepare.
*/
gboolean dR_fft_schedule(dR_Graph* net, dR_Node* layer);

/**
* \brief Computes the nodes output shape, given the inputs output shape. Is executed in dR_prepare.
*/
gboolean dR_fft_propagateShape(dR_Graph* net, dR_Node* layer);

/**
* \brief Returns required output buffer size.
*/
gint32 dR_fft_getRequiredOutputBufferSize(dR_Node* layer);


/**
* Calls all Opencl kernel creation routines that are required for this node. Is executed in dR_prepare.
*/
gboolean dR_fft_createKernel(dR_Graph* net, dR_Node* layer);

/**
* Creates all OpenCL memory buffers that are required for this node (except input and output buffers). Is executed in dR_prepare.
*/
gboolean dR_fft_allocateBuffers(dR_Graph* net, dR_Node* layer);

/**
* Performs initialization of all constant buffers, required for this node. Is executed in the "prepare"-step of dr.
*/
gboolean dR_fft_fillBuffers(dR_Graph* net, dR_Node* layer);

/**
* Releases all buffers of this node. Is called in dR_cleanup.
*/
gboolean dR_fft_cleanupBuffers(dR_Graph* net, dR_Node* layer);

/**
* Releases all layer specific memory.
*/
gboolean dR_fft_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_fft_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_fft_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);

gchar* dR_fft_printLayer(dR_Node* layer);

// fftshift implementation

typedef struct dR_FFTShift_Data dR_FFTShift_Data;

struct dR_FFTShift_Data {
    dR_Shape3                 ishape;
    cl_kernel                 copyKernel;
    cl_mem                    intermedBuf;
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Appends an FFT layer that computes the fast fourier transform on input image
*
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputnode A node of the graph to which the layer should be appended.
*
*
* \returns The appended graph node

*/
dR_Node* dR_FFTShift(dR_Graph* net, dR_Node* inputNode1);
#ifdef __cplusplus
}
#endif
/**
* \brief The compute function. All non-initialization compute functionality has to be called here. Set kernel parameters and enqueue kernels.

*/
gboolean dR_fftshift_compute(dR_Graph* net, dR_Node* layer);

/**
* \brief The schedule function. All platform dependent scheduling and selection of parameters should be done here. Is executed in dR_prepare.
*/
gboolean dR_fftshift_schedule(dR_Graph* net, dR_Node* layer);

/**
* \brief Computes the nodes output shape, given the inputs output shape. Is executed in dR_prepare.
*/
gboolean dR_fftshift_propagateShape(dR_Graph* net, dR_Node* layer);

/**
* \brief Returns required output buffer size.
*/
gint32 dR_fftshift_getRequiredOutputBufferSize(dR_Node* layer);


/**
* Calls all Opencl kernel creation routines that are required for this node. Is executed in dR_prepare.
*/
gboolean dR_fftshift_createKernel(dR_Graph* net, dR_Node* layer);

/**
* Creates all OpenCL memory buffers that are required for this node (except input and output buffers). Is executed in dR_prepare.
*/
gboolean dR_fftshift_allocateBuffers(dR_Graph* net, dR_Node* layer);

/**
* Performs initialization of all constant buffers, required for this node. Is executed in the "prepare"-step of dr.
*/
gboolean dR_fftshift_fillBuffers(dR_Graph* net, dR_Node* layer);

/**
* Releases all buffers of this node. Is called in dR_cleanup.
*/
gboolean dR_fftshift_cleanupBuffers(dR_Graph* net, dR_Node* layer);

/**
* Releases all layer specific memory.
*/
gboolean dR_fftshift_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_fftshift_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_fftshift_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);

gchar* dR_fftshift_printLayer(dR_Node* layer);

// fftabs implementation

#endif
