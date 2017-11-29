#ifndef DR_LAYERS_FFT_H
#define DR_LAYERS_FFT_H
#include "dR_types.h"

// /////////////////////////////
// Fast Fourier Transform     //
// /////////////////////////////


// Struct that contains all operation specific parameters
typedef struct dR_FFT_Data dR_FFT_Data;

struct dR_FFT_Data {
    dR_Shape3                  ishape;
};

// TODO: struct for real and img part of complex number ?

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
dR_Node* dR_FFT(dR_Graph* net, dR_Node* inputNode1);
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

#endif
