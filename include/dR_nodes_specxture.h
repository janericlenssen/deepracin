#ifndef DR_LAYERS_SPECXTURE_H
#define DR_LAYERS_SPECXTURE_H
#include "dR_types.h"

// /////////////////////////////
// Specxture //
// /////////////////////////////


// Struct that contains all operation specific parameters
typedef struct dR_Specxture_Data dR_Specxture_Data;

struct dR_Specxture_Data {
    dR_Shape3                 ishape;
    gint32                    rmax;
    gint32                    x0;
    gint32                    y0;
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
dR_Node* dR_Specxture(dR_Graph* net, dR_Node* inputNode1);
#ifdef __cplusplus
}
#endif
/**
* \brief The compute function. All non-initialization compute functionality has to be called here. Set kernel parameters and enqueue kernels.

*/
gboolean dR_specxture_compute(dR_Graph* net, dR_Node* layer);

/**
* \brief The schedule function. All platform dependent scheduling and selection of parameters should be done here. Is executed in dR_prepare.
*/
gboolean dR_specxture_schedule(dR_Graph* net, dR_Node* layer);

/**
* \brief Computes the nodes output shape, given the inputs output shape. Is executed in dR_prepare.
*/
gboolean dR_specxture_propagateShape(dR_Graph* net, dR_Node* layer);

/**
* \brief Returns required output buffer size.
*/
gint32 dR_specxture_getRequiredOutputBufferSize(dR_Node* layer);


/**
* Calls all Opencl kernel creation routines that are required for this node. Is executed in dR_prepare.
*/
gboolean dR_specxture_createKernel(dR_Graph* net, dR_Node* layer);

/**
* Creates all OpenCL memory buffers that are required for this node (except input and output buffers). Is executed in dR_prepare.
*/
gboolean dR_specxture_allocateBuffers(dR_Graph* net, dR_Node* layer);

/**
* Performs initialization of all constant buffers, required for this node. Is executed in the "prepare"-step of dr.
*/
gboolean dR_specxture_fillBuffers(dR_Graph* net, dR_Node* layer);

/**
* Releases all buffers of this node. Is called in dR_cleanup.
*/
gboolean dR_specxture_cleanupBuffers(dR_Graph* net, dR_Node* layer);

/**
* Releases all layer specific memory.
*/
gboolean dR_specxture_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_specxture_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_specxture_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);

gchar* dR_specxture_printLayer(dR_Node* layer);

#endif
