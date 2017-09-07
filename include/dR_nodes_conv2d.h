#ifndef DR_LAYERS_CONV2D_H
#define DR_LAYERS_CONV2D_H
#include "dR_types.h"

// ///////////////////////////////////
// Standard 2D Convolutional Layer  //
// ///////////////////////////////////


typedef struct dR_Conv2d_Data dR_Conv2d_Data;

struct dR_Conv2d_Data {
    dR_Shape3    ishape;         // Size of Input Buffer
    dR_Shape4    shape;          // Example: [5,5,3,64] for 5x5 filter size, 3 input channels, 64 output channels
    dR_Shape4    stride;         // Example: [1,2,2,1] for a stride of 2x2 (calculate filter every 2x2 pixels)
    gfloat*             weights;
    gfloat*             biases;
    cl_mem              weightsBuf;
    cl_mem              biasBuf;
    size_t              localWorkSizexy;
    size_t              globalWorkSizeX;
    size_t              globalWorkSizeY;
    gint                gidXGap;
    gint                gidYGap;
    gint                winogradn;
    gint                winogradN;
    gint                winogradNperDim;
    gint              numberOfDepthPartitions;
    gint              numberOfDepthPartitionsInput;
    gint                lmemInputSize;
    gint                lmemFilterSize;
    gboolean            winogradWide;
    gboolean            useBias;
    gboolean            hasVariables;
    dR_ActivationType      activation;
    gboolean            useLMEM;
    gint                winoParallelizationPattern; // 0 for oParallel, 1 for iParallel, 2 for matrixElementParallel
};

// Mandatory

#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Appends a convolutional layer to a graph.
* \details Computes output[z,y,x] = Sum_{k,j,i} (input[k, y*stride.y + j, x*stride.x + i] * filter[k, z, j+filterheight/2, i+filterwidth/2]) with
* j from {-filterheight/2,...,filterheight/2},
* i from {-filterwidth/2,...,filterwidth/2}
* k from {0,...,number of inputchannels}
* assuming filter layout of [number of inputchannels, number of filters, filterheight, filterwidth]
*
* Generates output of size [number of filters, input.y/stride.y, input.x/stride.x].
* For undefined regions of input (input[k,y,x] with x<0, x>=input.x, y<0 or y=>input.y) a zero padding is used.
*
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputnode A node of the graph to which the layer should be appended.
* \param[in] filtershape A pointer to an integer array of size 4 containing [filterwidth, filterheight, number of inputchannels, number of filters]
* \param[in] stride A pointer to an integer array of size 2 containing [strideX, strideY]
* \param[in] activation type of activation function: 0 for linear activation (identity), 1 for ReLU activation, 2 for sigmoid activation, 3 for tanh activation
* \param[in] useBias True if bias addition should be used, False if not
*
*
* \returns The appended graph node

*/
dR_Node* dR_Conv2d(dR_Graph* net, dR_Node* inputnode, dR_Shape4* filtershape, dR_Shape4* stride, dR_ActivationType activation, gboolean useBias);

/**
* \brief Feeds variables (weights and biases) to a convolutional layer
*
* \author jan eric lenssen
*
* \param[in] node A convolutional layer node that should be fed with variables.
* \param[in] weights A pointer to float buffer containing the weights. Assuming a size of shape.s0*shape.s1*shape.s2*shape.s3
* with layout [number of inputchannels, number of filters, filterheight, filterwidth] (major: number of inputchannels, minor: filterwidth).
* \param[in] biases A pointer to a float buffer containing the biases. Assuming a size of [number of filters]. Can be NULL if useBias was set to False for node.
*
*
* \returns The appended graph node

*/
void        dR_Conv2d_setVariables(dR_Node* node, gfloat* weights, gfloat* biases);

#ifdef __cplusplus
}
#endif


gboolean    dR_conv2d_propagateShape(dR_Graph* net, dR_Node* layer);

gint32      dR_conv2d_getRequiredOutputBufferSize(dR_Node* layer);

gchar*      dR_conv2d_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_conv2d_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);

// Optional

gchar*      dR_conv2d_printLayer(dR_Node* layer);

// Direct Conv2d

gboolean    dR_conv2d_direct_createKernel(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_direct_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_direct_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_direct_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_direct_cleanupLayer(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_direct_compute(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_direct_schedule(dR_Graph* net, dR_Node* layer);

// Optional

gboolean    dR_conv2d_direct_generateKernel(dR_Graph* net, dR_Node* layer);

gchar*      dR_conv2d_direct_createKernelName(dR_Node* convlayer);

// Winograd Implementation

gboolean    dR_conv2d_winograd_createKernel(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_winograd_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_winograd_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_winograd_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_winograd_cleanupLayer(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_winograd_compute(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_winograd_schedule(dR_Graph* net, dR_Node* layer);

// Optional

gboolean    dR_conv2d_winograd_generateKernel(dR_Graph* net, dR_Node* layer);

gchar*      dR_conv2d_winograd_createKernelName(dR_Node* convlayer);


// 1x1 Conv2d

gboolean    dR_conv2d_1x1_createKernel(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_1x1_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_1x1_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_1x1_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_1x1_cleanupLayer(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_1x1_compute(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2d_1x1_schedule(dR_Graph* net, dR_Node* layer);

// Opt



// ////////////////////////////////////
//  2D Convolutional Layer Transpose //
// ////////////////////////////////////



typedef struct dR_Conv2dTranspose_Data dR_Conv2dTranspose_Data;

struct dR_Conv2dTranspose_Data {
    dR_Shape3    ishape;          // Size of Input Image
    dR_Shape4    shape;          // Example: [5,5,3,64] for 5x5 filter size, 3 input channels, 64 output channels
    dR_Shape4    stride;         // Example: [1,2,2,1] for a stride of 2x2 (calculate filter every 2x2 pixels)
    gfloat*             weights;
    gfloat*             biases;
    cl_mem              weightsBuf;
    cl_mem              biasBuf;
    size_t              localWorkSizexy;
    size_t              globalWorkSizeX;
    size_t              globalWorkSizeY;
    gint                gidXGap;
    gint                gidYGap;
    gint              numberOfDepthPartitions;
    gboolean            useBias;
    gboolean            hasVariables;
    dR_ActivationType      activation;
    gboolean            useLMEM;
    gchar**             shaderSrc;
};

// Mandatory

#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Appends a deconvolutional layer to a graph.
* \details Computes output[z,y,x] = Sum_{k,j,i} (input[k, y*stride.y + j, x*stride.x + i] * filter[k, z, j+filterheight/2, i+filterwidth/2]) with
* j from {-filterheight/2,...,filterheight/2},
* i from {-filterwidth/2,...,filterwidth/2}
* k from {0,...,number of inputchannels} TODO, still conv2d description
* assuming filter layout of [number of inputchannels, number of filters, filterheight, filterwidth]
*
* Generates output of size [number of filters, input.y/stride.y, input.x/stride.x].
* For undefined regions of input (input[k,y,x] with x<0, x>=input.x, y<0 or y=>input.y) a zero padding is used.
*
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputnode A node of the graph to which the layer should be appended.
* \param[in] filtershape A pointer to an integer array of size 4 containing [filterwidth, filterheight, number of outputchannels, number of inputchannels]
* \param[in] stride A pointer to an integer array of size 2 containing [strideX, strideY]
* \param[in] activation type of activation function: 0 for linear activation (identity), 1 for ReLU activation, 2 for sigmoid activation, 3 for tanh activation
* \param[in] useBias True if bias addition should be used, False if not
*
*
* \returns The appended graph node

*/
dR_Node* dR_Conv2dTransposed(dR_Graph* net, dR_Node* inputnode, dR_Shape4 filtershape, dR_Shape4 stride, dR_ActivationType activation, gboolean useBias);

/**
* \brief Feeds variables (weights) to a deconvolutional layer
*
* \author jan eric lenssen
*
* \param[in] node A deconvolutional layer node that should be fed with variables.
* \param[in] weights A pointer to float buffer containing the weights. Assuming a size of shape.s0*shape.s1*shape.s2*shape.s3
* with layout [number of inputchannels, number of filters, filterheight, filterwidth] (major: number of inputchannels, minor: filterwidth).
* \param[in] biases A pointer to a float buffer containing the biases. Assuming a size of [number of filters]. Can be NULL if useBias was set to False for node.
*
*
* \returns The appended graph node

*/
void        dR_Conv2dtransposed_setVariables(dR_Node* node, gfloat* weights, gfloat* biases);

#ifdef __cplusplus
}
#endif

gboolean    dR_conv2dtranspose_compute(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2dtranspose_schedule(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2dtranspose_propagateShape(dR_Graph* net, dR_Node* layer);

gint32      dR_conv2dtranspose_getRequiredOutputBufferSize(dR_Node* layer);

gboolean    dR_conv2dtranspose_createKernel(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2dtranspose_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2dtranspose_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2dtranspose_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2dtranspose_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_conv2dtranspose_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_conv2dtranspose_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);



// Optional

gchar*      dR_conv2dtranspose_createKernelName(dR_Node* convlayer);

gboolean    dR_conv2dtranspose_generateKernel(dR_Graph* net, dR_Node* layer);

gchar*      dR_conv2dtranspose_printLayer(dR_Node* layer);




#endif // DR_LAYERS_CONV2D_H
