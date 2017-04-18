#ifndef DR_LAYERS_FILTER_H
#define DR_LAYERS_FILTER_H
#include "dR_types.h"


// /////////////////////////////////////////
// Apply Mask-dependent selected filter   //
// /////////////////////////////////////////


typedef struct dR_MDFilter_Data dR_MDFilter_Data;

struct dR_MDFilter_Data {
    dR_Shape3    iimgshape;           // Size of Input Image
    dR_Shape3    ifiltershape;        // Size of Input Image
    dR_Shape3    shape;               // Example: [5,5,3] for 5x5 filter size, 3 Filters to chose from
    gfloat*             filters;
    cl_mem              filterBuf;
    size_t              localWorkSizexy;
    size_t              globalWorkSizeX;
    size_t              globalWorkSizeY;
    gboolean            hasVariables;
    gint                gidXGap;
    gint                gidYGap;
    gboolean            useLMEM;
    gchar**             shaderSrc;
};

// Mandatory

#ifdef __cplusplus
extern "C"{
#endif

/**
* \brief Appends a layer that applies indiviual filters on an image.
* \details For each input pixel a filter window is chosen from a set of filters based on a filter selection mask that provides an integer for each pixel.
* Then, the corresponding filter is applied to this pixel in all input channels.
* Computes output[z,y,x] = Sum_{j,i} (input[z, y*stride.y + j, x*stride.x + i] * filter[filtermask[y,x], j+filterheight/2, i+filterwidth/2])
* j from {-filterheight/2,...,filterheight/2},
* i from {-filterwidth/2,...,filterwidth/2}
* assuming filter layout of [number of filters, filterheight, filterwidth]
*
* Generates output of size [number of inputchannels, input.y/stride.y, input.x/stride.x].
* For undefined regions of input (input[k,y,x] with x<0, x>=input.x, y<0 or y=>input.y) a zero padding is used.
*
* Functionality may be useful for context sensitive image transformations when a pixel classifier provides information which filter to use.
*
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputImage A node of the graph which is used as input image
* \param[in] inputFilterMask A node of the graph whose output is used as an indicator which filter to use for this pixel. Must have the same x,y-size as the inputImage node output.
* \param[in] shape A pointer to an integer array of size 4 containing [filterwidth, filterheight, number of inputchannels, number of outputchannels]. Last two values have to be equal.
*
*
* \returns The appended graph node

*/
dR_Node* dR_MaskDependentFilter(dR_Graph* net, dR_Node* inputImage, dR_Node* inputFilterMask, dR_Shape3 shape);

void        dR_MaskDependentFilter_setVariables(dR_Node* layer, gfloat* weights, gfloat* bias);
#ifdef __cplusplus
}
#endif

gboolean    dR_cdfilter_compute(dR_Graph* net, dR_Node* layer);

gboolean    dR_cdfilter_schedule(dR_Graph* net, dR_Node* layer);

gboolean    dR_cdfilter_propagateShape(dR_Graph* net, dR_Node* layer);

gint32      dR_cdfilter_getRequiredOutputBufferSize(dR_Node* layer);

gboolean    dR_cdfilter_createKernel(dR_Graph* net, dR_Node* layer);

gboolean    dR_cdfilter_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_cdfilter_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_cdfilter_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_cdfilter_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_cdfilter_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_cdfilter_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);


// Optional


gchar*      dR_cdfilter_createKernelName(dR_Node* convlayer);

gboolean    dR_cdfilter_generateKernel(dR_Graph* net, dR_Node* layer);

gchar*      dR_cdfilter_printLayer(dR_Node* layer);





// ////////////////////////////////
// Per Pixel  Filter Application //
// ////////////////////////////////

typedef struct dR_PPFilter_Data dR_PPFilter_Data;

struct dR_PPFilter_Data {
    dR_Shape3    iimgshape;           //  Size of Input Buffer
    dR_Shape3    ifiltershape;        //  Size of Input Buffer
    dR_Shape4    shape;                  // Example: [5,5,3,64] for 5x5 filter size, 3 input channels, 64 output channels
    dR_Shape4    stride;                 // Example: [1,2,2,1] for a stride of 2x2 (calculate filter every 2x2 pixels)
    gfloat*             weights;
    gfloat*             biases;
    cl_mem              weightsBuf;
    cl_mem              biasBuf;
    size_t              localWorkSizexy;
    size_t              globalWorkSizeX;
    size_t              globalWorkSizeY;
    gint                gidXGap;
    gint                gidYGap;
    size_t              numberOfDepthPartitions;
    gboolean            useLMEM;
    gchar**             shaderSrc;
};

// Mandatory

#ifdef __cplusplus
extern "C"{
    #endif
/**
* \brief Appends a layer that applies per-pixel-indiviual filters on an image.
* \details Similar to the CDfilter node, but with completely individual, generated filters for each pixel. For each input pixel a filter window has to be provided by the inputFilter node. Then, the corresponding filter is applied to this pixel in all input channels.
* Can be understood as a Conv2d layer with unique instead of shared weights with only one filter per pixel. (But is not a convolution, in fact)
* Computes output[z,y,x] = Sum_{j,i} (input[z, y*stride.y + j, x*stride.x + i] * filter[y, x, j+filterheight/2, i+filterwidth/2])
* j from {-filterheight/2,...,filterheight/2},
* i from {-filterwidth/2,...,filterwidth/2}
* assuming filter layout of [input.y, input.x, filterheight, filterwidth]
*
* Generates output of size [number of inputchannels, input.y/stride.y, input.x/stride.x].
* For undefined regions of input (input[k,y,x] with x<0, x>=input.x, y<0 or y=>input.y) a zero padding is used.
*
* Functionality may be useful for context sensitive image transformations when the individual filters are generated beforehand.
*
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputImage A node of the graph which is used as input image
* \param[in] inputFilter A node of the graph which is used as input filters.
* \param[in] shape A pointer to an integer array of size 4 containing [filterwidth, filterheight, number of inputchannels, number of outputchannels]. Last two values have to be equal.
* \param[in] stride A pointer to an integer array of size 2 containing [stride x, stride y]
*
*
* \returns The appended graph node

*/
    dR_Node* dR_PerPixelFilter(dR_Graph* net, dR_Node* inputImage, dR_Node* inputFilter, dR_Shape4 shape, dR_Shape4 stride);
    #ifdef __cplusplus
}
#endif

gboolean    dR_conv2duw_compute(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2duw_schedule(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2duw_propagateShape(dR_Graph* net, dR_Node* layer);

gint32      dR_conv2duw_getRequiredOutputBufferSize(dR_Node* layer);

gboolean    dR_conv2duw_createKernel(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2duw_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2duw_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2duw_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean    dR_conv2duw_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_conv2duw_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_conv2duw_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);


// Optional

gchar*      dR_conv2duw_createKernelName(dR_Node* convlayer);

gboolean    dR_conv2duw_generateKernel(dR_Graph* net, dR_Node* layer);

gchar*      dR_conv2duw_printLayer(dR_Node* layer);



#endif // DR_LAYERS_FILTER_H
