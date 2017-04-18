#ifndef DR_LAYERS_MISC_H
#define DR_LAYERS_MISC_H
#include "dR_types.h"



// //////////////
// Resolve RoI //
// //////////////

typedef struct dR_ResolveRoI_Data dR_ResolveRoI_Data;

struct dR_ResolveRoI_Data {
    dR_Shape3          ishape;          // Size of Input Buffer
    dR_Shape3          shape;           // Shape of RoI
    cl_kernel       clKernelRoIFromBufferIndex;

};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Appends a 'resolve region of interest' layer to a graph that produces a subimage of the input image as output.
* \details Resolve depends on the region of interest origin, given by the dR_setNodeRoI or dR_setNodeRoIBufferAndIndex functions.
* The origin can be set and changed between graph applications. The region of interest size however, needs to be defined in the graph creation step by this function.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputnode A node of the graph to which the layer should be appended.
* \param[in] roiDims The number of dimensions of the region of interest. Must be 1, 2 or 3.
* \param[in] size An integer array with size roiDims. Must contain the x,y and z dimension size of the region of interest.
*
*
* \returns The appended graph node

*/
dR_Node* dR_ResolveRoI(dR_Graph* net, dR_Node* inputLayer, dR_Shape3 shape);
#ifdef __cplusplus
}
#endif

gboolean dR_resolveRoI_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_resolveRoI_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_resolveRoI_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_resolveRoI_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_resolveRoI_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_resolveRoI_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_resolveRoI_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_resolveRoI_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_resolveRoI_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_resolveRoI_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_resolveRoI_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);


// Optional


gchar* dR_resolveRoI_printLayer(dR_Node* layer);




// ///////////////////
// RGB to Grayscale //
// ///////////////////

typedef struct dR_RGB2Gray_Data dR_RGB2Gray_Data;

struct dR_RGB2Gray_Data {
    dR_Shape3          ishape;          // Size of Input Buffer
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Appends a 'RGB to Grayscale' layer to a graph that produces an grayscale image with one channel out of an color image with three channels.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputnode A node of the graph to which the layer should be appended.
*
*
* \returns The appended graph node

*/
dR_Node* dR_RGB2gray(dR_Graph* net, dR_Node* inputLayer);
#ifdef __cplusplus
}
#endif

gboolean dR_rgb2gray_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_rgb2gray_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_rgb2gray_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_rgb2gray_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_rgb2gray_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_rgb2gray_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_rgb2gray_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_rgb2gray_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_rgb2gray_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_rgb2gray_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_rgb2gray_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);


// Optional


gchar* dR_rgb2gray_printLayer(dR_Node* layer);









// //////////////////
// Upscaling Layer //
// //////////////////

typedef struct dR_Upscaling_Data dR_Upscaling_Data;

enum dR_UpscalingType {
    tNearestNUp,
    tLinearUp,
    tBicubicUp   // Not supported
};
typedef enum dR_UpscalingType dR_UpscalingType;

struct dR_Upscaling_Data {
    dR_Shape3          ishape;          // Size of Input Buffer
    dR_UpscalingType   type;
    gint            scaleFactorX;
    gint            scaleFactorY;
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
dR_Node* dR_Upscaling(dR_Graph* net, dR_Node* inputLayer, dR_UpscalingType type, gint scaleFactorX, gint scaleFactorY);
#ifdef __cplusplus
}
#endif

gboolean dR_us_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_us_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_us_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_us_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_us_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_us_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_us_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_us_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_us_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_us_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_us_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);


// Optional

gboolean dR_us_generateKernel(dR_Graph* net, dR_Node* layer);

gchar* dR_us_createKernelName(dR_Node* uslayer);

gchar* dR_us_printLayer(dR_Node* layer);

// /////////////////
// Label Creation //
// /////////////////

typedef struct dR_LabelCreation_Data dR_LabelCreation_Data;

enum dR_LabelCreationType {
    t3to2ClassesBin,
    t3to2ClassesConf,
    t2ClassesBin,
    t2ClassesConf,
    tnClasses                         //Not Supported yet
};
typedef enum dR_LabelCreationType dR_LabelCreationType;

struct dR_LabelCreation_Data {
    dR_Shape3    ishape;          // Size of Input Buffer
    dR_LabelCreationType   type;
    gfloat              class0offset;
    gfloat              class1offset;
    gfloat              class2offset;
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
dR_Node* dR_LabelCreation(dR_Graph* net, dR_Node* inputLayer, dR_LabelCreationType type, gfloat class0offset, gfloat class1offset, gfloat class2offset);
#ifdef __cplusplus
}
#endif

gboolean dR_lc_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_lc_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_lc_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_lc_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_lc_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_lc_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_lc_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_lc_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_lc_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_lc_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_lc_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);



gchar* dR_lc_printLayer(dR_Node* layer);


// /////////////////
// Normalization  //
// /////////////////

typedef struct dR_Normalization_Data dR_Normalization_Data;

enum dR_NormalizationType {
    tNormMean,
    tNormDev,
    tNormMeanDev
};
typedef enum dR_NormalizationType dR_NormalizationType;

struct dR_Normalization_Data {
    dR_Shape3              ishape;          // Size of Input Buffer
    dR_NormalizationType   type;
    gfloat              targetMean;
    gfloat              targetDev;
    cl_mem              tempFloatBuffer1;
    cl_mem              tempFloatBuffer2;
    cl_kernel           clKernelGetAvg;
    cl_kernel           clKernelGetDev;
    gint                localworksizex;
    gint                localworksizey;
    gint                globalWorkSizeX;
    gint                globalWorkSizeY;
    gfloat*             resultHost;
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
dR_Node* dR_Normalization(dR_Graph* net, dR_Node* inputLayer, dR_NormalizationType type, gfloat targetMean, gfloat targetDev);
#ifdef __cplusplus
}
#endif

gboolean dR_norm_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_norm_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_norm_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_norm_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_norm_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_norm_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_norm_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_norm_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_norm_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_norm_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_norm_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);



gchar* dR_norm_printLayer(dR_Node* layer);

#endif // DR_LAYERS_MISCLAYERS_H
