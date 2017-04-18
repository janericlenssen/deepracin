#ifndef DR_LAYERS_TRANSFORM_H
#define DR_LAYERS_TRANSFORM_H
#include "dR_types.h"

// /////////////////////
// Extract Slice Node //
// /////////////////////

typedef struct dR_Slice_Data dR_Slice_Data;

struct dR_Slice_Data {
    dR_Shape4    ishape;          // Size of Input Buffer
    dR_Shape4    origin;          // Origin of slice to cut out
    dR_Shape4    oshape;          // Output shape
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Cuts out a part of another nodes output
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputnode A node of the graph to which the layer should be appended.
* \param[in] origin The origin of the slice which should be cut out
* \param[in] shape The shape of the slice
*
*
* \returns The appended graph nodes

*/
dR_Node* dR_Slice(dR_Graph* net, dR_Node* inputLayer, dR_Shape4 origin, dR_Shape4 shape);
#ifdef __cplusplus
}
#endif

gboolean dR_slice_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_slice_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_slice_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_slice_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_slice_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_slice_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_slice_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_slice_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_slice_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_slice_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_slice_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);


// Optional

gchar* dR_slice_printLayer(dR_Node* layer);



// ///////////////
// Concat Layer //
// ///////////////

typedef struct dR_Concat_Data dR_Concat_Data;

struct dR_Concat_Data {
    gint                    n;
    gint                    concatDim;
    gint                    maxInputSizeInConcatDim;
    gint*                   inputSizesInConcatDim;
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Concatenates n nodes in a specific dimension x
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputNodes A list of node of the graph to which the layer should be appended.
* \param[in] n The number of nodes to be concatenated
* \param[in] concatDim the dimension along which the concat should be performed
*
*
* \returns The appended graph node

*/
dR_Node* dR_Concat(dR_Graph* net, dR_Node** inputNodes, gint n, gint concatDim);
#ifdef __cplusplus
}
#endif

gboolean dR_concat_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_concat_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_concat_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_concat_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_concat_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_concat_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_concat_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_concat_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_concat_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_concat_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_concat_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);


// Optional

gchar* dR_concat_printLayer(dR_Node* layer);

gboolean  dR_concat_generateKernel(dR_Graph* net, dR_Node* layer);

gchar*    dR_concat_createKernelName(dR_Node* convlayer);




// ///////////////////
// Crop or Pad Node //
// ///////////////////

typedef struct dR_CropOrPad_Data dR_CropOrPad_Data;

struct dR_CropOrPad_Data {
    dR_Shape3    ishape;          // Size of Input Buffer
    dR_Shape3    oshape;          // Output shape
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Crops or pads a buffer to a target size. Input data starts at origin (0, 0, 0) in outputbuffer. Pads with Zeros.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputnode A node of the graph to which the layer should be appended.
* \param[in] origin The origin of the slice which should be cut out
* \param[in] shape The shape of the slice
*
*
* \returns The appended graph nodes

*/
dR_Node* dR_CropOrPad(dR_Graph* net, dR_Node* inputLayer, dR_Shape3 targetshape);
#ifdef __cplusplus
}
#endif

gboolean dR_croporpad_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_croporpad_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_croporpad_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_croporpad_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_croporpad_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_croporpad_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_croporpad_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_croporpad_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_croporpad_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_croporpad_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_croporpad_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);


// Optional

gchar* dR_croporpad_printLayer(dR_Node* layer);




#endif // DR_LAYERS_TRANSFORM_H
