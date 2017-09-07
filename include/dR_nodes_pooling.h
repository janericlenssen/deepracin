#ifndef DR_LAYERS_POOLING_H
#define DR_LAYERS_POOLING_H
#include "dR_types.h"

typedef struct dR_Pooling_Data dR_Pooling_Data;


enum dR_PoolingType {
    tMax,
    tAverage,
    tl2norm                         //Not Supported yet
};
typedef enum dR_PoolingType dR_PoolingType;

struct dR_Pooling_Data {
    dR_Shape3      ishape;          // Size of Input Buffer
    dR_Shape4     shape;
    dR_Shape4     stride;         // Example: [1,2,2,1] for a stride of 2x2 (calculate filter every 2x2 pixels)
    dR_PoolingType poolingType;
    gchar**     shaderSrc;
    gboolean    useLMEM;
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
dR_Node* dR_Pooling(dR_Graph* net, dR_Node* inputLayer, dR_Shape4* shape, dR_Shape4* stride, dR_PoolingType poolingtype);
#ifdef __cplusplus
}
#endif

gboolean dR_pooling_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_pooling_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_pooling_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_pooling_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_pooling_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_pooling_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_pooling_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_pooling_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_pooling_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_pooling_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_pooling_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);



// Optional

gboolean dR_pooling_generateKernel(dR_Graph* net, dR_Node* layer);

gchar* dR_pooling_createKernelName(dR_Node* convlayer);

gchar* dR_pooling_printLayer(dR_Node* layer);

#endif // DR_LAYERS_POOLING_H
