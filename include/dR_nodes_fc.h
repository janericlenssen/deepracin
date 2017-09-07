#ifndef DR_LAYERS_FC_H
#define DR_LAYERS_FC_H
#include "dR_types.h"

typedef struct dR_FC_Data dR_FC_Data;

struct dR_FC_Data {
    dR_Shape3    ishape;          //  Size of Input Buffer
    dR_Shape2    shape;          // Example: [384,16] 384 to 16 neurons fully connected
    gfloat*             weights;
    gfloat*             biases;
    cl_mem              weightsBuf;
    cl_mem              biasBuf;
    size_t              localWorkSizeStepOne;
    size_t              localWorkSizeStepTwo;
    cl_int              sizeOfPartitions;
    size_t              globalWorkSizeX;
    size_t              globalWorkSizeY;
    cl_kernel           clKernelStepTwo;
    gboolean            useBias;
    gboolean            hasVariables;
    dR_ActivationType      activation;
    gchar**             shaderSrc;
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
dR_Node* dR_FullyConnected(dR_Graph* net, dR_Node* inputLayer, dR_Shape2* shape, dR_ActivationType activation, gboolean useBias);

void dR_FullyConnected_setVariables(dR_Node* layer, gfloat* weights, gfloat* bias);

#ifdef __cplusplus
}
#endif

gboolean dR_fc_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_fc_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_fc_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_fc_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_fc_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_fc_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_fc_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_fc_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_fc_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_fc_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_fc_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);




// Optional

gchar* dR_fc_printLayer(dR_Node* layer);

#endif // DR_LAYERS_FC_H
