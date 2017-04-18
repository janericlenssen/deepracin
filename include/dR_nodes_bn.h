#ifndef DR_LAYERS_LRN_H
#define DR_LAYERS_LRN_H
#include "dR_types.h"


typedef struct dR_BN_Data dR_BN_Data;


struct dR_BN_Data {
    dR_Shape3    ishape;          // Size of Input Buffer
    gfloat              bias;
    gfloat              alpha;
    gfloat              beta;
};

#ifdef __cplusplus
extern "C"{
#endif
dR_Node* dR_BatchNormalization(dR_Graph* net, dR_Node* inputLayer);
#ifdef __cplusplus
}
#endif

gboolean dR_bn_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_bn_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_bn_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_bn_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_bn_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_bn_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_bn_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_bn_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_bn_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar* dR_bn_printLayer(dR_Node* layer);

#endif // DR_LAYERS_LRN_H
