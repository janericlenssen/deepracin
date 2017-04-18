#ifndef DR_LAYERS_MATH_H
#define DR_LAYERS_MATH_H
#include "dR_types.h"

// /////////////////////////////
// Element-wise 2-operations  //
// /////////////////////////////


enum dR_ElemWise2OperationType {
    tAdd,
    tSub,
    tMul,
    tDiv,
    tPow
};
typedef enum dR_ElemWise2OperationType dR_ElemWise2OperationType;


typedef struct dR_ElemWise2Op_Data dR_ElemWise2Op_Data;

struct dR_ElemWise2Op_Data {
    dR_Shape3                  ishape;          // Actual Size of Input Buffer
    dR_ElemWise2OperationType  op;
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Appends a Add layer to a graph that adds a constant scalar to each element of input.
* \details Computes output[z,y,x] = input[z, y, x] + s
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputnode A node of the graph to which the layer should be appended.
* \param[in] scalar The scalar that should be added to each element
*
*
* \returns The appended graph node

*/
dR_Node* dR_ElemWise2Operation(dR_Graph* net, dR_Node* inputNode1, dR_Node* inputNode2, dR_ElemWise2OperationType op);
#ifdef __cplusplus
}
#endif

gboolean dR_elemwise2op_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise2op_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise2op_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_elemwise2op_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_elemwise2op_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise2op_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise2op_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise2op_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise2op_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_elemwise2op_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_elemwise2op_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);


// Optional

gchar* dR_elemwise2op_printLayer(dR_Node* layer);

// /////////////////////////////
// Element-wise 1-operations  //
// /////////////////////////////


enum dR_ElemWise1OperationType {
    tAddS,
    tSubS,
    tMulS,   // Not supported
    tDivS,
    tLog,
    tExp,
    tSqrt,
    tFill,
    tPowS
};
typedef enum dR_ElemWise1OperationType dR_ElemWise1OperationType;


typedef struct dR_ElemWise1Op_Data dR_ElemWise1Op_Data;

struct dR_ElemWise1Op_Data {
    dR_Shape3                  ishape;          // Actual Size of Input Buffer
    dR_ElemWise1OperationType  op;
    gfloat                     scalar;
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Appends a Add layer to a graph that adds a constant scalar to each element of input.
* \details Computes output[z,y,x] = input[z, y, x] + s
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputnode A node of the graph to which the layer should be appended.
* \param[in] scalar The scalar that should be added to each element
*
*
* \returns The appended graph node

*/
dR_Node* dR_ElemWise1Operation(dR_Graph* net, dR_Node* inputNode, dR_ElemWise1OperationType op, gfloat scalar);
#ifdef __cplusplus
}
#endif

gboolean dR_elemwise1op_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise1op_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise1op_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_elemwise1op_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_elemwise1op_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise1op_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise1op_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise1op_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_elemwise1op_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_elemwise1op_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_elemwise1op_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);


// Optional

gchar* dR_elemwise1op_printLayer(dR_Node* layer);



// //////////
// Softmax //
// //////////

typedef struct dR_Softmax_Data dR_Softmax_Data;


struct dR_Softmax_Data {
    dR_Shape3              ishape;          // Size of Input Buffer
    cl_kernel           clKernelComputeExp;
    gfloat*             expsHost;
};

// Mandatory
#ifdef __cplusplus
extern "C"{
#endif

/**
* \brief Appends a softmax layer to a graph that creates a softmax distribution from input.
* \details Computes output[x] = e^(input[x])/(sum_i (e^(input[i]))) with
* i from {0,...,number of input values}
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] inputnode A node of the graph to which the layer should be appended. If the input nodes output has more than one dimension, the data is serialized.
*
*
* \returns The appended graph node

*/

dR_Node* dR_Softmax(dR_Graph* net, dR_Node* inputLayer);
#ifdef __cplusplus
}
#endif

gboolean dR_softmax_compute(dR_Graph* net, dR_Node* layer);

gboolean dR_softmax_schedule(dR_Graph* net, dR_Node* layer);

gboolean dR_softmax_propagateShape(dR_Graph* net, dR_Node* layer);

gint32 dR_softmax_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_softmax_createKernel(dR_Graph* net, dR_Node* layer);

gboolean dR_softmax_allocateBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_softmax_fillBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_softmax_cleanupBuffers(dR_Graph* net, dR_Node* layer);

gboolean dR_softmax_cleanupLayer(dR_Graph* net, dR_Node* layer);

gchar*      dR_softmax_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node*    dR_softmax_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);



gchar* dR_softmax_printLayer(dR_Node* layer);



#endif // DR_LAYERS_MATH_H
