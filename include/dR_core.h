#ifndef DR_CORE_H
#define DR_CORE_H

#include "dR_clwrap.h"
#include "dR_base.h"


dR_Graph* dR_appendLayer(dR_Graph* net, dR_Node* l);


gboolean dR_generateAndCompileKernels(dR_Graph* net);

gboolean dR_propagateShapesAndSchedule(dR_Graph* net);

gboolean dR_estimateAndAllocateBuffers(dR_Graph* net);

gboolean dR_createKernels(dR_Graph* net);

gboolean dR_fillBuffers(dR_Graph* net);


// DataFeedNode Functions
/**
* \brief Appends a data feed node to a graph.
* \details Data feed nodes are 'start' nodes in the graph which can be fed with data from the CPU or external OpenCL buffers. They must not have other nodes as inputs.
* Every data feed node needs to have data before the graph is executed. The spatial size of buffer feed nodes have to be defined in the graph creation step.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] shape A Shape3 struct or pointer to an int array  with 3 elements. Describes [x,y,z] shape of node data.
*
* \returns The appended graph node

*/
#ifdef __cplusplus
extern "C"{
#endif
dR_Node* dR_Datafeednode(dR_Graph* graph, dR_Shape3* shape);
#ifdef __cplusplus
}
#endif
gchar* dR_datafeednode_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);

dR_Node* dR_datafeednode_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);

gchar* dR_datafeednode_printLayer(dR_Node* layer);

gboolean dR_datafeednode_allocateBuffers(dR_Graph* net, dR_Node* layer);

gint32 dR_datafeednode_getRequiredOutputBufferSize(dR_Node* layer);

gboolean dR_datafeednode_cleanupLayer(dR_Graph* net, dR_Node* layer);


/**
* \brief Cleans up Buffers and Kernels
*
* \author jan eric lenssen
*
* \param net Parameter for the dR_Graph Object

*/
void dR_cleanupBuffers(dR_Graph* net);

/**
* \brief Cleans up deepRACIN data structures
*
* \author jan eric lenssen
*
* \param net Parameter for the dR_Graph Object

*/
void dR_cleanupNet(dR_Graph* net);


dR_MemoryHandler* dR_newMemoryHandler(gboolean custom);


void dR_cleanupMemoryHandler(dR_Graph* net, dR_MemoryHandler* handler);

// LayerList Functionality //

dR_List* dR_list_createEmptyList(void);

void dR_list_append(dR_List* list, void* elem);

gboolean dR_list_removeFirstOcc(dR_List* list, void* elem);

void* dR_list_pop(dR_List* list);

void* dR_list_next(dR_List* list);

void dR_list_resetIt(dR_List* list);

void dR_list_cleanup(dR_List* list);

// String helper

gchar* concat_and_free_old(gchar* string1, gchar* string2);

// Helper

/**
* \brief Naive Matmul Helper (not so fast! O(n^3)). Computes mat1*mat2.
*
* \author jan eric lenssen
*
* \param mat1 First Matrix for matrix mul (Row-major)
* \param mat2 Second Matrix for mattrix mul (Row-major)
* \param[out] result Pointer to pre-allocated memory for result - Buffer needs to have size mat1rows*mat2cols
*/
gboolean dR_matmul(gfloat* mat1, int mat1rows, int mat1cols, gfloat* mat2, int mat2rows, int mat2cols, gfloat* result);


/**
* \brief Naive Matmul Helper (not so fast! O(n^3)). Computes mat1*mat2^T.
*
* \author jan eric lenssen
*
* \param mat1 First Matrix for matrix mul (Row-major)
* \param mat2 Second Matrix for mattrix mul (Row-major)
* \param[out] result Pointer to pre-allocated memory for result - Buffer needs to have size mat1rows*mat2cols
*/
gboolean dR_matmulT(gfloat* mat1, int mat1rows, int mat1cols, gfloat* mat2, int mat2rows, int mat2cols, gfloat* result);

#endif // DR_CORE_H
