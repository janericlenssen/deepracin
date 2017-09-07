#ifndef DR_H
#define DR_H

#include "dR_core.h"

// CNN Model Application Functionality

#ifdef __cplusplus
extern "C"{
#endif
/**
* \brief Creates and returns an empty network to which layers can be appended.
*
* \author jan eric lenssen
*
* \returns The empty dR_Graph object pointer
*/
dR_Graph* dR_NewGraph(void);


/**
* \brief Creates and returns OpenCL buffers with proper size for the graph outputs which were defined by dR_setAsOutput(...). dR_Graph has to be prepared by dR_prepare beforehand.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[out] outptr A pointer to empty memory where OpenCL buffer pointers are stored by the functions. Size needs to match the defined number of output buffers of the graph.
*
* \returns The number of returned output buffers

*/
gint dR_getOutputBuffers(dR_Graph* graph, cl_mem** outptr);


/**
* \brief Feeds data to a data feed node. dR_Graph has to be prepared by dR_prepare beforehand.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] feednode A pointer to the node to which the data should be fed
* \param[in] data A pointer to a host float buffer containing data that will be pushed to the feednode
* \param[in] offset The offset in the data buffer from which data will be pushed to the feednode
* \param[in] numfloats The number of floats that will be pushed to the feednode

*/
void dR_feedData(dR_Graph* graph, dR_Node* feednode, gfloat* data, gint offset, gint numfloats);

/**
* \brief Gives the input shape of a feed node.
*
* \author jan eric lenssen
*
* \param[in] node The feed node whose output shape is requested.

*/
dR_Shape3* dR_getFeedNodeInputShape(dR_Node* node);

/**
* \brief Sets an existing OpenCL buffer as source for a data feed node. Needs to be called before graph preparation.
* \details Assumes that data lies in [z, y, x] (z major, x minor) order in the sequential float buffer.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] feednode The feednode object pointer for which the OpenCL buffer will be the source.
* \param[in] clbuf A pointer to the existing OpenCL buffer.
*
* \returns True for success, False for failure

*/
gboolean dR_setDataFeedNodeBuffer(dR_Graph* graph, dR_Node* feednode, cl_mem* clbuf);


/**
* \brief Sets a region of interest origin for the given node. The region of interest can be extracted by a resolveRoI node in the graph.
* \details The size of the region of interest needs to be defined at graph definition time by the resolveRoI node. The origin however, can be set by this function
* between graph executions in order to continue processing on different parts of the input buffer.
* Assumes that data lies in [z, y, x] (z major, x minor) order in the sequential buffer.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] node The node object pointer for which the region of interest is defined.
* \param[in] originX x coordinate of the roi origin in the buffer.
* \param[in] originY y coordinate of the roi origin in the buffer.
* \param[in] originZ z coordinate of the roi origin in the buffer.
*
* \returns True for success, False for failure

*/
gboolean dR_setNodeRoI(dR_Node* node, dR_Shape3 origin);


/**
* \brief Sets an OpenCL buffer from which the region of interest origin for the given node is fetched. The region of interest can be extracted by a resolveRoI node in the graph.
* \details The size of the region of interest needs to be defined at graph definition time by the resolveRoI node. The source of the origin however, can be set by this function
* between graph executions in order to continue processing on different parts of the input buffer.
* In comparison to dR_setNodeRoI, this function can be used if the origin coordinates lie in an OpenCL Buffer on the GPU in order to avoid the GPU->CPU->GPU cycle of the data.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] node The node object pointer for which the region of interest is defined.
* \param[in] buf A Pointer to the OpenCL buffer objects in which the origin coordinates lie.
* \param[in] index An offset within this buffer that points to the position of the origin coordinates.
*
* \returns True for success, False for failure

*/
gboolean dR_setNodeRoIBufferAndIndex(dR_Node* node, cl_mem* buf, gint index);

/**
* \brief Sets a preexisting OpenCL Buffer as buffer for an output node.  Needs to be called before graph preparation.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] node The node object pointer for which the region of interest is defined.
* \param[in] buf A Pointer to the OpenCL buffer objects in which the origin coordinates lie.
* \param[in] index An offset within this buffer that points to the position of the origin coordinates.
*
* \returns True for success, False for failure

*/
gboolean dR_setPreexistingOutputBuffer(dR_Graph* graph, dR_Node* node, cl_mem* buf);

/**
* \brief Marks a graph node as an output node. Needs to be called before graph preparation.
* \details For output nodes, it is guaranteed that the buffers are not reused for other parts of the graph so that after graph execution the data of this node is available.
* Having more output nodes than one will most likely increase the number of required OpenCL buffers and therefore the GPU Memory consumption.
* The buffers of all output nodes can be fetched by dR_getOutputBuffers after graph preparation
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] node A pointer to the node object that should be marked as output node.
*
* \returns True for success, False for failure

*/
gboolean dR_setAsOutput(dR_Graph* graph, dR_Node* node);


/**
* \brief Initializes OpenCL with default Parameters. Only required if no existing OpenCL context should be used. 
* \details Parameters can be changes afterwards by accessing the graph->clConfig struct.
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
*
* \returns True for success, False for failure

*/
void dR_initCL(dR_Graph* graph);



/**
* \brief Applies the given graph. dR_Graph has to be prepared by dR_prepareCNNApplication beforehand and all data feed nodes have to have data.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
*
* \returns True for success, False for failure

*/
gboolean dR_apply(dR_Graph* graph);


/**
* \brief Prepares Buffers and Kernels on GPU for CNN model application
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
*
* \returns True for success, False for failure

*/
gboolean dR_prepare(dR_Graph* net);


/**
* \brief Cleans up the net object and everything that was setup by dR_prepare
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] cleanupcl If true, OpenCL is cleaned up as well. For applications that use OpenCL only for deepRACIN and initialized OpenCL with dR_initCL.

*/
void dR_cleanup(dR_Graph* net, gboolean cleanupcl);



/**
* \brief Loads a convolutional neural network model file from disc, parses it and appends the subgraph to the given input node
* \details For more information about the expected file format, take a look at the format readme.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] path the Path to the TensorFlow model file.
*
* \returns The last node of the appended subgraph

*/
//dR_Node* dR_loadModelFile(dR_Graph* net, dR_Node* input, gchar* path, dR_Node** nodelist, gint* numnodes);


gboolean dR_saveGraph(dR_Graph* net, gchar* path);


dR_Node* dR_loadGraph(dR_Graph* net, gchar* path, dR_Node*** nodelist, gint* numnodes, dR_Node*** feednodes, gint* numfeednodes);

/**
* \brief Prints the net to Console or file
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] path Path to file where the net should be printed. If NULL, it will be printed to console.

*/
void dR_printNetObject(dR_Graph* net, char* path);


/**
* \brief Gives a custom OpenCL context to the net object. Function must be called, if an existing OpenCL context should be used.
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] clcontext The OpenCL context
* \param[in] cl_platform_id The OpenCL Platform ID
* \param[in] cl_command_queue The OpenCL Command Queue
* \param[in] cl_device_id The OpenCL Device ID

*/
void dR_setClEnvironment(dR_Graph* net,
                                cl_context clcontext,
                                cl_platform_id clplatformid,
                                cl_command_queue clcommandqueue,
                                cl_device_id cldeviceid);


/**
* \brief Configures the graph object
*
* \author jan eric lenssen
*
* \param[in] graph The dR_Graph object pointer.
* \param[in] platformname Gives a hint about the prefered platform (Nvidia, AMD, Intel,...). Required if different platforms are available.
* \param[in] silent If true, silents all outputs
* \param[in] debuginfo if true, more debug information is printed
* \param[in] profileGPU If true, GPU profiling is enabled
* \param[in] profileCPU If true, CPU profiling is enabled

*/
void dR_config(dR_Graph* net,
                                gchar* platformname,
                                gboolean silent,
                                gboolean debuginfo,
                                gboolean profileGPU,
                                gboolean profileCPU,
                                gchar* temppath);


/**
* \brief Gives the output shape of a node. Only works after calling dR_prepare() on the associated graph.
*
* \author jan eric lenssen
*
* \param[in] node The node whose output shape is requested.

*/
dR_Shape3* dR_getOutputShape(dR_Node* node);




#ifdef __cplusplus
}
#endif

#endif // DR_H



