#include "deepRACIN.h"
#include "dR_parser.h"
#include <glib.h>
#include <assert.h>

dR_Graph* dR_NewGraph(){
    dR_Graph* a = g_malloc(sizeof(dR_Graph));
    dR_ClConfig* clConfig = g_malloc(sizeof(dR_ClConfig));
    dR_Config* config = g_malloc(sizeof(dR_Config));

    a->clConfig = clConfig;
    // Init clConfig
    a->clConfig->clDeviceName = NULL;
    a->clConfig->cleanupDone = FALSE;
    a->clConfig->clEvent = &a->clConfig->clEventObject;
    a->clConfig->clContext = NULL;
    a->clConfig->clDeviceType = CL_DEVICE_TYPE_GPU;
    a->clConfig->clDeviceNumber = 0;
    a->clConfig->clInfo = TRUE;
    a->clConfig->clKernelPath = KERNEL_PATH;
    a->clConfig->clPlatformName = " ";
    a->clConfig->configH = NULL;
    a->clConfig->fastRelaxedMath = TRUE;
    a->clConfig->forceRecompile = TRUE;
    a->clConfig->maxTotalKernelGPUTime = 0;
    a->clConfig->queueKernelGPUTime = 0;
    a->clConfig->sumTotalKernelGPUTime = 0;
    a->clConfig->totalKernelGPUTime = 0;
    a->clConfig->memConsumptionGPU = 0;
    a->config = config;

    // Init config
    a->config->convLocalWorkSizeWindow = 0;
    a->config->profilingCPU = FALSE;
    a->config->profilingGPU = FALSE;
    a->config->silent = TRUE;
    a->config->debugInfo = FALSE;
    a->config->totalNodeCPUTime = 0.0;

    a->feed_layers = dR_list_createEmptyList();
    a->output_layers = dR_list_createEmptyList();
    a->allNodes = dR_list_createEmptyList();

    a->prepared = FALSE;
    a->number_of_layers=0;
    return a;
}



gint dR_getOutputBuffers(dR_Graph* net, cl_mem** outptr)
{
    int i = 0;
    if(net->prepared)
    {
		dR_Node* node;
        dR_list_resetIt(net->output_layers);
        node = (dR_Node*)dR_list_next(net->output_layers);
        while(node&&i<10)
        {
            outptr[i] = node->outputBuf->bufptr;
            i++;
            node = (dR_Node*)dR_list_next(net->output_layers);
        }
    }
    else
    {
        g_print("dR_Graph must be prepared before having output buffers. Call dR_prepare first! \n");
        return 0;
    }
    return i;
}

void dR_feedData(dR_Graph* net, dR_Node* feednode, gfloat* data, gint offset, gint numfloats)
{
    dR_uploadArray(net,"",data,offset*sizeof(cl_float),numfloats*sizeof(cl_float),feednode->outputBuf->buf);
}

gboolean dR_setDataFeedNodeBuffer(dR_Graph* net, dR_Node* node, cl_mem* buf)
{

    dR_DataFeedNode_Data* feednode;
    if(node->type != tDataFeedNode)
    {
        if(!net->config->silent)
        {
            g_print("dR_setDataFeedNodeBuffer Error: Data Feed Buffer can only be set to a DataFeedNode!\n");
        }
        return FALSE;
    }
    feednode = (dR_DataFeedNode_Data*)(node->layer);
    //dR_cleanupMemoryHandler(net,node->outputBuf);
    node->outputBuf = dR_newMemoryHandler(TRUE);
    node->outputBuf->bufptr = buf;
    node->outputBuf->buf = *buf;
    node->outputBuf->size = feednode->shape.s0*feednode->shape.s1*feednode->shape.s2;
    dR_list_append(node->outputBuf->outputFor,node);
    return TRUE;
}


gboolean dR_setNodeRoI(dR_Node* node, dR_Shape3 origin)
{
    node->outputBuf->regionOfInterest = TRUE;
    node->outputBuf->useIndexBufferForRoI = FALSE;
    node->outputBuf->roiOrigin=origin;

    return TRUE;
}

gboolean dR_setNodeRoIBufferAndIndex(dR_Node* node, cl_mem* buf, gint index)
{
    node->outputBuf->regionOfInterest = TRUE;
    node->outputBuf->useIndexBufferForRoI = TRUE;
    node->outputBuf->index=index;
    node->outputBuf->indexBuffer=buf;

    return TRUE;
}


gboolean dR_setPreexistingOutputBuffer(dR_Graph* net, dR_Node* node, cl_mem* buf)
{
    node->outputBuf = dR_newMemoryHandler(TRUE);
    node->outputBuf->bufptr = buf;
    if(!node->output)
        dR_setAsOutput(net, node);
    return TRUE;
}



gboolean dR_setAsOutput(dR_Graph* net, dR_Node* node)
{
    if(!node->output)
    {
        node->output = TRUE;
        dR_list_append(net->output_layers,node);
    }
    return TRUE;
}


void dR_initCL(dR_Graph* net){
    dR_clInit(net);
}

gboolean dR_prepare(dR_Graph* net){
    gboolean ret;
    if(!net->config->silent)
        g_print("Propagate shapes and schedule...\n");
    ret = dR_propagateShapesAndSchedule(net);
    if(!ret)
    {
        g_print("dR_propagateShapesAndSchedule Failed \n");
        return FALSE;
    }
    if(!net->config->silent)
        g_print("Done.\n");

    if(!net->config->silent)
        g_print("Generate and compile kernels...\n");
    ret = dR_generateAndCompileKernels(net);
    if(!ret)
    {
        g_print("\ndR_generateAndCompileKernels Failed \n");
        return FALSE;
    }
    if(!net->config->silent)
        g_print("Done.\n");

    if(!net->config->silent)
        g_print("Estimate and allocate buffers...\n");
    ret = dR_estimateAndAllocateBuffers(net);
    if(!ret)
    {
        g_print("dR_propagateShapes Failed \n");
        return FALSE;
    }
    if(!net->config->silent)
        g_print("Done. \n");


    if(!net->config->silent)
        g_print("Fill buffers...\n");
    ret = dR_fillBuffers(net);
    if(!ret)
    {
        g_print("dR_fillBuffers Failed \n");
        return FALSE;
    }
    if(!net->config->silent)
        g_print("Done.\n\n");

    net->prepared = TRUE;
    if(net->config->profilingGPU)
    {
        g_print("\nGPU Profiling: Whole graph GPU memory consumption: %4.6f MB \n\n", (double)net->clConfig->memConsumptionGPU/1048576.0);
        net->clConfig->totalKernelGPUTime = 0;
    }
    return TRUE;
}

gboolean dR_apply(dR_Graph* net){
    dR_Node* current_layer;
	GTimeVal result;
    gint64 gstarttime = 0;
    gint64 nstarttime = 0;
    if(!net->config->silent)
        g_print("Applying graph...\n");
    dR_list_resetIt(net->scheduledLayers);
    current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    if(net->config->profilingCPU)
    {
		GTimeVal result;
		g_get_current_time (&result);
		gstarttime = result.tv_usec;
        net->config->totalNodeCPUTime = 0.0;
    }
    while(current_layer){
        if(net->config->profilingCPU)
        {
			g_get_current_time (&result);
            nstarttime = result.tv_usec;
        }
        if(!current_layer->compute(net, current_layer))
        {
            g_print("Execution of layer %d failed \n",current_layer->layerID);
            return FALSE;
        }
        if(current_layer->layerID==net->debugLayer)
        {
            dR_downloadArray(net, "",current_layer->outputBuf->bufptr,0*current_layer->oshape.s0*current_layer->oshape.s1*sizeof(cl_float),current_layer->oshape.s0*current_layer->oshape.s1*sizeof(cl_float), net->hostDebugBuffer);
        }
        if(net->config->profilingCPU)
        {
			gdouble noderuntime;
            clFinish(net->clConfig->clCommandQueue);
			g_get_current_time (&result);
			noderuntime = (gdouble)(result.tv_usec - nstarttime)/1000.0;
            net->config->totalNodeCPUTime+=noderuntime;
            g_print("CPU Profiling: Node %d took: %2.3fms \n",current_layer->layerID, noderuntime);
        }
        current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    }
    if(net->config->profilingGPU)
    {
        g_print("GPU Profiling: Sum of all kernel run times: %2.3fms \n", net->clConfig->totalKernelGPUTime);
        net->clConfig->totalKernelGPUTime = 0;
    }
    if(net->config->profilingCPU)
    {
		gdouble graphruntime;
		g_get_current_time (&result);
		graphruntime = (gdouble)(result.tv_usec - gstarttime) / 1000.0;
        g_print("CPU Profiling: Whole graph took: %2.3fms, Sum of all nodes: %2.3fms \n", graphruntime,net->config->totalNodeCPUTime);
    }
    if(!net->config->silent)
        g_print("Done.\n\n");
    return TRUE;
}

void dR_cleanup(dR_Graph* net, gboolean cleanupcl){
    gboolean silent = net->config->silent;
    if(!silent)
        g_print("Cleaning up deepRACIN...\n");
    dR_cleanupBuffers(net);
    if(cleanupcl)
        dR_cleanupCL(net);
    dR_cleanupNet(net);
    if(!silent)
        g_print("Done.\n");
}


dR_Node* dR_loadModelFile(dR_Graph* net, dR_Node* input, gchar* path, dR_Node** nodelist, gint* numnodes){
    return dR_parseModel(net, input, path, nodelist, numnodes);
}



gboolean dR_saveGraph(dR_Graph* net, gchar* path)
{
    return dR_serializeGraph(net,path);
}


dR_Node* dR_loadGraph(dR_Graph* net, gchar* path, dR_Node*** nodelist, gint* numnodes, dR_Node*** feednodes, gint* numfeednodes)
{
    return dR_parseGraph(net, path, nodelist, numnodes, feednodes, numfeednodes);
}



void dR_printNetObject(dR_Graph* net, char* path){
    dR_printNet(net,path);
}

void dR_setClEnvironment(dR_Graph* net,
                                cl_context clcontext,
                                cl_platform_id clplatformid,
                                cl_command_queue clcommandqueue,
                                cl_device_id cldeviceid)
{
    net->clConfig->clCommandQueue = clcommandqueue;
    net->clConfig->clContext = clcontext;
    net->clConfig->clDeviceId = cldeviceid;
    net->clConfig->clPlatformId = clplatformid;
}

void dR_config(dR_Graph* net,
                                gchar* platformname,
                                gboolean silent,
                                gboolean debuginfo,
                                gboolean profileGPU,
                                gboolean profileCPU,
                                gchar* temppath)
{
    net->clConfig->clPlatformName = g_strdup(platformname);
    net->config->silent = silent;
    net->config->debugInfo = debuginfo;
    net->config->profilingCPU = profileCPU;
    net->config->profilingGPU = profileGPU;
    net->config->modelPath = temppath;
}


dR_Shape3* dR_getOutputShape(dR_Node* node)
{
    return &node->oshape;
}
