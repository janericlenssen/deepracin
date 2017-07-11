#include "dR_core.h"
#include "dR_parser.h"
#include <assert.h>


dR_Graph* dR_appendLayer(dR_Graph* net, dR_Node* l){
    l->propagated = FALSE;
    l->output = FALSE;
    l->queued = FALSE;
    l->layerID = net->number_of_layers;
    if(l->type!=tDataFeedNode)
        l->outputBuf = NULL;
    dR_list_append(net->allNodes,l);
    net->number_of_layers++;
    return net;
}

dR_Node* dR_Datafeednode(dR_Graph* net, dR_Shape3 shape)
{
    dR_DataFeedNode_Data* datafeednode = g_malloc(sizeof(dR_DataFeedNode_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    datafeednode->shape = shape;

    l->layer = datafeednode;
    l->type = tDataFeedNode;
    l->outputBuf = NULL;
    l->next_layers = dR_list_createEmptyList();
    l->previous_layers = dR_list_createEmptyList();
    dR_list_append(net->feed_layers,l);

    l->printLayer = dR_datafeednode_printLayer;
    l->allocateBuffers = dR_datafeednode_allocateBuffers;
    l->getRequiredOutputBufferSize = dR_datafeednode_getRequiredOutputBufferSize;
    l->cleanupLayer = dR_datafeednode_cleanupLayer;
    l->serializeNode = dR_datafeednode_serializeNode;
    l->parseAppendNode = dR_datafeednode_parseAppendNode;

    dR_appendLayer(net,l);
    return l;
}

gchar* dR_datafeednode_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_DataFeedNode_Data* dfnode = (dR_DataFeedNode_Data*)(layer->layer);
    gchar* desc = "DataFeedNode";
    gint numNodeParams = 3;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("DataFeedNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",dfnode->shape.s0);
    params[1] = g_strdup_printf("%d",dfnode->shape.s1);
    params[2] = g_strdup_printf("%d",dfnode->shape.s2);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_datafeednode_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 0;
    gint numNodeParams = 3;
    gint numNodeVariables = 0;
    dR_Shape3 shape;
    if(numINodes!=0)
    {
        g_print("Parsing Error: DataFeedNode Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: DataFeedNode Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    shape.s0 = atoi(params[0]);
    shape.s1 = atoi(params[1]);
    shape.s2 = atoi(params[2]);
    return dR_Datafeednode(net, shape);
}

gchar* dR_datafeednode_printLayer(dR_Node* layer)
{
    dR_DataFeedNode_Data* dflayer = (dR_DataFeedNode_Data*)(layer->layer);
    gchar* out;
    out = g_strdup_printf("%s%d%s%d%s%d%s%d%s","DataFeedNode: ", layer->layerID,"\n Data dimensions: ", dflayer->shape.s0,"x",dflayer->shape.s1,"x",dflayer->shape.s2, "\n");
    return out;
}

gboolean dR_datafeednode_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    dR_DataFeedNode_Data* dfnode = (dR_DataFeedNode_Data*)(layer->layer);
    dR_Shape3 shape = dfnode->shape;
    return dR_createFloatBuffer(net,&(layer->outputBuf->buf),shape.s0*shape.s1*shape.s2,CL_MEM_READ_WRITE);
}

gint32 dR_datafeednode_getRequiredOutputBufferSize(dR_Node* layer)
{
    dR_DataFeedNode_Data* dfnode = (dR_DataFeedNode_Data*)(layer->layer);
    dR_Shape3 shape = dfnode->shape;
    return shape.s0*shape.s1*shape.s2;
}

gboolean dR_datafeednode_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
        g_free((dR_DataFeedNode_Data*)(layer->layer));
    return TRUE;
}

// Init Functionality

gboolean dR_propagateShapesAndSchedule(dR_Graph* net){

    dR_List* toPropagate = dR_list_createEmptyList();
    dR_Node* layer;
    // Layer Shapes
    dR_list_resetIt(net->feed_layers);
    if(net->feed_layers->length==0)
    {
        g_print("A minimum of one data feed node is required to prepare the net application!\n");
        return FALSE;
    }
    layer = (dR_Node*)dR_list_next(net->feed_layers);
    while(layer)
    {
        dR_DataFeedNode_Data* feednode = (dR_DataFeedNode_Data*)(layer->layer);
        dR_Node* next;
        layer->oshape.s0 = feednode->shape.s0;
        layer->oshape.s1 = feednode->shape.s1;
        layer->oshape.s2 = feednode->shape.s2;
        layer->propagated = TRUE;
        dR_list_resetIt(layer->next_layers);
        next = dR_list_next((layer->next_layers));
        while(next)
        {
            if(!next->queued)
            {
                dR_list_append(toPropagate,next);
                next->queued = TRUE;
            }
            next = dR_list_next((layer->next_layers));
        }
        layer = (dR_Node*)dR_list_next(net->feed_layers);
    }
    dR_list_resetIt(toPropagate);
    layer = dR_list_next(toPropagate);
    while(layer)
    {
        if(!layer->propagated)
        {
            gboolean requirementsMet = TRUE;
            dR_Node* prev;
            dR_list_resetIt(layer->previous_layers);
            prev = (dR_Node*)dR_list_next(layer->previous_layers);
            while(prev)
            {
                if(!prev->propagated)
                    requirementsMet = FALSE;
                prev = (dR_Node*)dR_list_next(layer->previous_layers);
            }
            if(!requirementsMet)
            {
                dR_list_removeFirstOcc(toPropagate,layer);
                dR_list_append(toPropagate,layer);
            }
            else
            {
                dR_Node* next;
                if(!layer->propagateShape(net,layer))
                {
                    g_print("Propagate Shapes failed for Layer %d!", layer->layerID);
                    return FALSE;
                }
                if(net->config->debugInfo&&!net->config->silent)
                    g_print("Propagated layer %d: %dx%dx%d\n",
                        layer->layerID,layer->oshape.s0,layer->oshape.s1,layer->oshape.s2);
                if(!layer->schedule(net, layer))
                {
                    g_print("Scheduling failed for Layer %d! \n",layer->layerID);
                    return FALSE;
                }
                dR_list_resetIt(layer->next_layers);
                next = (dR_Node*)dR_list_next((layer->next_layers));
                while(next)
                {
                    if(!next->queued)
                    {
                        dR_list_append(toPropagate,next);
                        next->queued = TRUE;
                    }
                    next = (dR_Node*)dR_list_next((layer->next_layers));
                }
                layer->propagated = TRUE;
            }
        }
        layer = dR_list_next(toPropagate);
    }
    net->scheduledLayers = toPropagate;
    if(net->config->debugInfo&&!net->config->silent)
        dR_printSchedule(net);
    return TRUE;
}

gboolean dR_estimateAndAllocateBuffers(dR_Graph* net){
	
    dR_List* memoryBufList = dR_list_createEmptyList();
    dR_List* freeMemoryBufs = dR_list_createEmptyList();
    dR_Node* current_layer;
    gint maxBufferSize;
    dR_MemoryHandler* memhandle;
    gboolean ret = TRUE;

    // Allocate Buffers for feed nodes
    dR_list_resetIt(net->feed_layers);
    current_layer = (dR_Node*)dR_list_next(net->feed_layers);
    while(current_layer){
        if(current_layer->outputBuf==NULL)
        {
            gint reqsize;
            dR_MemoryHandler* chosenbuffer;
            chosenbuffer = dR_newMemoryHandler(FALSE);
            dR_list_append(memoryBufList,chosenbuffer);
            current_layer->outputBuf = chosenbuffer;
            chosenbuffer->numberOfDependencies = current_layer->next_layers->length;
            dR_list_append(chosenbuffer->outputFor,current_layer);
            reqsize = current_layer->getRequiredOutputBufferSize(current_layer);
            if(chosenbuffer->size<reqsize)
                chosenbuffer->size = reqsize;
            if(!current_layer->allocateBuffers(net,current_layer))
            {
                g_print("OpenCL Buffer Allocation failed for Layer %d! \n",current_layer->layerID);
                return FALSE;
            }
            if(!net->config->silent&&net->config->debugInfo)
                g_print("Created Buffers for Layer %d\n\n",current_layer->layerID);
        }
        current_layer = (dR_Node*)dR_list_next(net->feed_layers);
    }

    // Assign Buffers to Layers and create individual layer buffers
    dR_list_resetIt(net->scheduledLayers);
    current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    while(current_layer){
        if(!current_layer->outputBuf)
        {
			gint reqsize;
            dR_Node* prev;
            dR_MemoryHandler* chosenbuffer;
            if(freeMemoryBufs->length==0)
            {
                chosenbuffer = dR_newMemoryHandler(FALSE);
                dR_list_append(memoryBufList,chosenbuffer);
            }
            else
            {
                chosenbuffer = (dR_MemoryHandler*)dR_list_pop(freeMemoryBufs);
            }
            current_layer->outputBuf = chosenbuffer;
            chosenbuffer->numberOfDependencies = current_layer->next_layers->length;
            dR_list_append(chosenbuffer->outputFor,current_layer);
            reqsize = current_layer->getRequiredOutputBufferSize(current_layer);
            if(chosenbuffer->size<reqsize)
                chosenbuffer->size = reqsize;

            dR_list_resetIt(current_layer->previous_layers);
            prev = (dR_Node*)dR_list_next(current_layer->previous_layers);
            while(prev)
            {
                if(prev->type!=tDataFeedNode&&!prev->output)
                {
                    prev->outputBuf->numberOfDependencies--;
                    if(prev->outputBuf->numberOfDependencies==0)
                    {
                        dR_list_append(freeMemoryBufs,prev->outputBuf);
                    }
                }
                prev = (dR_Node*)dR_list_next(current_layer->previous_layers);
            }
        }

        if(!net->config->silent&&net->config->debugInfo)
            g_print("OpenCL Buffer Allocation for Layer %d... \n",current_layer->layerID);

        if(!current_layer->allocateBuffers(net,current_layer))
        {
            g_print("OpenCL Buffer Allocation failed for Layer %d! \n",current_layer->layerID);
            return FALSE;
        }
        if(!net->config->silent&&net->config->debugInfo)
            g_print("Created Buffers for Layer! %d\n",current_layer->layerID);
        current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    }
    if(!net->config->silent&&net->config->debugInfo)
        g_print("Number of required stream buffers: %d, Allocating...\n",memoryBufList->length);
    net->streamBufferHandlers = memoryBufList;

    dR_list_resetIt(memoryBufList);
    memhandle = (dR_MemoryHandler*)dR_list_next(memoryBufList);
    maxBufferSize = 0;
    while(memhandle)
    {
        if(memhandle->size>maxBufferSize)
            maxBufferSize = memhandle->size;
        memhandle->bufptr = &memhandle->buf;
        ret &= dR_createFloatBuffer(net, memhandle->bufptr, memhandle->size, CL_MEM_READ_WRITE);
        if(!net->config->silent&&net->config->debugInfo)
            g_print("Stream buffer with size %d created!\n",memhandle->size);
        memhandle = (dR_MemoryHandler*)dR_list_next(memoryBufList);
    }
    if(!net->config->silent&&net->config->debugInfo)
        g_print("Stream buffers allocated!\n");

    net->hostDebugBuffer = g_malloc(maxBufferSize*4);

    return ret;
}


gboolean dR_fillBuffers(dR_Graph* net){
    dR_Node* current_layer;
    dR_list_resetIt(net->scheduledLayers);
    current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    while(current_layer){
        if(!current_layer->fillBuffers(net,current_layer))
        {
            g_print("OpenCL Buffer filling failed for Layer %d! \n",current_layer->layerID);
            return FALSE;
        }
        current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    }
    return TRUE;
}


dR_MemoryHandler* dR_newMemoryHandler(gboolean cust)
{
    dR_MemoryHandler* handle = g_malloc(sizeof(dR_MemoryHandler));
    handle->outputFor = dR_list_createEmptyList();
    handle->size = 0;
    handle->numberOfDependencies = 0;
    handle->outputBuf = cust;
    handle->bufptr = &handle->buf;
    handle->regionOfInterest = FALSE;

    return handle;
}

void dR_cleanupMemoryHandler(dR_Graph* net, dR_MemoryHandler* handler)
{
    if(handler)
    {
        dR_clMemoryBufferCleanup(net, handler->buf);
        dR_list_cleanup(handler->outputFor);
        g_free(handler);
        handler = NULL;
    }
}


// Layer Scheduling Functionality

gboolean dR_generateAndCompileKernels(dR_Graph* net)
{
    const gchar * const filenames[] = {
        "dR_helper.cl",
        "dR_fcl.cl",
        "dR_math.cl",
        "dR_transform.cl",
        "dR_winograd2.cl",
        "dR_conv2d1x1.cl"
    };
    gint numberoffiles = 6;
    gboolean ret;
    dR_Node* current_layer;
    dR_list_resetIt(net->scheduledLayers);
    current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    while(current_layer){
        if(current_layer->generateKernel)
        {
            if(!current_layer->generateKernel(net,current_layer))
            {
                g_print("Kernel Generation failed for Layer %d! \n", current_layer->layerID);
                return FALSE;
            }
        }
        if(!net->config->silent&&net->config->debugInfo)
            g_print("Created Kernel for layer %d \n",current_layer->layerID);
        current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    }

    if(!net->config->silent)
        g_print("Done.\n");

    ret = dR_clBuildWithSource(net, filenames, numberoffiles);
    if(!ret){
        g_print("deepRACIN Program Compile failed\n");
    }
    dR_list_resetIt(net->scheduledLayers);
    current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    while(current_layer){
        if(!current_layer->createKernel(net,current_layer))
        {
            g_print("OpenCL Kernel Creation failed for Layer %d! \n",current_layer->layerID);
            return FALSE;
        }
        current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    }
    return TRUE;
}


void dR_cleanupBuffers(dR_Graph* net)
{
    dR_Node* current_layer;
    //cleanup the two main swap buffers
    gboolean ret = TRUE;
    dR_MemoryHandler* memhandle;
    dR_list_resetIt(net->streamBufferHandlers);
    memhandle = (dR_MemoryHandler*)dR_list_next(net->streamBufferHandlers);
    while(memhandle){
        if(!dR_clMemoryBufferCleanup(net, memhandle->buf))
        {
            g_print("OpenCL Buffer and Kernel Cleanup failed stream buffer \n");
        }
        memhandle = (dR_MemoryHandler*)dR_list_next(net->streamBufferHandlers);
    }
    if(!ret)
    {
        g_print("OpenCL swap buffer cleanup failed! \n");
    }

    //cleanup individual layer buffers
    dR_list_resetIt(net->scheduledLayers);
    current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    while(current_layer){
        if(!current_layer->cleanupBuffers(net,current_layer))
        {
            g_print("OpenCL Buffer and Kernel Cleanup failed for Layer %d! \n",current_layer->layerID);
        }
        current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    }

    if(!dR_cleanupProgram(net->clConfig->clProgram))
    {
        g_print("OpenCL program cleanup failed! \n");
    }

    return;
}


void dR_cleanupNet(dR_Graph* net)
{
    dR_Node* current_layer;
    dR_Node* temp;
    dR_MemoryHandler* memhandle;
    dR_list_resetIt(net->streamBufferHandlers);
    memhandle = (dR_MemoryHandler*)dR_list_next(net->streamBufferHandlers);

    while(memhandle)
    {
        g_free(memhandle);
        memhandle = (dR_MemoryHandler*)dR_list_next(net->streamBufferHandlers);
    }

    dR_list_resetIt(net->scheduledLayers);
    current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    while(current_layer)
    {
        if(!current_layer->cleanupLayer(net,current_layer))
        {
            g_print("Layer cleanup failed for layer %d! \n",current_layer->layerID);
        }
		g_free(current_layer);
        current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    }

    dR_list_resetIt(net->feed_layers);
    temp = (dR_Node*)dR_list_next(net->feed_layers);
    while(temp)
    {
        if(!temp->cleanupLayer(net,temp))
        {
            g_print("Layer cleanup failed for layer %d! \n",temp->layerID);
        }
        g_free(temp);
        temp = (dR_Node*)dR_list_next(net->feed_layers);
    }

    dR_list_cleanup(net->scheduledLayers);
    dR_list_cleanup(net->streamBufferHandlers);
    dR_list_cleanup(net->output_layers);
    dR_list_cleanup(net->feed_layers);
    g_free(net->config);
    g_free(net->clConfig);
    g_free(net->hostDebugBuffer);
    g_free(net);
}


// List Functionality //

dR_List* dR_list_createEmptyList()
{
    dR_List* a = g_malloc(sizeof(dR_List));

    a->first = NULL;
    a->last = NULL;
    a->length=0;
    a->iterator = NULL;
    return a;
}

void dR_list_append(dR_List* list, void* layer)
{
    dR_ListElement* elem = g_malloc(sizeof(dR_ListElement));
    elem->element = layer;
    if(!list->first){
        list->first = elem;
        list->last = elem;
        elem->next = NULL;
        elem->previous = NULL;
    } else {
        dR_ListElement* current_last;
        current_last = list->last;
        current_last->next = elem;
        elem->previous = current_last;
        list->last = elem;
        elem->next = NULL;
    }
    list->length++;
}

gboolean dR_list_removeFirstOcc(dR_List* list, void* layer)
{
    dR_ListElement* temp = list->first;
    while(temp)
    {
        if(temp->element == layer)
        {
            if(list->length==1)
            {
                list->first=NULL;
                list->last=NULL;
                list->iterator=NULL;
                g_free(temp);
            }
            else if(list->length>1)
            {
                if(temp->previous==NULL)
                {
                    list->first = temp->next;
                    temp->next->previous = NULL;
                }
                else if(temp->next==NULL)
                {
                    list->last = temp->previous;
                    temp->previous->next=NULL;
                }
                else
                {
                    temp->next->previous = temp->previous;
                    temp->previous->next = temp->next;
                }
                if(list->iterator==temp)
                {
                    if(temp->previous)
                    {
                        list->iterator=temp->previous;
                    }
                    else
                    {
                        list->iterator=NULL;
                    }
                }
                g_free(temp);
            }
            list->length--;
            return TRUE;
        }
        temp = temp->next;
    }
    return FALSE;
}

void* dR_list_pop(dR_List* list)
{
	void* ret;
    dR_ListElement* retelem = list->first;
    if(list->length==0)
    {
        return NULL;
    }
    else if(list->length==1)
    {
        list->first = NULL;
        list->last = NULL;
        list->iterator = NULL;
    }
    else
    {
        list->first = list->first->next;
        list->first->previous = NULL;
    }
    ret = retelem->element;
    if(list->iterator==retelem)
    {
        list->iterator=retelem->next;
    }
    g_free(retelem);
    list->length--;
    return ret;
}


void* dR_list_next(dR_List* list)
{
    void* ret;
    if(list->iterator!=NULL)
    {
        if(list->iterator->next == NULL)
        {
            list->iterator = NULL;
            ret = NULL;
        }
        else
        {
            list->iterator = list->iterator->next;
            ret = list->iterator->element;
        }
    }
    else
    {
        list->iterator = list->first;
        if(list->iterator!=NULL)
        {
            ret = list->iterator->element;
        }
        else
        {
            ret = NULL;
        }
    }
    return ret;
}


void dR_list_resetIt(dR_List* list)
{
    list->iterator = NULL;
}

void dR_list_cleanup(dR_List* list)
{
    dR_ListElement* it = list->first;
    dR_ListElement* temp;
    while(it) //@todo FIXME? Muss hier evtl. "it" statt "temp" verwendet werden?
    {
        temp = it;
        it = it->next;
        g_free(temp);
    }
    g_free(list);
}

gchar* concat_and_free_old(gchar* string1, gchar* string2)
{
    gchar* concat_string;
    concat_string = g_strjoin (NULL, string1, string2, NULL);
    if(string1)
        g_free(string1);
    g_free(string2);
    return concat_string;
}

gboolean dR_matmul(gfloat* mat1, gint mat1rows, gint mat1cols, gfloat* mat2, gint mat2rows, gint mat2cols, gfloat* result)
{
    gint i,j,k;
    gfloat sum;
    if(mat1cols!=mat2rows)
    {
        g_print("Matmul failed: Dimensions not matching!\n");
        return FALSE;
    }
    for(i = 0; i<mat1rows;i++)
    {
        for(j=0;j<mat2cols;j++)
        {
            sum = 0.0;
            for(k = 0; k<mat1cols;k++)
            {
                sum += mat1[i*mat1cols+k]*mat2[k*mat2rows+j];
            }
            result[i*mat2cols+j] = sum;
        }
    }
    return TRUE;
}

gboolean dR_matmulT(gfloat* mat1, gint mat1rows, gint mat1cols, gfloat* mat2, gint mat2rows, gint mat2cols, gfloat* result)
{
    gint i,j,k;
    gfloat sum;
    if(mat1cols!=mat2cols)
    {
        g_print("MatmulT failed: Dimensions not matching!\n");
        return FALSE;
    }
    for(i = 0; i<mat1rows;i++)
    {
        for(j=0;j<mat2rows;j++)
        {
            sum = 0.0;
            for(k = 0; k<mat1cols;k++)
            {
                sum += mat1[i*mat1cols+k]*mat2[j*mat2cols+k];
            }
            result[i*mat2rows+j] = sum;
        }
    }
    return TRUE;
}
