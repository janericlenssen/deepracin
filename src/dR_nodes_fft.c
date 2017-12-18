#include "dR_nodes_fft.h"
#include "dR_core.h"

// ///////////////////////////
// Fast Fourier Transform //
// ///////////////////////////

dR_Node* dR_FFT(dR_Graph* net, dR_Node* inputNode1)
{
    // allocate memory for dR_Shape3 (3 dimensional vector)
    dR_FFT_Data* fft = g_malloc(sizeof(dR_FFT_Data));
    // allocate memory for a node
    dR_Node* l = g_malloc(sizeof(dR_Node));

    // set all attributes of fft node
    // dR_Shape3
    l->layer = fft;
    // node type
    l->type = tFFT;
    // set functions (implemented in this file) for this node
    l->compute = dR_fft_compute;
    l->schedule = dR_fft_schedule;
    l->propagateShape = dR_fft_propagateShape;
    l->getRequiredOutputBufferSize = dR_fft_getRequiredOutputBufferSize;
    l->createKernel = dR_fft_createKernel;
    l->allocateBuffers = dR_fft_allocateBuffers;
    l->fillBuffers = dR_fft_fillBuffers;
    l->cleanupBuffers = dR_fft_cleanupBuffers;
    l->cleanupLayer = dR_fft_cleanupLayer;
    l->serializeNode = dR_fft_serializeNode;
    l->parseAppendNode = dR_fft_parseAppendNode;

    l->generateKernel = NULL;
    l->createKernelName = NULL;
    l->setVariables = NULL;
    l->printLayer = dR_fft_printLayer;

    if (inputNode1)
    {
      // create empty list for previous nodes
      l->previous_layers = dR_list_createEmptyList();
      // append the input of this node to the list
      dR_list_append(l->previous_layers,inputNode1);
      // create empty list for following nodes
      l->next_layers = dR_list_createEmptyList();
      // append the current (fft) node as the following node of the previous node
      dR_list_append(inputNode1->next_layers,l);
    }
    else
    {
        g_print("Error: FFT node needs an appropriate Inputnode");
    }
    // append node to graph
    dR_appendLayer(net, l);
    // return pointer to node
    return l;
}

gchar* dR_fft_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_FFT_Data* fft = (dR_FFT_Data*)(layer->layer);
    gchar* desc = "fft";
    gint numNodeParams = 1;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    //params[0] = g_strdup_printf("%d",fft->op);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_fft_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 1;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: fft Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: fft Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    out = dR_FFT(net, iNodes[0]);
    return out;
}

gboolean dR_fft_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }


gboolean dR_fft_compute(dR_Graph* net, dR_Node* layer){

    // call compute functionality
    // set kernel parameters and enqueue kernels

    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input1;
    dR_list_resetIt(layer->previous_layers);
    input1 = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;

    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input1);                      paramid++;

    /* for intermediate Buffer */
    dR_FFT_Data* fft = ((dR_FFT_Data*)(layer->layer));
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)fft->intermedBuf);          paramid++;

    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);          paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "FFT Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:fft");

}

gboolean dR_fft_propagateShape(dR_Graph* net, dR_Node* layer)
{
    // compute output shape of node
    // output of fft is complex number with re + im
    // input for fft is a picture (2D array) which can have a few levels (determined by dR_Shape)
    dR_FFT_Data* fft = (dR_FFT_Data*)(layer->layer);
    dR_Node* lastlayer;
    // check if previous layer gives correct output for fft
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("FFT Node with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
        return FALSE;
    }
    dR_list_resetIt(layer->previous_layers); // to NULL
    // store last node
    lastlayer = dR_list_next(layer->previous_layers);
    // transfer output shape of previous node to fft node
    fft->ishape.s0 = lastlayer->oshape.s0;
    fft->ishape.s1 = lastlayer->oshape.s1;
    fft->ishape.s2 = lastlayer->oshape.s2;

    if(fft->ishape.s0!=lastlayer->oshape.s0||fft->ishape.s1!=lastlayer->oshape.s1||fft->ishape.s2!=lastlayer->oshape.s2)
    {
        if(!net->config->silent)
        {
            g_print("FFT Node needs 1 input node with the same shape!\n");
            g_print("[%d, %d, %d] and [%d, %d, %d] not matching!\n",
                    fft->ishape.s0,fft->ishape.s1,fft->ishape.s2,lastlayer->oshape.s0,lastlayer->oshape.s1,lastlayer->oshape.s2);
        }
        return FALSE;
    }

    layer->oshape.s0 = fft->ishape.s0;
    layer->oshape.s1 = fft->ishape.s1;
    layer->oshape.s2 = fft->ishape.s2;
    return TRUE;
}

gint32 dR_fft_getRequiredOutputBufferSize(dR_Node* layer)
{
    /* input elements * 2 = output, complex numbers in output*/
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2 * 2;
}

gboolean dR_fft_createKernel(dR_Graph* net, dR_Node* layer)
{
    //call all Opencl kernel creation routines required
    dR_FFT_Data* fft = (dR_FFT_Data*)(layer->layer);
    gboolean ret=FALSE;
    ret = dR_createKernel(net,"fft",&(layer->clKernel));
    return ret;
}


gboolean dR_fft_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    /* create buffer for fft intermediate steps */
    gboolean ret = TRUE;
    if(!net->prepared)
    {
        dR_FFT_Data* fft = ((dR_FFT_Data*)(layer->layer));
        /* *2 to store real and imag part of complex number */
        fft->intermedBuf = g_malloc(fft->ishape.s0*fft->ishape.s1*fft->ishape.s2*sizeof(gfloat) * 2 );
    }
    return ret;
}

gboolean dR_fft_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_fft_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    if(net->prepared)
    {
        dR_FFT_Data* fft = ((dR_FFT_Data*)(layer->layer));
        ret &= dR_cleanupKernel((fft->intermedBuf));
        ret &= dR_cleanupKernel((layer->clKernel));
    }
    return ret;
}

gboolean dR_fft_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    dR_FFT_Data* fft = ((dR_FFT_Data*)(layer->layer));
    // free all memory that was reserved for node
    if(net->prepared)
        g_free(fft->intermedBuf);
        g_free(fft);
    return TRUE;
}

gchar* dR_fft_printLayer(dR_Node* layer)
{
    // print node
    dR_FFT_Data* fft = (dR_FFT_Data*)(layer->layer);
    gchar* out;
    out = g_strdup_printf("%s%d%s",
            "FFT input operation node: ",layer->layerID, "\n");
    return out;
}
