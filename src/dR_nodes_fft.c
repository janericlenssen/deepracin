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
        g_print("FFTNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
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

    //for easier intermedBuf access
    dR_FFT_Data* fft = ((dR_FFT_Data*)(layer->layer));

    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input1;
    dR_list_resetIt(layer->previous_layers);
    input1 = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;

    int even_odd = 0;
    int lastIn = 0;
    int w = layer->oshape.s0;
    cl_int r_c; // if r_c is 1 then rows, if r_c is 0 then columns fft is executed

    void* in;
    void* out;

    r_c = 1; // just real input for p==1
    for(cl_int p = 1; p <= w/2; p *= 2)
    {
      if (p==1)
      {
        in = (void*)input1;
        out = (void *)layer->outputBuf->bufptr;
      }
      else
      {
        if (even_odd % 2) // odd
        {
          in = (void *)layer->outputBuf->bufptr;
          out = (void*)&fft->intermedBuf;
          lastIn = 0;
        }
        else // even
        {
          in = (void*)&fft->intermedBuf;
          out = (void *)layer->outputBuf->bufptr;
          lastIn = 1;
        }
      }

      net->clConfig->clError = clSetKernelArg(layer->clKernel, 0, sizeof(cl_mem), in);

      net->clConfig->clError |= clSetKernelArg(layer->clKernel, 1, sizeof(cl_mem), out);

      net->clConfig->clError |= clSetKernelArg(layer->clKernel, 2, sizeof(cl_int), (void *)&p);

      net->clConfig->clError |= clSetKernelArg(layer->clKernel, 3, sizeof(cl_int), (void *)&r_c);

      if (dR_openCLError(net, "Setting kernel args failed.", "FFT Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:fft");
      even_odd++;
    }

    if( lastIn == 0 )
    {
      //transform is in intermedBuf, transpose from intermedBuf to outputarray
      in = (void*)&fft->intermedBuf;
      out = (void*)layer->outputBuf->bufptr;
    }
    else
    {
      //transform is already in outputarray, transpose into intermedBuf and then copy to outputBuf later
      in =  (void*)layer->outputBuf->bufptr;
      out = (void*)&fft->intermedBuf;
    }

    //transpose
    net->clConfig->clError = clSetKernelArg(fft->transposeKernel, 0, sizeof(cl_mem), in);

    net->clConfig->clError |= clSetKernelArg(fft->transposeKernel, 1, sizeof(cl_mem), out);

    net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, fft->transposeKernel, 3, NULL, globalWorkSize,
       NULL, 0, NULL, net->clConfig->clEvent);
    dR_finishCLKernel(net, "deepRACIN:transpose");

    if (lastIn == 1)
    {
      //copy to output buffer because transpose is in intermedBuf
      in =  (void*)&fft->intermedBuf;
      out = (void*)layer->outputBuf->bufptr;

      net->clConfig->clError = clSetKernelArg(fft->copyKernel, 0, sizeof(cl_mem), in);

      net->clConfig->clError |= clSetKernelArg(fft->copyKernel, 1, sizeof(cl_mem), out);

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, fft->copyKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:copy");
    }

    //------------------------------------------------------------------------//

    // then do fft on transposed array again
    even_odd = 1;
    lastIn = 1;
    int h = layer->oshape.s1; // height

    r_c = 0; // complex input for p==1
    for(cl_int p = 1; p <= h/2; p *= 2)
    {
      if (even_odd % 2) // odd
      {
        in = (void *)layer->outputBuf->bufptr;
        out = (void*)&fft->intermedBuf;
        lastIn = 0;
      }
      else // even
      {
        in = (void*)&fft->intermedBuf;
        out = (void *)layer->outputBuf->bufptr;
        lastIn = 1;
      }
      net->clConfig->clError = clSetKernelArg(layer->clKernel, 0, sizeof(cl_mem), in);

      net->clConfig->clError |= clSetKernelArg(layer->clKernel, 1, sizeof(cl_mem), out);

      net->clConfig->clError |= clSetKernelArg(layer->clKernel, 2, sizeof(cl_int), (void *)&p);

      net->clConfig->clError |= clSetKernelArg(layer->clKernel, 3, sizeof(cl_int), (void *)&r_c);

      if (dR_openCLError(net, "Setting kernel args failed.", "FFT Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:fft");
      even_odd++;
    }

    if( lastIn == 0 )
    {
      //transform is in intermedBuf, transpose from intermedBuf to outputarray
      in = (void*)&fft->intermedBuf;
      out = (void*)layer->outputBuf->bufptr;
    }
    else
    {
      //transform is already in outputarray, transpose into intermedBuf and then copy to outputBuf later
      in =  (void*)layer->outputBuf->bufptr;
      out = (void*)&fft->intermedBuf;
    }

    //transpose
    net->clConfig->clError = clSetKernelArg(fft->transposeKernel, 0, sizeof(cl_mem), in);

    net->clConfig->clError |= clSetKernelArg(fft->transposeKernel, 1, sizeof(cl_mem), out);

    net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, fft->transposeKernel, 3, NULL, globalWorkSize,
       NULL, 0, NULL, net->clConfig->clEvent);
    dR_finishCLKernel(net, "deepRACIN:transpose");

    if (lastIn == 1)
    {
      //copy to output buffer because transpose is in intermedBuf
      in =  (void*)&fft->intermedBuf;
      out = (void*)layer->outputBuf->bufptr;

      net->clConfig->clError = clSetKernelArg(fft->copyKernel, 0, sizeof(cl_mem), in);

      net->clConfig->clError |= clSetKernelArg(fft->copyKernel, 1, sizeof(cl_mem), out);

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, fft->copyKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:copy");
    }
    return 1; // TODO: what should be returned ?
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
    //double this dimension to store real and img parts of fft
    layer->oshape.s2 = 2*fft->ishape.s2;
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
    ret = dR_createKernel(net,"transpose",&(fft->transposeKernel));
    ret = dR_createKernel(net,"copy",&(fft->copyKernel));
    return ret;
}


gboolean dR_fft_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    /* create buffer for fft intermediate steps */
    gboolean ret = TRUE;
    dR_FFT_Data* fft = ((dR_FFT_Data*)(layer->layer));
    dR_Shape3 shape = fft->ishape;
    if(!net->prepared)
    {
        /* *2 to store real and imag part of complex number */
        ret &= dR_createFloatBuffer(net, &(fft->intermedBuf),shape.s0*shape.s1*shape.s2*sizeof(gfloat)*2, CL_MEM_READ_WRITE);
    }
    return ret;
}

gboolean dR_fft_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    /*
    dR_FFT_Data* fft = ((dR_FFT_Data*)(layer->layer));
    shape = fft->shape;
    ret &=  dR_uploadArray(net,"",fft->HOSTMEMTOUPLOAD,0,shape.s0*shape.s1*shape.s2*shape.s3*sizeof(gfloat)*2,fft->OPENCLMEM);
    */
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
        ret &= dR_clMemoryBufferCleanup(net, fft->intermedBuf);
        ret &= dR_cleanupKernel((layer->clKernel));
        ret &= dR_cleanupKernel((fft->transposeKernel));
        ret &= dR_cleanupKernel((fft->copyKernel));
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
