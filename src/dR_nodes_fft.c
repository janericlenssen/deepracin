#include "dR_nodes_fft.h"
#include "dR_core.h"

// ///////////////////////////
// Fast Fourier Transform //
// ///////////////////////////

dR_Node* dR_FFT(dR_Graph* net, dR_Node* inputNode1, gboolean inverse)
{
    // allocate memory for dR_Shape3 (3 dimensional vector)
    dR_FFT_Data* fft = g_malloc(sizeof(dR_FFT_Data));
    // allocate memory for a node
    dR_Node* l = g_malloc(sizeof(dR_Node));

    // set all attributes of fft node
    // dR_Shape3
    l->layer = fft;
    ((dR_FFT_Data*)(l->layer))->inv = inverse;
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
    params[0] = g_strdup_printf("%d",fft->inv);

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

    out = dR_FFT( net, iNodes[0], atoi(params[0]) );
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
    #if 0 // TEST if upscaling the shape in propagateshape destroys image (it does) + TEST if more workitems then pixels destroys image (it does).
    dR_FFT_Data* fft = ((dR_FFT_Data*)(layer->layer));
    cl_int real_width = fft->real_width;
    cl_int real_height = fft->real_height;

    gint power_two = 1;
    gint padToSize = fft->real_width;
    while( power_two < padToSize )
    {
      power_two *= 2;
    }
    padToSize = power_two;

    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input1;
    dR_list_resetIt(layer->previous_layers);

    void *in = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    void *out = (void*)layer->outputBuf->bufptr;

    globalWorkSize[0] = padToSize;
    globalWorkSize[1] = padToSize;
    globalWorkSize[2] = layer->oshape.s2;

    net->clConfig->clError = clSetKernelArg(fft->copyKernel, 0, sizeof(cl_mem), in);

    net->clConfig->clError |= clSetKernelArg(fft->copyKernel, 1, sizeof(cl_mem), out);

    net->clConfig->clError |= clSetKernelArg(fft->copyKernel, 2, sizeof(cl_int), (void *)&real_width);

    net->clConfig->clError |= clSetKernelArg(fft->copyKernel, 3, sizeof(cl_int), (void *)&real_height);

    if (dR_openCLError(net, "Setting kernel args failed.", "copy Kernel"))
        return FALSE;

    net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, fft->copyKernel, 3, NULL, globalWorkSize,
       NULL, 0, NULL, net->clConfig->clEvent);
    dR_finishCLKernel(net, "deepRACIN:copy");
    #endif

    #if 1
    // TODO: seperate fft and inverse fft ?
    dR_FFT_Data* fft = ((dR_FFT_Data*)(layer->layer));
    cl_int real_width = fft->real_width;
    cl_int real_height = fft->real_height;

    gint power_two = 1;
    gint padToSize = fft->real_width;
    while( power_two < padToSize )
    {
      power_two *= 2;
    }
    padToSize = power_two;

    //g_print("pad to: %d", power_two);

    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input1;
    dR_list_resetIt(layer->previous_layers);
    input1 = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;

    globalWorkSize[0] = padToSize / 2; // half of width many workitems needed
    globalWorkSize[1] = padToSize;
    globalWorkSize[2] = layer->oshape.s2;


    int even_odd = 0;
    int lastIn = 0;
    int width = globalWorkSize[0];

    void* in;
    void* out;

    int r_c; // rows_columns: if rows fft is executed then set 1, else 0

    cl_kernel kern;
    // check if inverse is used
    if(fft->inv == 0)
    {
      kern = layer->clKernel; //forward fft
    }
    else
    {
      kern = fft->inverseKernel; //inverse fft
    }
    // there are three arrays: input (the input image), intermedBuf (an intermediate buffer, because we use an out of place fft implementation) and output (the output buffer)
    // flow of data: input -> output -> intermedBuf -> output -> intermedBuf -> ... -> output
    //----------------------- rows fft ----------------------//
    r_c = 1;
    for(cl_int p = 1; p <= width; p *= 2)
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

      net->clConfig->clError = clSetKernelArg(kern, 0, sizeof(cl_mem), in);

      net->clConfig->clError |= clSetKernelArg(kern, 1, sizeof(cl_mem), out);

      net->clConfig->clError |= clSetKernelArg(kern, 2, sizeof(cl_int), (void *)&p);

      net->clConfig->clError |= clSetKernelArg(kern, 3, sizeof(cl_int), (void *)&real_width);

      net->clConfig->clError |= clSetKernelArg(kern, 4, sizeof(cl_int), (void *)&real_height);

      if(fft->inv != 1) // only forward fft uses r_c parameter
      {
        net->clConfig->clError |= clSetKernelArg(kern, 5, sizeof(cl_int), (void *)&r_c);
        if (dR_openCLError(net, "Setting kernel args failed.", "FFT Kernel"))
            return FALSE;
      }
      else
      {
        if (dR_openCLError(net, "Setting kernel args failed.", "FFT inverse Kernel"))
            return FALSE;
      }

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, kern, 3, NULL, globalWorkSize,
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
    // do transpose and copy with one workitem per pixel
    globalWorkSize[0] = padToSize;
    globalWorkSize[1] = padToSize;
    globalWorkSize[2] = layer->oshape.s2;
    //transpose
    net->clConfig->clError = clSetKernelArg(fft->transposeKernel, 0, sizeof(cl_mem), in);

    net->clConfig->clError |= clSetKernelArg(fft->transposeKernel, 1, sizeof(cl_mem), out);

    net->clConfig->clError |= clSetKernelArg(fft->transposeKernel, 2, sizeof(cl_int), (void *)&real_width);

    net->clConfig->clError |= clSetKernelArg(fft->transposeKernel, 3, sizeof(cl_int), (void *)&real_height);

    if (dR_openCLError(net, "Setting kernel args failed.", "transpose Kernel"))
        return FALSE;

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

      net->clConfig->clError |= clSetKernelArg(fft->copyKernel, 2, sizeof(cl_int), (void *)&real_width);

      net->clConfig->clError |= clSetKernelArg(fft->copyKernel, 3, sizeof(cl_int), (void *)&real_height);

      if (dR_openCLError(net, "Setting kernel args failed.", "copy Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, fft->copyKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:copy");
    }

    //----------------------- columns fft ----------------------//
    // TODO: could do a loop, with just one block of code. but would need more cases and could be harder to understand
    globalWorkSize[0] = padToSize/2; //height is the new width
    globalWorkSize[1] = padToSize;
    globalWorkSize[2] = layer->oshape.s2;
    // do fft on transposed image again
    even_odd = 1;
    lastIn = 1;
    int height = globalWorkSize[0];

    r_c = 0; //columns now
    for(cl_int p = 1; p <= height; p *= 2)
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
      net->clConfig->clError = clSetKernelArg(kern, 0, sizeof(cl_mem), in);

      net->clConfig->clError |= clSetKernelArg(kern, 1, sizeof(cl_mem), out);

      net->clConfig->clError |= clSetKernelArg(kern, 2, sizeof(cl_int), (void *)&p);

      net->clConfig->clError |= clSetKernelArg(kern, 3, sizeof(cl_int), (void *)&real_height);

      net->clConfig->clError |= clSetKernelArg(kern, 4, sizeof(cl_int), (void *)&real_width);

      if(fft->inv != 1) // only forward fft uses r_c parameter
      {
        net->clConfig->clError |= clSetKernelArg(kern, 5, sizeof(cl_int), (void *)&r_c);
        if (dR_openCLError(net, "Setting kernel args failed.", "FFT Kernel"))
            return FALSE;
      }
      else
      {
        if (dR_openCLError(net, "Setting kernel args failed.", "FFT inverse Kernel"))
            return FALSE;
      }

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, kern, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:fft");
      even_odd++;
    }

    if( lastIn == 0 )
    {
      in = (void*)&fft->intermedBuf;
      out = (void*)layer->outputBuf->bufptr;
    }
    else
    {
      in =  (void*)layer->outputBuf->bufptr;
      out = (void*)&fft->intermedBuf;
    }

    globalWorkSize[0] = padToSize;
    globalWorkSize[1] = padToSize;
    globalWorkSize[2] = layer->oshape.s2;

    net->clConfig->clError = clSetKernelArg(fft->transposeKernel, 0, sizeof(cl_mem), in);

    net->clConfig->clError |= clSetKernelArg(fft->transposeKernel, 1, sizeof(cl_mem), out);

    net->clConfig->clError |= clSetKernelArg(fft->transposeKernel, 2, sizeof(cl_int), (void *)&real_height);

    net->clConfig->clError |= clSetKernelArg(fft->transposeKernel, 3, sizeof(cl_int), (void *)&real_width);

    if (dR_openCLError(net, "Setting kernel args failed.", "transpose Kernel"))
        return FALSE;

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

      net->clConfig->clError |= clSetKernelArg(fft->copyKernel, 2, sizeof(cl_int), (void *)&real_width);

      net->clConfig->clError |= clSetKernelArg(fft->copyKernel, 3, sizeof(cl_int), (void *)&real_height);

      if (dR_openCLError(net, "Setting kernel args failed.", "copy Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, fft->copyKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:copy");
    }

    //normalize when using inverse
    if(fft->inv == 1)
    {
      out = (void*)layer->outputBuf->bufptr;
      net->clConfig->clError = clSetKernelArg(fft->normalizeKernel, 0, sizeof(cl_mem), out);

      net->clConfig->clError |= clSetKernelArg(fft->normalizeKernel, 1, sizeof(cl_int), (void *)&real_width);

      net->clConfig->clError |= clSetKernelArg(fft->normalizeKernel, 2, sizeof(cl_int), (void *)&real_height);

      if (dR_openCLError(net, "Setting kernel args failed.", "normalize Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, fft->normalizeKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:normalize");
    }

    return 1; // TODO: what should be returned ?
    #endif
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

    gint32 padToSize = lastlayer->oshape.s0 > lastlayer->oshape.s1 ? lastlayer->oshape.s0 : lastlayer->oshape.s1;

    int power_two = 1;
    while( power_two < padToSize )
    {
      power_two *= 2;
    }
    padToSize = power_two;
    //g_print("pad to: %d", power_two);

    // store the actual width and height of the image
    fft->real_width = lastlayer->oshape.s0;
    fft->real_height = lastlayer->oshape.s1;
    // if quadratic image
    if (lastlayer->oshape.s0 == lastlayer->oshape.s1)
    {
      fft->ishape.s0 = lastlayer->oshape.s0;
      fft->ishape.s1 = lastlayer->oshape.s1;
      fft->ishape.s2 = lastlayer->oshape.s2;
    }
    else // image is not quadratic
    {
      //g_print("non-quadratic image\n");
      //save bigger size to make a quadratic image out of non-quadratic image
      //g_print("bigger size: %d\n", padToSize);

      fft->ishape.s0 = lastlayer->oshape.s0;
      fft->ishape.s1 = lastlayer->oshape.s1;
      fft->ishape.s2 = lastlayer->oshape.s2;
    }

    layer->oshape.s0 = fft->ishape.s0;
    layer->oshape.s1 = fft->ishape.s1;
    //double this dimension to store real and img parts of fft
    if (fft->inv == 1)
    {
      layer->oshape.s2 = fft->ishape.s2;
    }
    else
    {
      layer->oshape.s2 = 2*fft->ishape.s2;
    }

    return TRUE;
}

gint32 dR_fft_getRequiredOutputBufferSize(dR_Node* layer)
{
    /* input elements * 2 = output, complex numbers in output*/
    dR_FFT_Data* fft = (dR_FFT_Data*)(layer->layer);
    gint32 ret;
    if(fft->inv == 1)
    {
      ret = layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
    }
    else
    {
      ret = layer->oshape.s0*layer->oshape.s1*layer->oshape.s2*2;
    }
    return ret;
}

gboolean dR_fft_createKernel(dR_Graph* net, dR_Node* layer)
{
    //call all Opencl kernel creation routines required
    dR_FFT_Data* fft = (dR_FFT_Data*)(layer->layer);
    gboolean ret=FALSE;
    ret = dR_createKernel(net,"fft",&(layer->clKernel));
    ret = dR_createKernel(net,"transpose",&(fft->transposeKernel));
    ret = dR_createKernel(net,"copy",&(fft->copyKernel));
    ret = dR_createKernel(net,"fft_inv",&(fft->inverseKernel));
    ret = dR_createKernel(net,"normalizeFFT",&(fft->normalizeKernel));
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
        if (fft->inv == 1)
        {
          ret &= dR_createFloatBuffer(net, &(fft->intermedBuf),shape.s0*shape.s1*shape.s2*sizeof(gfloat), CL_MEM_READ_WRITE);
        }
        else
        {
          ret &= dR_createFloatBuffer(net, &(fft->intermedBuf),shape.s0*shape.s1*shape.s2*sizeof(gfloat)*2, CL_MEM_READ_WRITE);
        }
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
        ret &= dR_cleanupKernel((fft->inverseKernel));
        ret &= dR_cleanupKernel((fft->normalizeKernel));
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

    if(fft->inv == 1)
    {
      out =  g_strdup_printf("%s%d%s",
              "FFT Inverse input operation node: ",layer->layerID, "\n");
    }
    else
    {
      out = g_strdup_printf("%s%d%s",
            "FFT input operation node: ",layer->layerID, "\n");
    }
    return out;
}

//------------fftshift------------------
// inspired by https://www.researchgate.net/publication/278847958_CufftShift_High_performance_CUDA-accelerated_FFT-shift_library

dR_Node* dR_FFTShift(dR_Graph* net, dR_Node* inputNode1)
{
    // allocate memory for dR_Shape3 (3 dimensional vector)
    dR_FFTShift_Data* fftshift = g_malloc(sizeof(dR_FFTShift_Data));
    // allocate memory for a node
    dR_Node* l = g_malloc(sizeof(dR_Node));

    // set all attributes of fft node
    // dR_Shape3
    l->layer = fftshift;
    // node type
    l->type = tFFTShift;
    // set functions (implemented in this file) for this node

    l->compute = dR_fftshift_compute;

    l->schedule = dR_fftshift_schedule;
    l->propagateShape = dR_fftshift_propagateShape;
    l->getRequiredOutputBufferSize = dR_fftshift_getRequiredOutputBufferSize;
    l->createKernel = dR_fftshift_createKernel;
    l->allocateBuffers = dR_fftshift_allocateBuffers;
    l->fillBuffers = dR_fftshift_fillBuffers;
    l->cleanupBuffers = dR_fftshift_cleanupBuffers;
    l->cleanupLayer = dR_fftshift_cleanupLayer;
    l->serializeNode = dR_fftshift_serializeNode;
    l->parseAppendNode = dR_fftshift_parseAppendNode;

    l->generateKernel = NULL;
    l->createKernelName = NULL;
    l->setVariables = NULL;
    l->printLayer = dR_fftshift_printLayer;

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
        g_print("Error: FFTShift node needs an appropriate Inputnode");
    }
    // append node to graph
    dR_appendLayer(net, l);
    // return pointer to node
    return l;
}

gchar* dR_fftshift_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_FFTShift_Data* fft = (dR_FFTShift_Data*)(layer->layer);
    gchar* desc = "fftshift";
    gint numNodeParams = 0;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("FFTShiftNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_fftshift_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 0;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: fftshift Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: fftshift Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }

    out = dR_FFTShift( net, iNodes[0] );
    return out;
}

gboolean dR_fftshift_compute(dR_Graph* net, dR_Node* layer){

    dR_FFTShift_Data* fft = ((dR_FFTShift_Data*)(layer->layer));

    size_t globalWorkSize[3];
    int paramid = 0;
    dR_list_resetIt(layer->previous_layers);

    void *in = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    void *out = (void*)layer->outputBuf->bufptr;

    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;

    net->clConfig->clError = clSetKernelArg(layer->clKernel, 0, sizeof(cl_mem), in);

    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 1, sizeof(cl_mem), out);

    if (dR_openCLError(net, "Setting kernel args failed.", "fftshift Kernel"))
        return FALSE;

    net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
       NULL, 0, NULL, net->clConfig->clEvent);
    dR_finishCLKernel(net, "deepRACIN:fftshift");
    return TRUE;

}

gboolean dR_fftshift_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }

 gboolean dR_fftshift_propagateShape(dR_Graph* net, dR_Node* layer)
 {
     // compute output shape of node
     // output of fftshift is frequency spectrum with re in first array and im in second array
     // input for fftshift is frequency spectrum
     dR_FFTShift_Data* fft = (dR_FFTShift_Data*)(layer->layer);
     dR_Node* lastlayer;
     // check if previous layer gives correct output for fftshift
     if(layer->previous_layers->length!=1)
     {
         if(!net->config->silent)
             g_print("FFTShift Node with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
         return FALSE;
     }
     dR_list_resetIt(layer->previous_layers); // to NULL
     // store last node
     lastlayer = dR_list_next(layer->previous_layers);

     // transfer output shape of previous node to fftshift node
     // input is frequency spectrum, output aswell, but only shifted. shapes are the same in input and output
     fft->ishape.s0 = lastlayer->oshape.s0;
     fft->ishape.s1 = lastlayer->oshape.s1;
     fft->ishape.s2 = lastlayer->oshape.s2;

     layer->oshape.s0 = fft->ishape.s0;
     layer->oshape.s1 = fft->ishape.s1;
     layer->oshape.s2 = fft->ishape.s2;

     return TRUE;
 }

 gint32 dR_fftshift_getRequiredOutputBufferSize(dR_Node* layer)
 {
     dR_FFTShift_Data* fft = (dR_FFTShift_Data*)(layer->layer);
     gint32 ret;
     ret = layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
     return ret;
 }

 gboolean dR_fftshift_createKernel(dR_Graph* net, dR_Node* layer)
 {
     //call all Opencl kernel creation routines required
     dR_FFTShift_Data* fft = (dR_FFTShift_Data*)(layer->layer);
     gboolean ret=FALSE;
     ret = dR_createKernel(net,"shiftFFT",&(layer->clKernel));
     return ret;
 }

 gboolean dR_fftshift_allocateBuffers(dR_Graph* net, dR_Node* layer)
 {
     /* create buffer for intermediate steps */
     gboolean ret = TRUE;
     return ret;
 }

 gboolean dR_fftshift_fillBuffers(dR_Graph* net, dR_Node* layer)
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

 gboolean dR_fftshift_cleanupBuffers(dR_Graph* net, dR_Node* layer)
 {
     gboolean ret = TRUE;
     if(net->prepared)
     {
         dR_FFTShift_Data* fft = ((dR_FFTShift_Data*)(layer->layer));
         ret &= dR_cleanupKernel((layer->clKernel));
     }
     return ret;
 }

 gboolean dR_fftshift_cleanupLayer(dR_Graph* net, dR_Node* layer)
 {
     dR_FFTShift_Data* fft = ((dR_FFTShift_Data*)(layer->layer));
     // free all memory that was reserved for node
     if(net->prepared)
         g_free(fft);
     return TRUE;
 }

 gchar* dR_fftshift_printLayer(dR_Node* layer)
 {
     // print node
     dR_FFTShift_Data* fft = (dR_FFTShift_Data*)(layer->layer);
     gchar* out;
     out =  g_strdup_printf("%s%d%s", "FFTShift operation node: ",layer->layerID, "\n");
     return out;
 }

//fftabs-----------

dR_Node* dR_FFTAbs(dR_Graph* net, dR_Node* inputNode1)
{
    // allocate memory for dR_Shape3 (3 dimensional vector)
    dR_FFTAbs_Data* fftabs = g_malloc(sizeof(dR_FFTAbs_Data));
    // allocate memory for a node
    dR_Node* l = g_malloc(sizeof(dR_Node));

    // set all attributes of fft node
    // dR_Shape3
    l->layer = fftabs;
    // node type
    l->type = tFFTAbs;
    // set functions (implemented in this file) for this node

    l->compute = dR_fftabs_compute;

    l->schedule = dR_fftabs_schedule;
    l->propagateShape = dR_fftabs_propagateShape;
    l->getRequiredOutputBufferSize = dR_fftabs_getRequiredOutputBufferSize;
    l->createKernel = dR_fftabs_createKernel;
    l->allocateBuffers = dR_fftabs_allocateBuffers;
    l->fillBuffers = dR_fftabs_fillBuffers;
    l->cleanupBuffers = dR_fftabs_cleanupBuffers;
    l->cleanupLayer = dR_fftabs_cleanupLayer;
    l->serializeNode = dR_fftabs_serializeNode;
    l->parseAppendNode = dR_fftabs_parseAppendNode;

    l->generateKernel = NULL;
    l->createKernelName = NULL;
    l->setVariables = NULL;
    l->printLayer = dR_fftabs_printLayer;

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
        g_print("Error: FFTAbs node needs an appropriate Inputnode");
    }
    // append node to graph
    dR_appendLayer(net, l);
    // return pointer to node
    return l;
}

gchar* dR_fftabs_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_FFTAbs_Data* fft = (dR_FFTAbs_Data*)(layer->layer);
    gchar* desc = "fftabs";
    gint numNodeParams = 0;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("FFTAbs Node needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_fftabs_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 0;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: fftabs Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: fftabs Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }

    out = dR_FFTAbs( net, iNodes[0] );
    return out;
}

gboolean dR_fftabs_compute(dR_Graph* net, dR_Node* layer){

    dR_FFTAbs_Data* fft = ((dR_FFTAbs_Data*)(layer->layer));

    size_t globalWorkSize[3];
    int paramid = 0;
    dR_list_resetIt(layer->previous_layers);

    void *in = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    void *out = (void*)layer->outputBuf->bufptr;

    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;

    net->clConfig->clError = clSetKernelArg(layer->clKernel, 0, sizeof(cl_mem), in);

    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 1, sizeof(cl_mem), out);

    if (dR_openCLError(net, "Setting kernel args failed.", "fftabs Kernel"))
        return FALSE;

    net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
       NULL, 0, NULL, net->clConfig->clEvent);
    dR_finishCLKernel(net, "deepRACIN:fftabs");
    return TRUE;

}

gboolean dR_fftabs_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }

 gboolean dR_fftabs_propagateShape(dR_Graph* net, dR_Node* layer)
 {
     // compute output shape of node
     dR_FFTAbs_Data* fft = (dR_FFTAbs_Data*)(layer->layer);
     dR_Node* lastlayer;

     if(layer->previous_layers->length!=1)
     {
         if(!net->config->silent)
             g_print("FFTAbs Node with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
         return FALSE;
     }
     dR_list_resetIt(layer->previous_layers); // to NULL
     // store last node
     lastlayer = dR_list_next(layer->previous_layers);

     fft->ishape.s0 = lastlayer->oshape.s0;
     fft->ishape.s1 = lastlayer->oshape.s1;
     fft->ishape.s2 = lastlayer->oshape.s2;

     layer->oshape.s0 = fft->ishape.s0;
     layer->oshape.s1 = fft->ishape.s1;
     layer->oshape.s2 = fft->ishape.s2/2;

     return TRUE;
 }

 gint32 dR_fftabs_getRequiredOutputBufferSize(dR_Node* layer)
 {
     dR_FFTAbs_Data* fft = (dR_FFTAbs_Data*)(layer->layer);
     gint32 ret;
     ret = layer->oshape.s0*layer->oshape.s1;
     return ret;
 }

 gboolean dR_fftabs_createKernel(dR_Graph* net, dR_Node* layer)
 {
     //call all Opencl kernel creation routines required
     dR_FFTAbs_Data* fft = (dR_FFTAbs_Data*)(layer->layer);
     gboolean ret=FALSE;
     ret = dR_createKernel(net,"absFFT",&(layer->clKernel));
     return ret;
 }

 gboolean dR_fftabs_allocateBuffers(dR_Graph* net, dR_Node* layer)
 {
     /* create buffer for intermediate steps */
     gboolean ret = TRUE;
     return ret;
 }

 gboolean dR_fftabs_fillBuffers(dR_Graph* net, dR_Node* layer)
 {
     net = net;
     layer = layer;
     return TRUE;
 }

 gboolean dR_fftabs_cleanupBuffers(dR_Graph* net, dR_Node* layer)
 {
     gboolean ret = TRUE;
     if(net->prepared)
     {
         dR_FFTAbs_Data* fft = ((dR_FFTAbs_Data*)(layer->layer));
         ret &= dR_cleanupKernel((layer->clKernel));
     }
     return ret;
 }

 gboolean dR_fftabs_cleanupLayer(dR_Graph* net, dR_Node* layer)
 {
     dR_FFTShift_Data* fft = ((dR_FFTShift_Data*)(layer->layer));
     // free all memory that was reserved for node
     if(net->prepared)
         g_free(fft);
     return TRUE;
 }

 gchar* dR_fftabs_printLayer(dR_Node* layer)
 {
     // print node
     dR_FFTAbs_Data* fft = (dR_FFTAbs_Data*)(layer->layer);
     gchar* out;
     out =  g_strdup_printf("%s%d%s", "FFTAbs operation node: ",layer->layerID, "\n");
     return out;
 }
