#include "dR_nodes_haarwt.h"
#include "dR_core.h"

dR_Node* dR_Haarwt(dR_Graph* net, dR_Node* inputNode1, gint level)
{
    // allocate memory for dR_Shape3 (3 dimensional vector)
    dR_Haarwt_Data* haarwt = g_malloc(sizeof(dR_Haarwt_Data));
    // allocate memory for a node
    dR_Node* l = g_malloc(sizeof(dR_Node));

    haarwt->level = level;
    // set all attributes of haarwt node
    // dR_Shape3
    l->layer = haarwt;
    // node type
    l->type = tHaarwt;
    // set functions (implemented in this file) for this node

    l->compute = dR_haarwt_compute;

    l->schedule = dR_haarwt_schedule;
    l->propagateShape = dR_haarwt_propagateShape;
    l->getRequiredOutputBufferSize = dR_haarwt_getRequiredOutputBufferSize;
    l->createKernel = dR_haarwt_createKernel;
    l->allocateBuffers = dR_haarwt_allocateBuffers;
    l->fillBuffers = dR_haarwt_fillBuffers;
    l->cleanupBuffers = dR_haarwt_cleanupBuffers;
    l->cleanupLayer = dR_haarwt_cleanupLayer;
    l->serializeNode = dR_haarwt_serializeNode;
    l->parseAppendNode = dR_haarwt_parseAppendNode;

    l->generateKernel = NULL;
    l->createKernelName = NULL;
    l->setVariables = NULL;
    l->printLayer = dR_haarwt_printLayer;

    if (inputNode1)
    {
      // create empty list for previous nodes
      l->previous_layers = dR_list_createEmptyList();
      // append the input of this node to the list
      dR_list_append(l->previous_layers,inputNode1);
      // create empty list for following nodes
      l->next_layers = dR_list_createEmptyList();
      // append the current (haarwt) node as the following node of the previous node
      dR_list_append(inputNode1->next_layers,l);
    }
    else
    {
        g_print("Error: haarwt node needs an appropriate Inputnode");
    }
    // append node to graph
    dR_appendLayer(net, l);
    // return pointer to node
    return l;
}

gchar* dR_haarwt_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_Haarwt_Data* haarwt = (dR_Haarwt_Data*)(layer->layer);
    gchar* desc = "haarwt";
    gint numNodeParams = 0;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("haarwt Node needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_haarwt_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 1;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: haarwt Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: haarwt Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }

    out = dR_Haarwt( net, iNodes[0], atoi(params[0]) );
    return out;
}

gboolean dR_haarwt_compute(dR_Graph* net, dR_Node* layer)
{
    dR_Haarwt_Data* haarwt = ((dR_Haarwt_Data*)(layer->layer));
    cl_mem* input1 = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    cl_mem* output1 = (void*)layer->outputBuf->bufptr;
    gint level = 2 << (haarwt->level - 1);
    size_t globalWorkSize[3];
    int paramid = 0;
    dR_list_resetIt(layer->previous_layers);

    void *in = input1;
    void *out = output1;
    cl_int img_width = layer->oshape.s0;

    for(int i = 2; i <= level; i*=2)
    {
      // rows decimation
      globalWorkSize[0] = layer->oshape.s0/i;
      globalWorkSize[1] = layer->oshape.s1/i*2;
      globalWorkSize[2] = layer->oshape.s2;

      net->clConfig->clError = clSetKernelArg(layer->clKernel, 0, sizeof(cl_mem), in);

      net->clConfig->clError |= clSetKernelArg(layer->clKernel, 1, sizeof(cl_mem), out);

      net->clConfig->clError |= clSetKernelArg(layer->clKernel, 2, sizeof(cl_int), (void *)&img_width);

      if (dR_openCLError(net, "Setting kernel args failed.", "haarwt Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:haarwt");

      // copy from out to in. 1
      globalWorkSize[0] = layer->oshape.s0;
      globalWorkSize[1] = layer->oshape.s1;
      globalWorkSize[2] = layer->oshape.s2;

      net->clConfig->clError = clSetKernelArg(haarwt->copyKernel, 0, sizeof(cl_mem), out);

      net->clConfig->clError |= clSetKernelArg(haarwt->copyKernel, 1, sizeof(cl_mem), in);

      if (dR_openCLError(net, "Setting kernel args failed.", "hwtcopy Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, haarwt->copyKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:hwtcopy");

      // transpose
      net->clConfig->clError = clSetKernelArg(haarwt->transposeKernel, 0, sizeof(cl_mem), in);

      net->clConfig->clError |= clSetKernelArg(haarwt->transposeKernel, 1, sizeof(cl_mem), out);

      if (dR_openCLError(net, "Setting kernel args failed.", "hwttranspose Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, haarwt->transposeKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:hwttranspose");

      //copy
      net->clConfig->clError = clSetKernelArg(haarwt->copyKernel, 0, sizeof(cl_mem), out);

      net->clConfig->clError |= clSetKernelArg(haarwt->copyKernel, 1, sizeof(cl_mem), in);

      if (dR_openCLError(net, "Setting kernel args failed.", "hwtcopy Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, haarwt->copyKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:hwtcopy");

      // columns decimation
      globalWorkSize[0] = layer->oshape.s0/i;
      globalWorkSize[1] = layer->oshape.s1/i*2;
      globalWorkSize[2] = layer->oshape.s2;

      net->clConfig->clError = clSetKernelArg(layer->clKernel, 0, sizeof(cl_mem), in);

      net->clConfig->clError |= clSetKernelArg(layer->clKernel, 1, sizeof(cl_mem), out);

      net->clConfig->clError |= clSetKernelArg(layer->clKernel, 2, sizeof(cl_int), (void *)&img_width);

      if (dR_openCLError(net, "Setting kernel args failed.", "haarwt Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:haarwt");

      // transpose back
      globalWorkSize[0] = layer->oshape.s0;
      globalWorkSize[1] = layer->oshape.s1;
      globalWorkSize[2] = layer->oshape.s2;
      net->clConfig->clError = clSetKernelArg(haarwt->transposeKernel, 0, sizeof(cl_mem), out);

      net->clConfig->clError |= clSetKernelArg(haarwt->transposeKernel, 1, sizeof(cl_mem), in);

      if (dR_openCLError(net, "Setting kernel args failed.", "hwttranspose Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, haarwt->transposeKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:hwttranspose");

      //copy
      net->clConfig->clError = clSetKernelArg(haarwt->copyKernel, 0, sizeof(cl_mem), in);

      net->clConfig->clError |= clSetKernelArg(haarwt->copyKernel, 1, sizeof(cl_mem), out);

      if (dR_openCLError(net, "Setting kernel args failed.", "hwtcopy Kernel"))
          return FALSE;

      net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, haarwt->copyKernel, 3, NULL, globalWorkSize,
         NULL, 0, NULL, net->clConfig->clEvent);
      dR_finishCLKernel(net, "deepRACIN:hwtcopy");
    }
    return TRUE;
}


gboolean dR_haarwt_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }

 gboolean dR_haarwt_propagateShape(dR_Graph* net, dR_Node* layer)
 {
     dR_Haarwt_Data* haarwt = (dR_Haarwt_Data*)(layer->layer);
     dR_Node* lastlayer;

     if(layer->previous_layers->length!=1)
     {
         if(!net->config->silent)
             g_print("haarwt Node with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
         return FALSE;
     }
     dR_list_resetIt(layer->previous_layers); // to NULL
     // store last node
     lastlayer = dR_list_next(layer->previous_layers);

     // input shape of haarwt node is a grayscale image
     haarwt->ishape.s0 = lastlayer->oshape.s0;
     haarwt->ishape.s1 = lastlayer->oshape.s1;
     haarwt->ishape.s2 = lastlayer->oshape.s2;

     layer->oshape.s0 = lastlayer->oshape.s0;
     layer->oshape.s1 = lastlayer->oshape.s1;//1;//lastlayer->oshape.s1; TODO: for now. later 2D
     layer->oshape.s2 = lastlayer->oshape.s2;//1;
     return TRUE;
 }

 gint32 dR_haarwt_getRequiredOutputBufferSize(dR_Node* layer)
 {
     dR_Haarwt_Data* haarwt = (dR_Haarwt_Data*)(layer->layer);
     gint32 ret = layer->oshape.s0*layer->oshape.s1;
     return ret;
 }

 gboolean dR_haarwt_createKernel(dR_Graph* net, dR_Node* layer)
 {
     //call all Opencl kernel creation routines required
     dR_Haarwt_Data* haarwt = (dR_Haarwt_Data*)(layer->layer);
     gboolean ret = FALSE;
     ret = dR_createKernel(net,"haarwt",&(layer->clKernel));
     ret = dR_createKernel(net,"hwtcopy",&(haarwt->copyKernel));
     ret = dR_createKernel(net,"hwttranspose",&(haarwt->transposeKernel));
     return ret;
 }

 gboolean dR_haarwt_allocateBuffers(dR_Graph* net, dR_Node* layer)
 {
     /* create buffer for intermediate steps */
     dR_Haarwt_Data* haarwt = ((dR_Haarwt_Data*)(layer->layer));
     gboolean ret = TRUE;
     //haarwt->hostmem = g_malloc(haarwt->x0*haarwt->y0*sizeof(cl_float)*4);
     return ret;
 }

 gboolean dR_haarwt_fillBuffers(dR_Graph* net, dR_Node* layer)
 {
     net = net;
     layer = layer;
     return TRUE;
 }

 gboolean dR_haarwt_cleanupBuffers(dR_Graph* net, dR_Node* layer)
 {
     gboolean ret = TRUE;
     if(net->prepared)
     {
         dR_Haarwt_Data* haarwt = ((dR_Haarwt_Data*)(layer->layer));
         ret &= dR_cleanupKernel((layer->clKernel));
         ret &= dR_cleanupKernel((haarwt->copyKernel));
         ret &= dR_cleanupKernel((haarwt->transposeKernel));
     }
     return ret;
 }

 gboolean dR_haarwt_cleanupLayer(dR_Graph* net, dR_Node* layer)
 {
     dR_Haarwt_Data* haarwt = ((dR_Haarwt_Data*)(layer->layer));
     // free all memory that was reserved for node
     if(net->prepared)
     {
        // g_free(haarwt->hostmem);
         g_free(haarwt);
     }
     return TRUE;
 }

 gchar* dR_haarwt_printLayer(dR_Node* layer)
 {
     // print node
     dR_Haarwt_Data* haarwt = (dR_Haarwt_Data*)(layer->layer);
     gchar* out;
     out =  g_strdup_printf("%s%d%s", "haarwt operation node: ",layer->layerID, "\n");
     return out;
 }

// ***************************************************************************
// WENERGY2 lvl 3


dR_Node* dR_Wenergy2(dR_Graph* net, dR_Node* inputNode1)
{
    // allocate memory for dR_Shape3 (3 dimensional vector)
    dR_Wenergy2_Data* wenergy2 = g_malloc(sizeof(dR_Wenergy2_Data));
    // allocate memory for a node
    dR_Node* l = g_malloc(sizeof(dR_Node));

    // set all attributes of wenergy2 node
    // dR_Shape3
    l->layer = wenergy2;
    // node type
    l->type = tWenergy2;
    // set functions (implemented in this file) for this node

    l->compute = dR_wenergy2_compute;

    l->schedule = dR_wenergy2_schedule;
    l->propagateShape = dR_wenergy2_propagateShape;
    l->getRequiredOutputBufferSize = dR_wenergy2_getRequiredOutputBufferSize;
    l->createKernel = dR_wenergy2_createKernel;
    l->allocateBuffers = dR_wenergy2_allocateBuffers;
    l->fillBuffers = dR_wenergy2_fillBuffers;
    l->cleanupBuffers = dR_wenergy2_cleanupBuffers;
    l->cleanupLayer = dR_wenergy2_cleanupLayer;
    l->serializeNode = dR_wenergy2_serializeNode;
    l->parseAppendNode = dR_wenergy2_parseAppendNode;

    l->generateKernel = NULL;
    l->createKernelName = NULL;
    l->setVariables = NULL;
    l->printLayer = dR_wenergy2_printLayer;

    if (inputNode1)
    {
      // create empty list for previous nodes
      l->previous_layers = dR_list_createEmptyList();
      // append the input of this node to the list
      dR_list_append(l->previous_layers,inputNode1);
      // create empty list for following nodes
      l->next_layers = dR_list_createEmptyList();
      // append the current (wenergy2) node as the following node of the previous node
      dR_list_append(inputNode1->next_layers,l);
    }
    else
    {
        g_print("Error: wenergy2 node needs an appropriate Inputnode");
    }
    // append node to graph
    dR_appendLayer(net, l);
    // return pointer to node
    return l;
}

gchar* dR_wenergy2_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_Wenergy2_Data* wenergy2 = (dR_Wenergy2_Data*)(layer->layer);
    gchar* desc = "wenergy2";
    gint numNodeParams = 0;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("wenergy2 Node needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_wenergy2_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 1;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: wenergy2 Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: wenergy2 Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }

    out = dR_Wenergy2( net, iNodes[0] );
    return out;
}

gboolean dR_wenergy2_compute(dR_Graph* net, dR_Node* layer)
{
    dR_Wenergy2_Data* wenergy2  = ((dR_Wenergy2_Data*)(layer->layer));
    cl_mem* input1 = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    cl_mem output1 = (cl_mem)((void*)layer->outputBuf->buf);

    size_t globalWorkSize[3];
    int paramid = 0;
    dR_list_resetIt(layer->previous_layers);

    void *in = input1;
    //void *out = (void*)&wenergy2->intermed;

    cl_int key = 0;
    cl_int offset = 0;
    // whole image Energy summands

    globalWorkSize[0] = wenergy2->ishape.s0;
    globalWorkSize[1] = wenergy2->ishape.s1;
    globalWorkSize[2] = wenergy2->ishape.s2;

    net->clConfig->clError = clSetKernelArg(wenergy2->wenergy2All, 0, sizeof(cl_mem), in);

    if (dR_openCLError(net, "Setting kernel args failed.", "wenergy2All Kernel"))
        return FALSE;

    net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, wenergy2->wenergy2All, 3, NULL, globalWorkSize,
       NULL, 0, NULL, net->clConfig->clEvent);
    dR_finishCLKernel(net, "deepRACIN:wenergy2All");

    //*************************************************************
    // comment out parallel sum reduction
    #if 0
    // parallel sum reduction
    globalWorkSize[0] = wenergy2->ishape.s0;
    globalWorkSize[1] = wenergy2->ishape.s1;
    globalWorkSize[2] = wenergy2->ishape.s2;

    size_t localWorkSize[3];
    localWorkSize[0] = (size_t)wenergy2->localworksizex;
    localWorkSize[1] = (size_t)wenergy2->localworksizey;
    localWorkSize[2] = 1;

    size_t lMemSize = localWorkSize[0] * localWorkSize[1];

    // create buffer to store reduced sum
    // with size globalworksize/localworksize = nr of workgroups
    gint32 lws = localWorkSize[0]*localWorkSize[1];
    gint32 gws = globalWorkSize[0]*globalWorkSize[1];
    gint32 nrwg = gws/lws; // nr of workgroups

    gboolean ret = TRUE;
    ret &= dR_createFloatBuffer(net, &(wenergy2->intermed),nrwg*sizeof(gfloat), CL_MEM_READ_WRITE);

    if (!ret)
    {
        dR_openCLError(net, "Allocating memory for sum reduction failed.", "wenergy2Sum");
        return FALSE;
    }

    void * reducedSums = (void*)&wenergy2->intermed;

    net->clConfig->clError = clSetKernelArg(wenergy2->wenergy2Sum, 0, sizeof(cl_mem), in);
    net->clConfig->clError |= clSetKernelArg(wenergy2->wenergy2Sum, 1, sizeof(cl_mem), reducedSums);
    net->clConfig->clError |= clSetKernelArg(wenergy2->wenergy2Sum, 2, lMemSize * sizeof(cl_float), NULL);

    if (dR_openCLError(net, "Setting kernel args failed.", "wenergy2Sum"))
        return FALSE;

    net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, wenergy2->wenergy2Sum, 3, NULL, globalWorkSize,
        localWorkSize, 0, NULL, net->clConfig->clEvent);
    if (dR_openCLError(net, "Executing kernel failed.", "wenergy2Sum"))
        return FALSE;

    dR_finishCLKernel(net, "deepRACIN:wenergy2Sum");

    // download intermed for debug
    size_t numBytes = nrwg*sizeof(gfloat)*4;
    gfloat* dlarray = (gfloat*)g_malloc(nrwg*sizeof(cl_float));
    dR_downloadArray(net, "reducedSums", reducedSums, 0, numBytes, dlarray);
    gfloat wholeEnergy = 0.0f;
    //printf("\nArray Elements: ");
    for (int i = 0; i < nrwg; i++)
    {
        wholeEnergy += dlarray[i];
        //printf("%f, ", dlarray[i] );
    }
    //printf("\n\nSUM: %f\n\n", wholeEnergy );
    g_free(wenergy2->intermed);

    #endif

    // compute rest on CPU
    // download array from GPU to CPU
    int numBytes = wenergy2->ishape.s0*wenergy2->ishape.s1*sizeof(cl_float);
    cl_mem* wtEnergyArray;
    gfloat* out = (gfloat*)wenergy2->hostmem;
    wtEnergyArray = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    dR_downloadArray(net, "wtEnergyDownload", wtEnergyArray, 0, numBytes, out);

    // summing level 1:
    // H, V, D
    float H1=0.0f, D1=0.0f, V1=0.0f, H2=0.0f, D2=0.0f, V2=0.0f, H3=0.0f, D3=0.0f, V3=0.0f, Ea=0.0f;

    int indexH1=0, indexD1=0, indexV1=0, indexH2=0, indexD2=0, indexV2=0, indexH3=0, indexD3=0, indexV3=0, indexEa=0;

    int width = wenergy2->ishape.s0;
    int height = wenergy2->ishape.s1;

    int cWidth = wenergy2->ishape.s0/2;
    int cHeight = wenergy2->ishape.s1/2;

    float Ea1 = 0.0f, Ea2 = 0.0f, Ea3 = 0.0f;

    for (int i = 0; i < cHeight; i++)
    {
      for (int j = 0; j < cWidth; j++)
      {
        indexH1 = width*height/2 + width*i + j;
        H1 += out[indexH1];

        indexD1 = width*(height+1)/2 + width*i + j;
        D1 += out[indexD1];

        indexV1 = width/2 + width*i + j;
        V1 += out[indexV1];

        Ea1 += out[width*i + j];
      }
    }

    float sumAll = H1 + D1 + V1 + Ea1;
    //printf("\nSum all Energies lvl1: %f\n", sumAll);
    //printf("\nEa1: %f\n", Ea1);

    // sum level 2
    cWidth /= 2;
    cHeight /= 2;

    for (int i = 0; i < cHeight; i++)
    {
      for (int j = 0; j < cWidth; j++)
      {
        indexH2 = width*height/4 + width*i + j;
        H2 += out[indexH2];

        indexD2 = width*(height+1)/4 + width*i + j;
        D2 += out[indexD2];

        indexV2 = width/4 + width*i + j;
        V2 += out[indexV2];

        Ea2 += out[width*i + j];
      }
    }

    //float sumAll2 = H2 + D2 + V2 + Ea2;
    //printf("\nSum all Energies lvl2: %f\n", sumAll);
    //printf("\nEa2: %f\n", Ea2);

    // sum level 3
    cWidth /= 2;
    cHeight /= 2;

    for (int i = 0; i < cHeight; i++)
    {
      for (int j = 0; j < cWidth; j++)
      {
        indexH3 = width*height/8 + width*i + j;
        H3 += out[indexH3];

        indexD3 = width*(height+1)/8 + width*i + j;
        D3 += out[indexD3];

        indexV3 = width/8 + width*i + j;
        V3 += out[indexV3];

        Ea3 += out[width*i + j];
      }
    }

    //float sumAll3 = H3 + D3 + V3 + Ea3;
    //printf("\nSum all Energies lvl3: %f\n", sumAll);
    //printf("\nEa3: %f\n", Ea3);

    gfloat* features = wenergy2->feat;
    features[0] = (Ea3*100)/sumAll;
    features[1] = (H1*100)/sumAll;
    features[2] = (H2*100)/sumAll;
    features[3] = (H3*100)/sumAll;
    features[4] = (V1*100)/sumAll;
    features[5] = (V2*100)/sumAll;
    features[6] = (V3*100)/sumAll;
    features[7] = (D1*100)/sumAll;
    features[8] = (D2*100)/sumAll;
    features[9] = (D3*100)/sumAll;
    //for (int i = 0; i < 10; i++)
    //{
    //  features[i] = 1.0;
    //}
    // upload to GPU memory
    dR_uploadArray(net, "wtEnergyUpload", features, 0, 10*sizeof(cl_float), output1);

    printf("\n");
    for (int i = 0; i < 10; i++)
    {
      printf("out[%d] = %f\n",i, features[i]);//printf("out[%d] = %f\n", i, *(float*)(wenergy2->feat+i) );
    }
    printf("\n");

    // copy
    /*

    //    | EFn   Ea   EHn   EVn   EDn
    //----------------------------------
    //LVL1 | EF1   /    EH1   EV1   ED1
    //LVL2 | EF2   /    EH2   EV2   ED2
    //LVL3 | EF3   Ea   EH3   EV3   ED3
    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;

    net->clConfig->clError = clSetKernelArg(wenergy2->copyKernel, 0, sizeof(cl_mem), in);

    net->clConfig->clError |= clSetKernelArg(wenergy2->copyKernel, 1, sizeof(cl_mem), feat);

    if (dR_openCLError(net, "Setting kernel args failed.", "hwtcopy Kernel"))
        return FALSE;

    net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, wenergy2->copyKernel, 3, NULL, globalWorkSize,
       NULL, 0, NULL, net->clConfig->clEvent);
    dR_finishCLKernel(net, "deepRACIN:hwtcopy");
    */

    // Whole image sum of all energy summands
    /*
    net->clConfig->clError = clSetKernelArg(wenergy2->wenergy2Subset, 0, sizeof(cl_mem), in);

    net->clConfig->clError = clSetKernelArg(wenergy2->wenergy2Subset, 1, sizeof(cl_mem), feat);

    net->clConfig->clError = clSetKernelArg(wenergy2->wenergy2Subset, 2, sizeof(cl_int), (void *)&key);

    if (dR_openCLError(net, "Setting kernel args failed.", "wenergy2Subset Kernel"))
        return FALSE;

    net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, wenergy2->wenergy2Subset, 3, NULL, globalWorkSize,
       NULL, 0, NULL, net->clConfig->clEvent);
    dR_finishCLKernel(net, "deepRACIN:wenergySubset");
    */
    /*
    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;

    size_t localWorkSize[3];
    cl_int width = wenergy2->ishape.s0;
    cl_int height = wenergy2->ishape.s1;
    cl_int firststep = 1;
    size_t lMemSize;

    localWorkSize[0] = (size_t)wenergy2->localworksizex;
    localWorkSize[1] = (size_t)wenergy2->localworksizey;
    localWorkSize[2] = 1;
    */
    //***LVL1
    // EH1, left bottom

    // EV1, right top

    // ED1, diagonal

    //***LVL2

    //***LVL3
    return TRUE;
}


gboolean dR_wenergy2_schedule(dR_Graph* net, dR_Node* layer){
    dR_Wenergy2_Data* wenergy2  = ((dR_Wenergy2_Data*)(layer->layer));

    wenergy2->localworksizex = 8;
    wenergy2->localworksizey = 1;

    net = net;
    layer = layer;
    return TRUE;
 }

 gboolean dR_wenergy2_propagateShape(dR_Graph* net, dR_Node* layer)
 {
     dR_Wenergy2_Data* wenergy2 = (dR_Wenergy2_Data*)(layer->layer);
     dR_Node* lastlayer;

     if(layer->previous_layers->length!=1)
     {
         if(!net->config->silent)
             g_print("wenergy2 Node with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
         return FALSE;
     }
     dR_list_resetIt(layer->previous_layers); // to NULL
     // store last node
     lastlayer = dR_list_next(layer->previous_layers);

     // input shape of wenergy2 node is a wavelet coefficients
     wenergy2->ishape.s0 = lastlayer->oshape.s0;
     wenergy2->ishape.s1 = lastlayer->oshape.s1;
     wenergy2->ishape.s2 = lastlayer->oshape.s2;

     layer->oshape.s0 = 10;
     layer->oshape.s1 = 1;
     layer->oshape.s2 = 1;
     return TRUE;
 }

 gint32 dR_wenergy2_getRequiredOutputBufferSize(dR_Node* layer)
 {
     dR_Wenergy2_Data* wenergy2 = (dR_Wenergy2_Data*)(layer->layer);
     /*
       EF1: Energy of whole image
       EF2: Energy of image with w/2 h/2 in level 2 decimation
       EF3: Energy of image with w/4 h/4 in level 3 decimation

       EH1: Energy of horizontal decimation in level 1
       EV1: Energy of vertical decimation in level 1
       ED1: Energy of diagonal decimation in level 1

       EH2: Energy of horizontal decimation in level 2
       ...
       EH3: ...

       Ea: Energy of scaled down image with w/8 h/8

       Array is size 15

            | EFn   Ea   EHn   EVn   EDn
      ----------------------------------
       LVL1 | EF1   /    EH1   EV1   ED1
       LVL2 | EF2   /    EH2   EV2   ED2
       LVL3 | EF3   Ea   EH3   EV3   ED3
     */
     gint32 ret = 10;
     return ret;
 }

 gboolean dR_wenergy2_createKernel(dR_Graph* net, dR_Node* layer)
 {
     //call all Opencl kernel creation routines required
     dR_Wenergy2_Data* wenergy2 = (dR_Wenergy2_Data*)(layer->layer);
     gboolean ret = FALSE;
     ret = dR_createKernel(net,"wenergy2All",&(wenergy2->wenergy2All));
     ret = dR_createKernel(net,"wenergy2Sum",&(wenergy2->wenergy2Sum));
     //ret = dR_createKernel(net,"wenergy2Subset",&(wenergy2->wenergy2Subset));
     ret = dR_createKernel(net,"hwtcopy",&(wenergy2->copyKernel));
     return ret;
 }

 gboolean dR_wenergy2_allocateBuffers(dR_Graph* net, dR_Node* layer)
 {
     /* create buffer for intermediate steps */
     dR_Wenergy2_Data* wenergy2 = ((dR_Wenergy2_Data*)(layer->layer));
     gboolean ret = TRUE;
     dR_Shape3 shape = wenergy2->ishape;
     //wenergy2->energies = g_malloc(sizeof(cl_float)*15);
     //ret &= dR_createFloatBuffer(net, &(wenergy2->intermed),shape.s0*shape.s1*shape.s2*sizeof(gfloat), CL_MEM_READ_WRITE);
     wenergy2->hostmem = g_malloc(wenergy2->ishape.s0*wenergy2->ishape.s1*sizeof(cl_float));
     wenergy2->feat = g_malloc(10*sizeof(cl_float));
     return ret;
 }

 gboolean dR_wenergy2_fillBuffers(dR_Graph* net, dR_Node* layer)
 {
     net = net;
     layer = layer;
     return TRUE;
 }

 gboolean dR_wenergy2_cleanupBuffers(dR_Graph* net, dR_Node* layer)
 {
     gboolean ret = TRUE;
     if(net->prepared)
     {
         dR_Wenergy2_Data* wenergy2 = ((dR_Wenergy2_Data*)(layer->layer));
         //ret &= dR_clMemoryBufferCleanup(net, wenergy2->intermed);
         //ret &= dR_clMemoryBufferCleanup(net, wenergy2->feat);
         ret &= dR_cleanupKernel((layer->clKernel));
         ret &= dR_cleanupKernel((wenergy2->wenergy2All));
         ret &= dR_cleanupKernel((wenergy2->wenergy2Sum));
         ret &= dR_cleanupKernel((wenergy2->wenergy2Subset));
         ret &= dR_cleanupKernel((wenergy2->copyKernel));
     }
     return ret;
 }

 gboolean dR_wenergy2_cleanupLayer(dR_Graph* net, dR_Node* layer)
 {
     dR_Wenergy2_Data* wenergy2 = ((dR_Wenergy2_Data*)(layer->layer));
     // free all memory that was reserved for node
     if(net->prepared)
     {
         g_free(wenergy2->hostmem);
         g_free(wenergy2->intermed);
         g_free(wenergy2->feat);
         g_free(wenergy2);
     }
     return TRUE;
 }

 gchar* dR_wenergy2_printLayer(dR_Node* layer)
 {
     // print node
     dR_Wenergy2_Data* wenergy2 = (dR_Wenergy2_Data*)(layer->layer);
     gchar* out;
     out =  g_strdup_printf("%s%d%s", "wenergy2 operation node: ",layer->layerID, "\n");
     return out;
 }
