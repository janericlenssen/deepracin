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
