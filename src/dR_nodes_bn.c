#include "dR_nodes_bn.h"
#include "dR_core.h"
// Mandatory


dR_Node* dR_BatchNormalization(dR_Graph* net, dR_Node* inputLayer){
    dR_BN_Data* bn = g_malloc(sizeof(dR_BN_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    l->layer = bn;
    l->type = tBN;

    l->compute = dR_bn_compute;
    l->schedule = dR_bn_schedule;
    l->propagateShape = dR_bn_propagateShape;
    l->getRequiredOutputBufferSize = dR_bn_getRequiredOutputBufferSize;
    l->createKernel = dR_bn_createKernel;
    l->allocateBuffers = dR_bn_allocateBuffers;
    l->fillBuffers = dR_bn_fillBuffers;
    l->cleanupBuffers = dR_bn_cleanupBuffers;
    l->cleanupLayer = dR_bn_cleanupLayer;

    l->generateKernel = NULL;
    l->setVariables = NULL;
    l->createKernelName = NULL;
    l->printLayer = dR_bn_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: Conv2d needs a appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}


gboolean dR_bn_schedule(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    (void)net;
    (void)layer;
    return TRUE;
}

gboolean dR_bn_compute(dR_Graph* net, dR_Node* layer){ //todo
    // Nothing to do
    // Warnings shut up, please
    (void)net;
    (void)layer;
    //dR_BN_Data* lrnlayer = ((dR_BN_Data*)(layer->layer));
    /*cl_int filterWindowWidth = convlayer->shape.s0;
    cl_int filterWindowHeight = convlayer->shape.s1;
    cl_int filterWindowDepth = convlayer->shape.s2;
    //gint outputDepth = lrnlayer->shape.s3;
    cl_int iwidth = layer->ishape.s0;
    cl_int iheight = layer->ishape.s1;

    cl_int lMemSize;
    size_t localWorkSize[3];
    localWorkSize[0] = net->config->convLocalWorkSize;
    localWorkSize[1] = net->config->convLocalWorkSize;
    localWorkSize[2] = net->config->convLocalWorkSize;
    lMemSize = (localWorkSize[0] + filterWindowWidth - 1) * (localWorkSize[1] + filterWindowHeight - 1)* (localWorkSize[2] + filterWindowDepth - 1);

    size_t globalWorkSize[3];
    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = outputDepth;

    if ((filterWindowWidth-1)/2 > (cl_int)localWorkSize[0]
    || (filterWindowHeight-1) > (cl_int)localWorkSize[1]
    || (filterWindowDepth-1) > (cl_int)localWorkSize[2])
    {
        g_print("Error: Kernel size for convolution is too big (is %ix%i; max %ix%i; approx. max sigma %f)\n", filterWindowWidth, filterWindowHeight, (gint)(2*localWorkSize[0]+1), (gint)(localWorkSize[1]+1), (gfloat)((2.0f*((gfloat)localWorkSize[0])+1.0f)/2.0f/2.0f)-0.5f);
        return;
    }*/

    /*
    net->clConfig->clError = clSetKernelArg(layer->clKernel, 0, sizeof(cl_mem), (void *)&input);
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 1, sizeof(cl_mem), (void *)&convlayer->weightsBuf);
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 2, sizeof(cl_mem), (void *)&convlayer->biasBuf);
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 3, sizeof(cl_mem), (void *)&layer->outputBuf);
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 4, lMemSize * sizeof(cl_float), NULL); //local memory
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 5, sizeof(cl_int), (void *)&iwidth);
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 6, sizeof(cl_int), (void *)&iheight);
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 7, sizeof(cl_int), (void *)&filterWindowDepth);
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 8, sizeof(cl_int), (void *)&filterWindowWidth);
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 9, sizeof(cl_int), (void *)&filterWindowHeight);
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, 10, sizeof(cl_int), (void *)&filterWindowDepth);
    */

   /* if (openCLError(net, "Setting kernel args failed.", "Conv2d Kernel"))
        return;
    //execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 2, NULL, globalWorkSize,
        localWorkSize, 0, NULL, net->clConfig->clEvent);

    submitClKernel(net, "conv2dReLU");*/
    return TRUE;
}

gboolean dR_bn_propagateShape(dR_Graph* net, dR_Node* layer)
{
    //dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("LRN Layer with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    return TRUE;
}

gint32 dR_bn_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_bn_createKernel(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    (void)net;
    (void)layer;
    return TRUE;
}

gboolean dR_bn_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    (void)net;
    (void)layer;
    return TRUE;
}

gboolean dR_bn_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    (void)net;
    (void)layer;
    return TRUE;
}

gboolean dR_bn_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    (void)net;
    (void)layer;
    return TRUE;
}

gboolean dR_bn_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
        g_free((dR_BN_Data*)(layer->layer));
    return TRUE;
}

gchar* dR_bn_printLayer(dR_Node* layer)
{
    dR_BN_Data* bnlayer = (dR_BN_Data*)(layer->layer);
    gchar* out;
    out = g_strdup_printf("%s%s%f%s%f%s%f%s",
            "Local BatchNormalization Layer:\n ",
            "\n Alpha: ", bnlayer->alpha,
            "\n Beta: ",bnlayer->beta,
            "\n Bias: ", bnlayer->bias,"\n");

    return out;
}
