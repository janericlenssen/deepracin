#include "dR_nodes_misc.h"
#include "dR_core.h"
#include "math.h"


// //////////////
// Resolve RoI //
// //////////////

dR_Node* dR_ResolveRoI(dR_Graph* net, dR_Node* inputLayer, dR_Shape3 shape){
    dR_ResolveRoI_Data* resolveRoI = g_malloc(sizeof(dR_ResolveRoI_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    l->layer = resolveRoI;
    l->type = tResolveRoI;
    resolveRoI->shape = shape;

    l->compute = dR_resolveRoI_compute;
    l->schedule = dR_resolveRoI_schedule;
    l->propagateShape = dR_resolveRoI_propagateShape;
    l->getRequiredOutputBufferSize = dR_resolveRoI_getRequiredOutputBufferSize;
    l->createKernel = dR_resolveRoI_createKernel;
    l->allocateBuffers = dR_resolveRoI_allocateBuffers;
    l->fillBuffers = dR_resolveRoI_fillBuffers;
    l->cleanupBuffers = dR_resolveRoI_cleanupBuffers;
    l->cleanupLayer = dR_resolveRoI_cleanupLayer;
    l->serializeNode = dR_resolveRoI_serializeNode;
    l->parseAppendNode = dR_resolveRoI_parseAppendNode;

    l->generateKernel = NULL;
    l->setVariables = NULL;
    l->createKernelName = NULL;
    l->printLayer = dR_resolveRoI_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: ResolveRoI Layer needs an appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_resolveRoI_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_ResolveRoI_Data* resolveroi = (dR_ResolveRoI_Data*)(layer->layer);
    gchar* desc = "ResolveRoI";
    gint numNodeParams = 3;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",resolveroi->shape.s0);
    params[1] = g_strdup_printf("%d",resolveroi->shape.s1);
    params[2] = g_strdup_printf("%d",resolveroi->shape.s2);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_resolveRoI_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 3;
    gint numNodeVariables = 0;
    dR_Node* out;
    dR_Shape3 shape;
    if(numINodes!=1)
    {
        g_print("Parsing Error: ResolveRoI Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: ResolveRoI Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    shape.s0 = atoi(params[0]);
    shape.s1 = atoi(params[1]);
    shape.s2 = atoi(params[2]);
    out = dR_ResolveRoI(net, iNodes[0], shape);
    return out;
}

gboolean dR_resolveRoI_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }


gboolean dR_resolveRoI_compute(dR_Graph* net, dR_Node* layer){
    dR_ResolveRoI_Data* resolveRoI = (dR_ResolveRoI_Data*)(layer->layer);
    size_t globalWorkSize[3];
    gint paramid = 0;
    cl_mem* input;
    dR_Shape3 origin;
    gint buforigin;
    gboolean ret;
    gint bufwidth, bufheight, bufdepth;
    dR_Node* inputlayer;
    dR_list_resetIt(layer->previous_layers);
    inputlayer = ((dR_Node*)dR_list_next(layer->previous_layers));
    if(!inputlayer->outputBuf->regionOfInterest)
    {
        g_print("In order to apply RoI resolve, the previous node has to have a defined RoI! Use dR_setNodeRoI(...) on it! \n");
        return FALSE;
    }
    if(!inputlayer->outputBuf->useIndexBufferForRoI)
    {

        input = inputlayer->outputBuf->bufptr;
        globalWorkSize[0] = layer->oshape.s0;
        globalWorkSize[1] = layer->oshape.s1;
        globalWorkSize[2] = layer->oshape.s2;

        origin = inputlayer->outputBuf->roiOrigin;
        bufwidth = inputlayer->oshape.s0;
        bufheight = inputlayer->oshape.s1;
        bufdepth = inputlayer->oshape.s2;
        // ClamppRoI
        if(origin.s0<0)
            origin.s0=0;
        if(origin.s1<0)
            origin.s1=0;
        if(origin.s2<0)
            origin.s2=0;
        if(origin.s0>=bufwidth-layer->oshape.s0)
            origin.s0=bufwidth-layer->oshape.s0;
        if(origin.s1>=bufheight-layer->oshape.s1)
            origin.s1=bufheight-layer->oshape.s1;
        if(origin.s2>=bufdepth-layer->oshape.s2)
            origin.s2=bufdepth-layer->oshape.s2;
        buforigin = origin.s2*bufwidth*bufheight+ origin.s1*bufwidth + origin.s0;

        net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                         paramid++;
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);     paramid++;
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&bufwidth);                    paramid++;
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&bufheight);                   paramid++;
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&bufdepth);                   paramid++;
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&(origin.s0));                 paramid++;
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&(origin.s1));                 paramid++;
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&(origin.s2));                 paramid++;

        g_print("Applying resolveRoI: Input: %dx%d, Origin: %dx%dx%d - %d Output: %dx%dx%d \n",bufwidth, bufheight, origin.s0, origin.s1, origin.s2, buforigin, layer->oshape.s0,layer->oshape.s1,layer->oshape.s2);

        if (dR_openCLError(net, "Setting kernel args failed.", "ResolveRoI Kernel"))
            return FALSE;
        // execute kernel
         net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
            NULL, 0, NULL, net->clConfig->clEvent);

        ret = dR_finishCLKernel(net, "deepRACIN:ResolveRoI");
    }
    else
    {
        cl_mem* indexBuffer;
        cl_int index = inputlayer->outputBuf->index;
        input = inputlayer->outputBuf->bufptr;
        globalWorkSize[0] = layer->oshape.s0;
        globalWorkSize[1] = layer->oshape.s1;
        globalWorkSize[2] = layer->oshape.s2;
        indexBuffer = inputlayer->outputBuf->indexBuffer;
        bufwidth = inputlayer->oshape.s0;
        bufheight = inputlayer->oshape.s1;
        bufdepth = inputlayer->oshape.s2;


        net->clConfig->clError = clSetKernelArg(resolveRoI->clKernelRoIFromBufferIndex, paramid, sizeof(cl_mem), (void *)input);                       paramid++;
        net->clConfig->clError |= clSetKernelArg(resolveRoI->clKernelRoIFromBufferIndex, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);   paramid++;
        net->clConfig->clError |= clSetKernelArg(resolveRoI->clKernelRoIFromBufferIndex, paramid, sizeof(cl_mem), (void *)indexBuffer);                paramid++;
        net->clConfig->clError |= clSetKernelArg(resolveRoI->clKernelRoIFromBufferIndex, paramid, sizeof(cl_int), (void *)&index);                     paramid++;
        net->clConfig->clError |= clSetKernelArg(resolveRoI->clKernelRoIFromBufferIndex, paramid, sizeof(cl_int), (void *)&bufwidth);                  paramid++;
        net->clConfig->clError |= clSetKernelArg(resolveRoI->clKernelRoIFromBufferIndex, paramid, sizeof(cl_int), (void *)&bufheight);                 paramid++;
        net->clConfig->clError |= clSetKernelArg(resolveRoI->clKernelRoIFromBufferIndex, paramid, sizeof(cl_int), (void *)&bufdepth);                 paramid++;

        if (dR_openCLError(net, "Setting kernel args failed.", "RoIFromBufferIndex Kernel"))
            return FALSE;
        // execute kernel
         net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, resolveRoI->clKernelRoIFromBufferIndex, 3, NULL, globalWorkSize,
            NULL, 0, NULL, net->clConfig->clEvent);

        ret = dR_finishCLKernel(net, "deepRACIN:RoIFromBufferIndex");
    }
    return ret;
}

gboolean dR_resolveRoI_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_ResolveRoI_Data* resolveRoI = ((dR_ResolveRoI_Data*)(layer->layer));
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("Resolve RoI Layer with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);

    resolveRoI->ishape = lastlayer->oshape;

    layer->oshape = resolveRoI->shape;

    if(resolveRoI->ishape.s2<layer->oshape.s2||resolveRoI->ishape.s1<layer->oshape.s1||resolveRoI->ishape.s0<layer->oshape.s0)
    {
        if(!net->config->silent)
            g_print("ResolveRoI Layer input needs to be equal or bigger than ResolveRoI Layer output!\n");
        return FALSE;
    }

    return TRUE;
}

gint32 dR_resolveRoI_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_resolveRoI_createKernel(dR_Graph* net, dR_Node* layer)
{
    dR_ResolveRoI_Data* resolveRoI = ((dR_ResolveRoI_Data*)(layer->layer));
    gboolean ret = TRUE;
    ret &= dR_createKernel(net,"resolveRoI",&(layer->clKernel));
    ret &= dR_createKernel(net,"resolveRoIIndexBuffer",&(resolveRoI->clKernelRoIFromBufferIndex));
    return ret;
}


gboolean dR_resolveRoI_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_resolveRoI_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_resolveRoI_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    dR_ResolveRoI_Data* resolveRoI = ((dR_ResolveRoI_Data*)(layer->layer));
    gboolean ret = FALSE;
    if(net->prepared)
    {
        ret &= dR_cleanupKernel((layer->clKernel));
        ret &= dR_cleanupKernel((resolveRoI->clKernelRoIFromBufferIndex));
    }
    return ret;
}

gboolean dR_resolveRoI_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
        g_free((dR_ResolveRoI_Data*)(layer->layer));
    return TRUE;
}


gchar* dR_resolveRoI_printLayer(dR_Node* layer)
{
    dR_ResolveRoI_Data* resolveRoI = ((dR_ResolveRoI_Data*)(layer->layer));
    gchar* out;
    out = g_strdup_printf("%s%d%s%d%s%d%s%d%s",
            "ResolveRoI Layer: ",layer->layerID,"\n Size: ",resolveRoI->shape.s0,"x",resolveRoI->shape.s1,"x",resolveRoI->shape.s2,"\n");

    return out;
}




// ///////////////////
// RGB to Grayscale //
// ///////////////////

dR_Node* dR_RGB2gray(dR_Graph* net, dR_Node* inputLayer){
    dR_RGB2Gray_Data* rgb2gray = g_malloc(sizeof(dR_RGB2Gray_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    l->layer = rgb2gray;
    l->type = tRGB2Gray;

    l->compute = dR_rgb2gray_compute;
    l->schedule = dR_rgb2gray_schedule;
    l->propagateShape = dR_rgb2gray_propagateShape;
    l->getRequiredOutputBufferSize = dR_rgb2gray_getRequiredOutputBufferSize;
    l->createKernel = dR_rgb2gray_createKernel;
    l->allocateBuffers = dR_rgb2gray_allocateBuffers;
    l->fillBuffers = dR_rgb2gray_fillBuffers;
    l->cleanupBuffers = dR_rgb2gray_cleanupBuffers;
    l->cleanupLayer = dR_rgb2gray_cleanupLayer;
    l->serializeNode = dR_rgb2gray_serializeNode;
    l->parseAppendNode = dR_rgb2gray_parseAppendNode;

    l->generateKernel = NULL;
    l->setVariables = NULL;
    l->createKernelName = NULL;
    l->printLayer = dR_rgb2gray_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: RGB to Grayscale Layer needs an appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_rgb2gray_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_ResolveRoI_Data* rgb2gray = (dR_ResolveRoI_Data*)(layer->layer);
    gchar* desc = "RGB2Gray";
    gint numNodeParams = 0;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_rgb2gray_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 0;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: RGB2Gray Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: RGB2Gray Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    out = dR_RGB2gray(net, iNodes[0]);
    return out;
}

gboolean dR_rgb2gray_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }


gboolean dR_rgb2gray_compute(dR_Graph* net, dR_Node* layer){
    //dR_RGB2Gray_Data* rgb2gray = (dR_RGB2Gray_Data*)(layer->layer);
    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = 1;


    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);          paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "RGB 2 Gray Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:RGB2Gray");

}

gboolean dR_rgb2gray_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_RGB2Gray_Data* rgb2gray = ((dR_RGB2Gray_Data*)(layer->layer));
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("RGB to Grayscale Layer with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    rgb2gray->ishape.s0 = lastlayer->oshape.s0;
    rgb2gray->ishape.s1 = lastlayer->oshape.s1;
    rgb2gray->ishape.s2 = lastlayer->oshape.s2;
    if(rgb2gray->ishape.s2!=3)
    {
        if(!net->config->silent)
            g_print("RGB to Grayscale Layer input needs depth 3 but has depth %d\n",rgb2gray->ishape.s2);
        return FALSE;
    }
    layer->oshape.s0 = rgb2gray->ishape.s0;
    layer->oshape.s1 = rgb2gray->ishape.s1;
    layer->oshape.s2 = 1;
    return TRUE;
}

gint32 dR_rgb2gray_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_rgb2gray_createKernel(dR_Graph* net, dR_Node* layer)
{
    return dR_createKernel(net,"RGB2Gray",&(layer->clKernel));
}


gboolean dR_rgb2gray_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_rgb2gray_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_rgb2gray_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = FALSE;
    if(net->prepared)
        ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_rgb2gray_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
        g_free((dR_RGB2Gray_Data*)(layer->layer));
    return TRUE;
}


gchar* dR_rgb2gray_printLayer(dR_Node* layer)
{
    gchar* out;
    out = g_strdup_printf("%s%d%s",
            "RGB to Grayscale Layer: ",layer->layerID,"\n");

    return out;
}








// //////////////////
// Upscaling Layer //
// //////////////////

dR_Node* dR_Upscaling(dR_Graph* net, dR_Node* inputLayer, dR_UpscalingType type, gint scaleFactorX, gint scaleFactorY){
    dR_Upscaling_Data* us = g_malloc(sizeof(dR_Upscaling_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));

    us->type = type;
    us->scaleFactorX = scaleFactorX;
    us->scaleFactorY = scaleFactorY;
    l->layer = us;
    l->type = tUpscaling;

    l->compute = dR_us_compute;
    l->schedule = dR_us_schedule;
    l->propagateShape = dR_us_propagateShape;
    l->getRequiredOutputBufferSize = dR_us_getRequiredOutputBufferSize;
    l->createKernel = dR_us_createKernel;
    l->allocateBuffers = dR_us_allocateBuffers;
    l->fillBuffers = dR_us_fillBuffers;
    l->cleanupBuffers = dR_us_cleanupBuffers;
    l->cleanupLayer = dR_us_cleanupLayer;
    l->serializeNode = dR_us_serializeNode;
    l->parseAppendNode = dR_us_parseAppendNode;

    l->generateKernel = dR_us_generateKernel;
    l->setVariables = NULL;
    l->createKernelName = dR_us_createKernelName;
    l->printLayer = dR_us_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: Upscaling Layer needs an appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_us_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_Upscaling_Data* upscaling = (dR_Upscaling_Data*)(layer->layer);
    gchar* desc = "Upscaling";
    gint numNodeParams = 3;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",upscaling->type);
    params[1] = g_strdup_printf("%d",upscaling->scaleFactorX);
    params[2] = g_strdup_printf("%d",upscaling->scaleFactorY);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_us_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 3;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: Upscaling Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: Upscaling Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    out = dR_Upscaling(net, iNodes[0], atoi(params[0]),atoi(params[1]),atoi(params[2]));
    return out;
}

gboolean dR_us_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }


gboolean dR_us_compute(dR_Graph* net, dR_Node* layer){
    dR_Upscaling_Data* uslayer = (dR_Upscaling_Data*)(layer->layer);
    size_t globalWorkSize[3];
    size_t localWorkSize[3];
    gint lMemImageSize;
    int paramid = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;

    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;

    if(uslayer->scaleFactorX<16)
        localWorkSize[0] = uslayer->scaleFactorX;
    else
        localWorkSize[0] = 16;
    if(uslayer->scaleFactorY<16)
        localWorkSize[1] = uslayer->scaleFactorY;
    else
        localWorkSize[1] = 16;
    localWorkSize[2] = 1;

    if(uslayer->type==tNearestNUp)
        lMemImageSize = 1;
    else
        lMemImageSize = 4;


    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemImageSize * sizeof(cl_float), NULL);				paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);          paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "LabelCreation Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        localWorkSize, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:upscaling");

}

gboolean dR_us_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_Upscaling_Data* uslayer = (dR_Upscaling_Data*)(layer->layer);
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("Upscaling Layer with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    uslayer->ishape = lastlayer->oshape;

    layer->oshape.s0 = uslayer->ishape.s0*uslayer->scaleFactorX;
    layer->oshape.s1 = uslayer->ishape.s1*uslayer->scaleFactorY;
    layer->oshape.s2 = uslayer->ishape.s2;
    return TRUE;
}

gint32 dR_us_getRequiredOutputBufferSize(dR_Node* layer)
{
    //dR_Upscaling_Data* uslayer = (dR_Upscaling_Data*)(layer->layer);
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_us_createKernel(dR_Graph* net, dR_Node* layer)
{
    //dR_Upscaling_Data* uslayer = (dR_Upscaling_Data*)(layer->layer);
    return dR_createKernel(net,dR_us_createKernelName(layer),&(layer->clKernel));
}


gboolean dR_us_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_us_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_us_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = FALSE;
    if(net->prepared)
        ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_us_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
        g_free((dR_Upscaling_Data*)(layer->layer));
    return TRUE;
}

gboolean dR_us_generateKernel(dR_Graph* net, dR_Node* layer)
{
    dR_Upscaling_Data* uslayer = (dR_Upscaling_Data*)(layer->layer);
	gint length;
    //Creating Shader Name and Return type
    gchar* source, *temp;
    gchar* kernelname = dR_us_createKernelName(layer);
	gchar * filePath;

    int scaleFactorX = uslayer->scaleFactorX;
    int scaleFactorY = uslayer->scaleFactorY;
    int iwidth = uslayer->ishape.s0;
    int iheight = uslayer->ishape.s1;


    source = g_strdup_printf(
    "#ifndef kerneldef_%s \n#define kerneldef_%s \n\n__kernel void %s(\n",kernelname,kernelname,kernelname);


    // Variables that get hardcoded into the shader


    // Rest of the Header and first part
    temp = g_strdup_printf(
    "    const __global  float * gInputImage,\n"
    "    __local float* lImage,\n"
    "    __global  float * gOutputImage\n"
    "    )\n"
    "{\n");
    source = concat_and_free_old(source,temp);
    if(uslayer->type==tNearestNUp)
    {
        temp = g_strdup_printf(
         "    if(get_local_id(0)==0&&get_local_id(1)==0) \n"
        "    {\n"
        "        const int copyIndex = get_global_id(2)*%d*%d +(get_global_id(1)/%d)*%d + get_global_id(0)/%d;\n"
        "        lImage[0] = gInputImage[copyIndex];\n"
        "    }\n", iwidth, iheight, scaleFactorX, iwidth, scaleFactorY);
        source = concat_and_free_old(source,temp);

        temp = g_strdup_printf(
        "    barrier(CLK_LOCAL_MEM_FENCE); \n"
        "    const int writeIndex = get_global_id(2)*get_global_size(1)*get_global_size(0) + get_global_id(1)*get_global_size(0) + get_global_id(0);\n"
        "    gOutputImage[writeIndex] = lImage[0];");
        source = concat_and_free_old(source,temp);

    }
    temp = g_strdup_printf(
    "} \n"
    "#endif \n");
    source = concat_and_free_old(source,temp);


    length = strlen(source);

    filePath = g_build_filename(net->config->modelPath, dR_us_createKernelName(layer), NULL);
    g_file_set_contents(filePath,source,length,NULL);

    g_free(source);

    return TRUE;
}

gchar* dR_us_createKernelName(dR_Node* layer)
{
    dR_Upscaling_Data* uslayer = (dR_Upscaling_Data*)(layer->layer);
    gint length = 0;
	gint maxsize = 100;
    gchar* string = g_malloc(sizeof(char)*maxsize);
    length += g_snprintf(string+length,maxsize-length,"upscale%dx%d", uslayer->scaleFactorX, uslayer->scaleFactorY);
    switch(uslayer->type){
    case tNearestNUp:
        length += g_snprintf(string+length,maxsize-length,"NearestN");
        break;
    case tLinearUp:
        length += g_snprintf(string+length,maxsize-length,"Linear");
        break;
    case tBicubicUp:
        length += g_snprintf(string+length,maxsize-length,"Bicubic");
        break;
    }
    return string;
}

gchar* dR_us_printLayer(dR_Node* layer)
{
    dR_Upscaling_Data* uslayer = (dR_Upscaling_Data*)(layer->layer);
    gchar* out;
    gchar* typestr;
    if(uslayer->type == tLinearUp)
        typestr = "Bilinear";
    else if(uslayer->type == tBicubicUp)
        typestr = "Bicubic";
    else if(uslayer->type == tNearestNUp)
        typestr = "Nearest Neighbor";

    out = g_strdup_printf("%s%d%s%s%s%d%s%d%s",
            "Upscaling Layer: ",layer->layerID,
            "\n Type: ", typestr,
            "\n Scale Factor X: ", uslayer->scaleFactorX,
            "\n Scale Factor Y: ", uslayer->scaleFactorY,"\n");

    return out;
}


// /////////////////
// Label Creation //
// /////////////////

dR_Node* dR_LabelCreation(dR_Graph* net, dR_Node* inputLayer, dR_LabelCreationType type, gfloat class0offset, gfloat class1offset, gfloat class2offset){
    dR_LabelCreation_Data* lc = g_malloc(sizeof(dR_LabelCreation_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));

    lc->type = type;
    lc->class0offset = class0offset;
    lc->class1offset = class1offset;
    lc->class2offset = class2offset;
    l->layer = lc;
    l->type = tLabelCreation;

    // Mandatory
    l->compute = dR_lc_compute;
    l->schedule = dR_lc_schedule;
    l->propagateShape = dR_lc_propagateShape;
    l->getRequiredOutputBufferSize = dR_lc_getRequiredOutputBufferSize;
    l->createKernel = dR_lc_createKernel;
    l->allocateBuffers = dR_lc_allocateBuffers;
    l->fillBuffers = dR_lc_fillBuffers;
    l->cleanupBuffers = dR_lc_cleanupBuffers;
    l->cleanupLayer = dR_lc_cleanupLayer;
    l->serializeNode = dR_lc_serializeNode;
    l->parseAppendNode = dR_lc_parseAppendNode;

    // Optional
    l->generateKernel = NULL;
    l->setVariables = NULL;
    l->createKernelName = NULL;
    l->printLayer = dR_lc_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: Label Creation Layer needs an appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_lc_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_LabelCreation_Data* labelcreation = (dR_LabelCreation_Data*)(layer->layer);
    gchar* desc = "LabelCreation";
    gint numNodeParams = 1;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",labelcreation->type);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_lc_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 1;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: LabelCreation Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: LabelCreation Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    out = dR_LabelCreation(net, iNodes[0], atoi(params[0]),0.0,0.0,0.0);
    return out;
}

gboolean dR_lc_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }


gboolean dR_lc_compute(dR_Graph* net, dR_Node* layer){

    dR_LabelCreation_Data* lclayer = (dR_LabelCreation_Data*)(layer->layer);
    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;

    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = 1;

    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);								paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);             paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_float), (void *)&lclayer->class0offset);           paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_float), (void *)&lclayer->class1offset);           paramid++;
    if(lclayer->type == t3to2ClassesBin||lclayer->type== t3to2ClassesConf)
    {
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_float), (void *)&lclayer->class2offset);       paramid++;
    }

    if (dR_openCLError(net, "Setting kernel args failed.", "LabelCreation Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:labelcreation");

}

gboolean dR_lc_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_LabelCreation_Data* lclayer = ((dR_LabelCreation_Data*)(layer->layer));
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("Label Creation Layer with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    lclayer->ishape.s0 = lastlayer->oshape.s0;
    lclayer->ishape.s1 = lastlayer->oshape.s1;
    lclayer->ishape.s2 = lastlayer->oshape.s2;

    layer->oshape.s0 = lclayer->ishape.s0;
    layer->oshape.s1 = lclayer->ishape.s1;
    layer->oshape.s2 = 1;
    return TRUE;
}

gint32 dR_lc_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1;
}

gboolean dR_lc_createKernel(dR_Graph* net, dR_Node* layer)
{
    dR_LabelCreation_Data* lclayer = (dR_LabelCreation_Data*)(layer->layer);
    gchar* kernelname;
    switch(lclayer->type)
    {
    case t3to2ClassesBin:
        kernelname = "toLabelImage3to2ClassesBin";
        break;
    case t3to2ClassesConf:
        kernelname = "toLabelImage3to2ClassesConf";
        break;
    case t2ClassesBin:
        kernelname = "toLabelImage2ClassesBin";
        break;
    case t2ClassesConf:
        kernelname = "toLabelImage2ClassesConf";
        break;
    case tnClasses:
        kernelname = "";
        break;
    }
    return dR_createKernel(net,kernelname,&(layer->clKernel));
}


gboolean dR_lc_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_lc_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_lc_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    if(net->prepared)
        ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_lc_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
        g_free((dR_LabelCreation_Data*)(layer->layer));
    return TRUE;
}

gchar* dR_lc_printLayer(dR_Node* layer)
{
    dR_LabelCreation_Data* lclayer = (dR_LabelCreation_Data*)(layer->layer);
    gchar* out;
    gchar* typestr;
    if(lclayer->type == t3to2ClassesBin)
        typestr = "3 to 2 Classes, Binary (c=1|2 / c=0) Labels";
    else if(lclayer->type == t3to2ClassesConf)
        typestr = "3 to 2 Classes, Confidence (c=1|c=2) Labels";
    else if(lclayer->type == t2ClassesBin)
        typestr = "2 Classes, Binary Labels";
    else if(lclayer->type == t2ClassesConf)
        typestr = "2 Classes, Confidence (c=1) Labels";
    else if(lclayer->type == tnClasses)
        typestr = "n Classes, Integer Labels";

    out = g_strdup_printf("%s%d%s%s%s",
            "Label Creation Layer: ",layer->layerID,
            "\n Type: ", typestr,"\n");

    return out;
}



// /////////////////
// Normalization  //
// /////////////////


dR_Node* dR_Normalization(dR_Graph* net, dR_Node* inputLayer, dR_NormalizationType type, gfloat targetMean, gfloat targetDev){
    dR_Normalization_Data* norm = g_malloc(sizeof(dR_Normalization_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    norm->type = type;
    norm->targetMean = targetMean;
    norm->targetDev = targetDev;
    l->layer = norm;
    l->type = tNormalization;

    // Mandatory
    l->compute = dR_norm_compute;
    l->schedule = dR_norm_schedule;
    l->propagateShape = dR_norm_propagateShape;
    l->getRequiredOutputBufferSize = dR_norm_getRequiredOutputBufferSize;
    l->createKernel = dR_norm_createKernel;
    l->allocateBuffers = dR_norm_allocateBuffers;
    l->fillBuffers = dR_norm_fillBuffers;
    l->cleanupBuffers = dR_norm_cleanupBuffers;
    l->cleanupLayer = dR_norm_cleanupLayer;
    l->serializeNode = dR_norm_serializeNode;
    l->parseAppendNode = dR_norm_parseAppendNode;

    // Optional
    l->generateKernel = NULL;
    l->setVariables = NULL;
    l->createKernelName = NULL;
    l->printLayer = dR_norm_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: Normalization Layer needs an appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}


gchar* dR_norm_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_Normalization_Data* normnode = (dR_Normalization_Data*)(layer->layer);
    gchar* desc = "Normalization";
    gint numNodeParams = 3;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",normnode->type);
    params[1] = g_strdup_printf("%f",normnode->targetMean);
    params[2] = g_strdup_printf("%f",normnode->targetDev);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_norm_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 3;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: Normalization Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: Normalization Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    out = dR_Normalization(net, iNodes[0], atoi(params[0]), (gfloat)atof(params[1]), (gfloat)atof(params[2]));
    return out;
}

gboolean dR_norm_schedule(dR_Graph* net, dR_Node* layer){
    dR_Normalization_Data* normlayer = (dR_Normalization_Data*)(layer->layer);

    // Todo: Could be chose more dynamic. But is very fast anyway.
    normlayer->localworksizex = 16;
    normlayer->localworksizey = 8;

    if(layer->oshape.s0%normlayer->localworksizex==0)
    {
        normlayer->globalWorkSizeX = layer->oshape.s0;
    }
    else
    {
        normlayer->globalWorkSizeX = ((layer->oshape.s0/normlayer->localworksizex)+1)*normlayer->localworksizex;
    }


    if(layer->oshape.s1%normlayer->localworksizey==0)
    {
        normlayer->globalWorkSizeY = layer->oshape.s1;
    }
    else
    {
        normlayer->globalWorkSizeY = ((layer->oshape.s1/normlayer->localworksizey)+1)*normlayer->localworksizey;
    }
    if(!net->config->silent)
    {
        g_print("Normlayer LocalWorkSizeX/Y: %d, %d\n",normlayer->localworksizex,normlayer->localworksizey);
    }
    return TRUE;
 }


gboolean dR_norm_compute(dR_Graph* net, dR_Node* layer){
    dR_Normalization_Data* normlayer = (dR_Normalization_Data*)(layer->layer);
    cl_float gdev = 1.0f;
    cl_float gavg = 0.0f;
	cl_float targetMean, targetDev;
    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;

    if(normlayer->type == tNormMean || normlayer->type == tNormMeanDev || normlayer->type == tNormDev)
    {
        // Reduce Average
        size_t globalWorkSize[3];
        size_t localWorkSize[3];
        cl_int nGroupsPerChannel;
        cl_int nGroupsX;
        cl_int nGroupsY;
        cl_int width = normlayer->ishape.s0;
        cl_int height = normlayer->ishape.s1;
        cl_int depth = normlayer->ishape.s2;

        cl_int gidOfLastValidElement =  (normlayer->ishape.s1-1) * normlayer->ishape.s0 + normlayer->ishape.s0 - 1;
        cl_mem* inputAvg = input;
        cl_mem* resultSum = &normlayer->tempFloatBuffer1;
        gboolean resultIsInFirstDeviceMem = FALSE; //will be changed to TRUE on first run
		size_t lMemSize;

		gfloat sum;
        int i;

        localWorkSize[0] = (size_t)normlayer->localworksizex;
        localWorkSize[1] = (size_t)normlayer->localworksizey;
        localWorkSize[2] = 1;
        lMemSize = localWorkSize[0] * localWorkSize[1];
        globalWorkSize[0] = (size_t)normlayer->globalWorkSizeX;
        globalWorkSize[1] = (size_t)normlayer->globalWorkSizeY;
        globalWorkSize[2] = normlayer->ishape.s2;
        nGroupsX = (cl_int)(globalWorkSize[0] / localWorkSize[0]);
        nGroupsY = (cl_int)(globalWorkSize[1] / localWorkSize[1]);
        nGroupsPerChannel = nGroupsX * nGroupsY;

        while (nGroupsPerChannel >= 1)
        {
            resultIsInFirstDeviceMem = ! resultIsInFirstDeviceMem;

            net->clConfig->clError = clSetKernelArg(normlayer->clKernelGetAvg, 0, sizeof(cl_mem), (void *)inputAvg);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetAvg, 1, sizeof(cl_mem), (void *)resultSum);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetAvg, 2, lMemSize * sizeof(cl_float), NULL);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetAvg, 3, sizeof(cl_int), (void *)&width);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetAvg, 4, sizeof(cl_int), (void *)&height);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetAvg, 5, sizeof(cl_int), (void *)&gidOfLastValidElement);

            if (dR_openCLError(net, "Setting kernel args failed.", "clKernelGetAvg"))
                return FALSE;

            // execute kernel
            net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, normlayer->clKernelGetAvg, 3 /* number of dimensions used*/, NULL, globalWorkSize,
                localWorkSize, 0, NULL, net->clConfig->clEvent);
            if (dR_openCLError(net, "Executing kernel failed.", "clKernelGetAvg"))
                return FALSE;

            dR_finishCLKernel(net, "libnnapp::clKernelGetAvg");

            //Update variables:

            //Last valid element equals to the previous number of groups, as each group produces one output.
            gidOfLastValidElement = nGroupsPerChannel -1;
            //Global work size must be a multiple of local work size.
            //Uses previous number of groups (x and y)
            //E.g. if previous number of group is 20x10 (200 groups) and local work size is 16x8 new global size will be 32x16
            globalWorkSize[0] = (size_t)(nGroupsX + (localWorkSize[0] - 1) - (nGroupsX + (localWorkSize[0] - 1)) % localWorkSize[0]);
            globalWorkSize[1] = (size_t)(nGroupsY + (localWorkSize[1] - 1) - (nGroupsY + (localWorkSize[1] - 1)) % localWorkSize[1]);
            width = nGroupsX;
            height = nGroupsY;
            //New group size
            nGroupsX = (cl_int)(globalWorkSize[0] / localWorkSize[0]);
            nGroupsY = (cl_int)(globalWorkSize[1] / localWorkSize[1]);
            if (nGroupsPerChannel > 1)
            {
                nGroupsPerChannel = nGroupsX * nGroupsY;
            }
            else
            {
                nGroupsPerChannel = 0;
            }
            if(resultIsInFirstDeviceMem)
            {
                inputAvg = &normlayer->tempFloatBuffer1;
                resultSum = &normlayer->tempFloatBuffer2;
            }
            else
            {
                inputAvg = &normlayer->tempFloatBuffer2;
                resultSum = &normlayer->tempFloatBuffer1;
            }

        }

        //download the avg
        if (resultIsInFirstDeviceMem)
        {
            dR_downloadArray(net, "tempFloatBuffer1", &normlayer->tempFloatBuffer1, 0 /*offset*/, depth * sizeof(cl_float), normlayer->resultHost);
        }
        else
        {
            dR_downloadArray(net, "tempFloatBuffer2", &normlayer->tempFloatBuffer2, 0 /*offset*/, depth * sizeof(cl_float), normlayer->resultHost);
        }
        sum = 0.0;
        for(i = 0; i<depth; i++)
        {
            sum += normlayer->resultHost[i];
        }
        gavg = sum / (cl_float)(normlayer->ishape.s0 * normlayer->ishape.s1 * normlayer->ishape.s2);
	    if(net->config->debugInfo)
        {        
            g_print("Average: %f\n",gavg);
        }    
    }

    if(normlayer->type == tNormDev || normlayer->type == tNormMeanDev)
    {
        // Reduce Stddev
        size_t globalWorkSize[3];
        size_t localWorkSize[3];
        cl_int nGroupsPerChannel;
        cl_int nGroupsX;
        cl_int nGroupsY;
        cl_int width = normlayer->ishape.s0;
        cl_int height = normlayer->ishape.s1;
        cl_int depth = normlayer->ishape.s2;

        cl_int gidOfLastValidElement =  (normlayer->ishape.s1-1) * normlayer->ishape.s0 + normlayer->ishape.s0 - 1;
        cl_mem* inputAvg = input;
        cl_mem* resultSum = &normlayer->tempFloatBuffer1;
        gboolean resultIsInFirstDeviceMem = FALSE; //will be changed to TRUE on first run
        cl_int firststep = 1;
        size_t lMemSize;
        cl_float avg = gavg;

        gfloat sum;
        int i;

        localWorkSize[0] = (size_t)normlayer->localworksizex;
        localWorkSize[1] = (size_t)normlayer->localworksizey;
        localWorkSize[2] = 1;
        lMemSize = localWorkSize[0] * localWorkSize[1];
        globalWorkSize[0] = (size_t)normlayer->globalWorkSizeX;
        globalWorkSize[1] = (size_t)normlayer->globalWorkSizeY;
        globalWorkSize[2] = normlayer->ishape.s2;
        nGroupsX = (cl_int)(globalWorkSize[0] / localWorkSize[0]);
        nGroupsY = (cl_int)(globalWorkSize[1] / localWorkSize[1]);
        nGroupsPerChannel = nGroupsX * nGroupsY;

        while (nGroupsPerChannel >= 1)
        {
            resultIsInFirstDeviceMem = ! resultIsInFirstDeviceMem;

            net->clConfig->clError = clSetKernelArg(normlayer->clKernelGetDev, 0, sizeof(cl_mem), (void *)inputAvg);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetDev, 1, sizeof(cl_mem), (void *)resultSum);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetDev, 2, lMemSize * sizeof(cl_float), NULL);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetDev, 3, sizeof(cl_int), (void *)&width);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetDev, 4, sizeof(cl_int), (void *)&height);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetDev, 5, sizeof(cl_int), (void *)&gidOfLastValidElement);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetDev, 6, sizeof(cl_float), (void *)&avg);
            net->clConfig->clError |= clSetKernelArg(normlayer->clKernelGetDev, 7, sizeof(cl_int), (void *)&firststep);

            if (dR_openCLError(net, "Setting kernel args failed.", "clKernelGetDev"))
                return FALSE;

            // execute kernel
            net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, normlayer->clKernelGetDev, 3 /* number of dimensions used*/, NULL, globalWorkSize,
                localWorkSize, 0, NULL, net->clConfig->clEvent);
            if (dR_openCLError(net, "Executing kernel failed.", "clKernelGetDev"))
                return FALSE;

            dR_finishCLKernel(net, "libnnapp::clKernelGetDev");

            //Update variables:

            //Last valid element equals to the previous number of groups, as each group produces one output.
            gidOfLastValidElement = nGroupsPerChannel -1;
            //Global work size must be a multiple of local work size.
            //Uses previous number of groups (x and y)
            //E.g. if previous group size is 20x10 (200 groups) and local work size is 16x8 new global size will be 32x16
            globalWorkSize[0] = (size_t)(nGroupsX + (localWorkSize[0] - 1) - (nGroupsX + (localWorkSize[0] - 1)) % localWorkSize[0]);
            globalWorkSize[1] = (size_t)(nGroupsY + (localWorkSize[1] - 1) - (nGroupsY + (localWorkSize[1] - 1)) % localWorkSize[1]);
            //for width and height the global size is used, even if there is padding. Because the padding is removed by checking gidOfLastValidElement
            width = nGroupsX;
            height = nGroupsY;
            //New group size
            nGroupsX = (cl_int)(globalWorkSize[0] / localWorkSize[0]);
            nGroupsY = (cl_int)(globalWorkSize[1] / localWorkSize[1]);
            if (nGroupsPerChannel > 1)
            {
                nGroupsPerChannel = nGroupsX * nGroupsY;
            }
            else
            {
                nGroupsPerChannel = 0;
            }
            if(resultIsInFirstDeviceMem)
            {
                inputAvg = &normlayer->tempFloatBuffer1;
                resultSum = &normlayer->tempFloatBuffer2;
            }
            else
            {
                inputAvg = &normlayer->tempFloatBuffer2;
                resultSum = &normlayer->tempFloatBuffer1;
            }
            firststep = 0;
        }

        //download the sum dev
        if (resultIsInFirstDeviceMem)
        {
            dR_downloadArray(net, "tempFloatBuffer1", &normlayer->tempFloatBuffer1, 0 /*offset*/, depth * sizeof(cl_float), normlayer->resultHost);
        }
        else
        {
            dR_downloadArray(net, "tempFloatBuffer2", &normlayer->tempFloatBuffer2, 0 /*offset*/, depth * sizeof(cl_float), normlayer->resultHost);
        }
        sum = 0.0;
        for(i = 0; i<depth; i++)
        {
            sum += normlayer->resultHost[i];
        }
        gdev = sqrt(sum / (cl_float)(normlayer->ishape.s0 * normlayer->ishape.s1 * normlayer->ishape.s2));
	    if(net->config->debugInfo)
        {
            g_print("Stddev: %f\n",gdev);
        }
    }
    // Normalize

    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;

    targetMean = normlayer->targetMean;
    targetDev = normlayer->targetDev;


    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);          paramid++;
    if(normlayer->type == tNormMean || normlayer->type == tNormMeanDev)
    {
        cl_float valuetoadd = -(gavg-targetMean);
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_float), (void *)&valuetoadd);          paramid++;
    }
    if(normlayer->type == tNormDev || normlayer->type == tNormMeanDev)
    {
        cl_float valuetomul = 1.0f/(gdev/targetDev);
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_float), (void *)&valuetomul);          paramid++;
    }
    if (dR_openCLError(net, "Setting kernel args failed.", "Normalization Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:normalization");

}

gboolean dR_norm_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_Normalization_Data* normlayer = ((dR_Normalization_Data*)(layer->layer));
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("Normalization Layer with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    normlayer->ishape.s0 = lastlayer->oshape.s0;
    normlayer->ishape.s1 = lastlayer->oshape.s1;
    normlayer->ishape.s2 = lastlayer->oshape.s2;

    layer->oshape.s0 = normlayer->ishape.s0;
    layer->oshape.s1 = normlayer->ishape.s1;
    layer->oshape.s2 = normlayer->ishape.s2;
    return TRUE;
}

gint32 dR_norm_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_norm_createKernel(dR_Graph* net, dR_Node* layer)
{
    dR_Normalization_Data* norm = (dR_Normalization_Data*)(layer->layer);
    gchar* kernelname;
    switch(norm->type)
    {
    case tNormDev:
        kernelname = "mulScalar";
        dR_createKernel(net,"getSumDev",&(norm->clKernelGetDev));
        dR_createKernel(net,"getAvg",&(norm->clKernelGetAvg));
        break;
    case tNormMean:
        kernelname = "addScalar";
        dR_createKernel(net,"getAvg",&(norm->clKernelGetAvg));
        break;
    case tNormMeanDev:
        kernelname = "addMulScalar";
        dR_createKernel(net,"getSumDev",&(norm->clKernelGetDev));
        dR_createKernel(net,"getAvg",&(norm->clKernelGetAvg));
        break;
    }
    return dR_createKernel(net,kernelname,&(layer->clKernel));
}


gboolean dR_norm_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_Normalization_Data* normlayer = ((dR_Normalization_Data*)(layer->layer));
    ret &= dR_createFloatBuffer(net, &(normlayer->tempFloatBuffer1),(normlayer->ishape.s0/normlayer->localworksizex)*(normlayer->ishape.s1/normlayer->localworksizey), CL_MEM_READ_WRITE);
    ret &= dR_createFloatBuffer(net, &(normlayer->tempFloatBuffer2),(normlayer->ishape.s0/normlayer->localworksizex)*(normlayer->ishape.s1/normlayer->localworksizey), CL_MEM_READ_WRITE);
    normlayer->resultHost = g_malloc(normlayer->ishape.s2*sizeof(gfloat));
    return ret;
}

gboolean dR_norm_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_norm_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_Normalization_Data* normlayer = ((dR_Normalization_Data*)(layer->layer));
    ret &= dR_clMemoryBufferCleanup(net, ((dR_Normalization_Data*)(layer->layer))->tempFloatBuffer1);
    ret &= dR_clMemoryBufferCleanup(net, ((dR_Normalization_Data*)(layer->layer))->tempFloatBuffer2);
    ret &= dR_cleanupKernel((normlayer->clKernelGetAvg));
    ret &= dR_cleanupKernel((normlayer->clKernelGetDev));
    ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_norm_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    dR_Normalization_Data* normlayer = ((dR_Normalization_Data*)(layer->layer));
    if(net->prepared)
    {
        g_free(normlayer->resultHost);
        g_free((dR_Normalization_Data*)(layer->layer));
    }
    return TRUE;
}

gchar* dR_norm_printLayer(dR_Node* layer)
{
    dR_Normalization_Data* normlayer = (dR_Normalization_Data*)(layer->layer);
    gchar* out;
    gchar* typestr;
    if(normlayer->type == tNormMean)
        typestr = "Mean Adjustment";
    else if(normlayer->type == tNormDev)
        typestr = "Dev Adjustment";
    else if(normlayer->type == tNormMeanDev)
        typestr = "Mean and Dev Adjustment";
    out = g_strdup_printf("%s%d%s%s%s%f%s%f%s",
            "Normalization Layer: ",layer->layerID,
            "\n Type: ", typestr,
            "\n Target Mean: ", normlayer->targetMean,
            "\n Target Dev: ", normlayer->targetDev,"\n");
    return out;
}
