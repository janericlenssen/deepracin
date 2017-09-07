#include "dR_nodes_pooling.h"
#include "dR_core.h"
// Mandatory

dR_Node* dR_Pooling(dR_Graph* net, dR_Node* inputLayer, dR_Shape4* sh, dR_Shape4* st, dR_PoolingType type){
    dR_Pooling_Data* pool = g_malloc(sizeof(dR_Pooling_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    pool->shape.s0 = sh->s0;
    pool->shape.s1 = sh->s1;
    pool->shape.s2 = sh->s2;
    pool->shape.s3 = sh->s3;
    pool->stride.s0 = st->s0;
    pool->stride.s1 = st->s1;
    pool->stride.s2 = st->s2;
    pool->stride.s3 = st->s3;
    pool->poolingType = type;
    l->layer = pool;
    l->type = tPooling;

    // Mandatory
    l->compute = dR_pooling_compute;
    l->schedule = dR_pooling_schedule;
    l->propagateShape = dR_pooling_propagateShape;
    l->getRequiredOutputBufferSize = dR_pooling_getRequiredOutputBufferSize;
    l->createKernel = dR_pooling_createKernel;
    l->allocateBuffers = dR_pooling_allocateBuffers;
    l->fillBuffers = dR_pooling_fillBuffers;
    l->cleanupBuffers = dR_pooling_cleanupBuffers;
    l->cleanupLayer = dR_pooling_cleanupLayer;
    l->serializeNode = dR_pooling_serializeNode;
    l->parseAppendNode = dR_pooling_parseAppendNode;

    // Optional
    l->generateKernel = dR_pooling_generateKernel;
    l->setVariables = NULL;
    l->createKernelName = dR_pooling_createKernelName;
    l->printLayer = dR_pooling_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: Pooling Layer needs an appropriate Inputnode \n");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_pooling_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_Pooling_Data* pooling = (dR_Pooling_Data*)(layer->layer);
    gchar* desc = "Pooling";
    gint numNodeParams = 9;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",pooling->poolingType);
    params[1] = g_strdup_printf("%d",pooling->shape.s0);
    params[2] = g_strdup_printf("%d",pooling->shape.s1);
    params[3] = g_strdup_printf("%d",pooling->shape.s2);
    params[4] = g_strdup_printf("%d",pooling->shape.s3);
    params[5] = g_strdup_printf("%d",pooling->stride.s0);
    params[6] = g_strdup_printf("%d",pooling->stride.s1);
    params[7] = g_strdup_printf("%d",pooling->stride.s2);
    params[8] = g_strdup_printf("%d",pooling->stride.s3);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_pooling_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 9;
    gint numNodeVariables = 0;
    dR_Node* out;
    dR_Shape4* shape;
    dR_Shape4* stride;
    stride = g_malloc(sizeof(dR_Shape4));
    shape = g_malloc(sizeof(dR_Shape4));
    if(numINodes!=1)
    {
        g_print("Parsing Error: Pooling Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: Pooling Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    shape->s0 = atoi(params[1]);
    shape->s1 = atoi(params[2]);
    shape->s2 = atoi(params[3]);
    shape->s3 = atoi(params[4]);
    stride->s0 = atoi(params[5]);
    stride->s1 = atoi(params[6]);
    stride->s2 = atoi(params[7]);
    stride->s3 = atoi(params[8]);
    out = dR_Pooling(net, iNodes[0], shape, stride, atoi(params[0]));
    g_free(shape);
    g_free(stride);
    return out;
}

gboolean dR_pooling_schedule(dR_Graph* net, dR_Node* layer){

    // Minimizing global memory accesses subject to available local memory
    /*
    size_t maxwgs;
    clGetKernelWorkGroupInfo(layer->clKernel, net->clConfig->clDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxwgs,NULL);
    FullyConnected* fclayer = (FullyConnected*)(layer->layer);
    gint lws = 16;
    cl_ulong lms;
    gint localMemoryConFilter = (fclayer->shape.s0*fclayer->shape.s1*fclayer->shape.s3*4);
    gint localMemoryConInput = ((lws+(fclayer->shape.s0-1))*(lws+(fclayer->shape.s0-1))*4);
    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lms, NULL);
    */

    dR_Pooling_Data* poollayer = (dR_Pooling_Data*)(layer->layer);
    poollayer->useLMEM = FALSE;

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Scheduled Pooling Layer \n");

    return TRUE;
 }

gboolean dR_pooling_compute(dR_Graph* net, dR_Node* layer){
    dR_Pooling_Data* poollayer = ((dR_Pooling_Data*)(layer->layer));
    cl_int iwidth = poollayer->ishape.s0;
    cl_int iheight = poollayer->ishape.s1;
    cl_int idepth = poollayer->ishape.s2;

    cl_int sizex = poollayer->shape.s1;
    cl_int sizey = poollayer->shape.s2;
    cl_int sizez = poollayer->shape.s3;

    cl_int strideWidth = poollayer->stride.s1;
    cl_int strideHeight = poollayer->stride.s2;
    cl_int strideDepth = poollayer->stride.s3;


    size_t globalWorkSize[3];

    int paramid = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;

    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;

    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                  paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr); paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&iwidth);                paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&iheight);               paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&idepth);                paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&sizex);                 paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&sizey);                 paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&sizez);                 paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&strideWidth);           paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&strideHeight);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&strideDepth);           paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Pooling Kernel"))
        return FALSE;
    //execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:pool");
}


gboolean dR_pooling_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_Pooling_Data* poollayer = ((dR_Pooling_Data*)(layer->layer));
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("Pooling Layer with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    poollayer->ishape.s0 = lastlayer->oshape.s0;
    poollayer->ishape.s1 = lastlayer->oshape.s1;;
    poollayer->ishape.s2 = lastlayer->oshape.s2;;

    if(poollayer->ishape.s0%poollayer->stride.s1==0)
        layer->oshape.s0 = poollayer->ishape.s0/poollayer->stride.s1;
    else
        layer->oshape.s0 = (poollayer->ishape.s0+(poollayer->stride.s1-(poollayer->ishape.s0%poollayer->stride.s1)))/poollayer->stride.s1;

    if(poollayer->ishape.s1%poollayer->stride.s2==0)
        layer->oshape.s1 = poollayer->ishape.s1/poollayer->stride.s2;
    else
        layer->oshape.s1 = (poollayer->ishape.s1+(poollayer->stride.s2-(poollayer->ishape.s1%poollayer->stride.s2)))/poollayer->stride.s2;

    if(poollayer->ishape.s2%poollayer->stride.s3==0)
        layer->oshape.s2 = poollayer->ishape.s2/poollayer->stride.s3;
    else
        layer->oshape.s2 = (poollayer->ishape.s2+(poollayer->stride.s3-(poollayer->ishape.s2%poollayer->stride.s3)))/poollayer->stride.s3;


    if(layer->oshape.s0<=0 || layer->oshape.s1<=0 || layer->oshape.s2<=0)
    {
        if(!net->config->silent)
            g_print("Pooling Shape propagates to Zero. False shape parameters! \n");
        return FALSE;
    }
    return TRUE;
}

gint32 dR_pooling_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_pooling_createKernel(dR_Graph* net, dR_Node* layer)
{
    //dR_Pooling_Data* pool = (dR_Pooling_Data*)(layer->layer);
    return dR_createKernel(net,dR_pooling_createKernelName(layer),&(layer->clKernel));
}

gboolean dR_pooling_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_pooling_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_pooling_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = FALSE;
    if(net->prepared)
    {
        ret = dR_cleanupKernel(layer->clKernel);
    }
    return ret;
}

gboolean dR_pooling_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
    {
        g_free((dR_Pooling_Data*)(layer->layer));
    }
    return TRUE;
}

// Optional


gboolean dR_pooling_generateKernel(dR_Graph* net, dR_Node* layer)
{
    dR_Pooling_Data* poollayer = ((dR_Pooling_Data*)(layer->layer));
    gchar* source, *temp;
    int length = 0;
    int z,y,x;
    gchar* kernelname = dR_pooling_createKernelName(layer);
    gchar * filePath;

    source = g_strdup_printf(
    "#ifndef kerneldef_%s \n#define kerneldef_%s \n\n__kernel void %s(\n",kernelname,kernelname,kernelname);


    temp = g_strdup_printf(
    "    const __global  float * gImage,\n"
    "    __global  float * gOutputImage,\n"
    "    const int width,\n"
    "    const int height,\n"
    "    const int depth,\n"
    "    const int sizex,\n"
    "    const int sizey,\n"
    "    const int sizez,\n"
    "    const int strifex,\n"
    "    const int strifey,\n"
    "    const int strifez\n"
    ")\n"
    "{\n"
    "    const int indexx = get_global_id(0)*strifex;\n"
    "    const int indexy = get_global_id(1)*strifey;\n"
    "    const int indexz = get_global_id(2)*strifez;\n"
    "    const int outputwidth = (int)ceil((float)width/strifex);\n"
    "    const int outputheight= (int)ceil((float)height/strifey);\n"
    "    const int ygap = width-sizex;\n"
    "    const int zgap = width*(height-sizey)-sizex;\n"
    "\n"
    "    //Implementation without using local memory. If overlapping is happening (stride<size) maybe inefficient.\n");
    source = concat_and_free_old(source,temp);
    switch(poollayer->poolingType){
    case tMax:
        temp = g_strdup_printf(
        "    float res = -FLT_MAX;\n");
        source = concat_and_free_old(source,temp);
        break;
    case tAverage:
        temp = g_strdup_printf(
        "    float res = 0.0f;\n");
        source = concat_and_free_old(source,temp);
        break;
    case tl2norm:
        temp = g_strdup_printf(
        "    float res = 0.0f;\n");  // Not implemented
        source = concat_and_free_old(source,temp);
        break;
    }
    temp = g_strdup_printf(
    "    int ix = indexx;\n"
    "    int iy = indexy;\n"
    "    int iz = indexz;\n");
    source = concat_and_free_old(source,temp);
    for(z = 0; z < poollayer->shape.s3; z++)
    {
        for (y = 0; y < poollayer->shape.s2; y++)
        {
            for (x = 0; x < poollayer->shape.s1; x++)
            {

                switch(poollayer->poolingType){
                case tMax:
                    temp = g_strdup_printf(
                    "    res = max(res,sampleZeroPaddedFloat(gImage, iz*width*height,ix,iy,width,height));\n");
                    source = concat_and_free_old(source,temp);
                    break;
                case tAverage:
                    temp = g_strdup_printf(
                    "    res += sampleZeroPaddedFloat(gImage, iz*width*height,ix,iy,width,height);\n");
                    source = concat_and_free_old(source,temp);
                    break;
                case tl2norm:
                    temp = g_strdup_printf(
                    "    res = max(maxf,sampleZeroPaddedFloat(gImage, iz*width*height,ix,iy,width,height));\n");  // Not implemented
                    source = concat_and_free_old(source,temp);
                    break;
                }

                temp = g_strdup_printf(
                "    ix++;\n");
                source = concat_and_free_old(source,temp);
            }
            temp = g_strdup_printf(
            "    iy++; ix=indexx;\n");
            source = concat_and_free_old(source,temp);
        }
        temp = g_strdup_printf(
        "    iz++; iy=indexy;\n");
        source = concat_and_free_old(source,temp);
    }
    switch(poollayer->poolingType){
    case tMax:
        temp = g_strdup_printf(
        "    gOutputImage[get_global_id(2)*outputwidth*outputheight + get_global_id(1)*outputwidth + get_global_id(0)] = res;\n");
        source = concat_and_free_old(source,temp);
        break;
    case tAverage:
        temp = g_strdup_printf(
        "    gOutputImage[get_global_id(2)*outputwidth*outputheight + get_global_id(1)*outputwidth + get_global_id(0)] = res/(sizex*sizey*sizez);\n");
        source = concat_and_free_old(source,temp);
        break;
    case tl2norm:
        temp = g_strdup_printf(
        "    gOutputImage[get_global_id(2)*outputwidth*outputheight + get_global_id(1)*outputwidth + get_global_id(0)] = res;\n");  // Not implemented
        source = concat_and_free_old(source,temp);
        break;
    }
    temp = g_strdup_printf(
    "}\n"
    "#endif \n");
    source = concat_and_free_old(source,temp);


    length = strlen(source);
    filePath = g_build_filename(net->config->modelPath, dR_pooling_createKernelName(layer), NULL);
    g_file_set_contents(filePath,source,length,NULL);
    g_free(source);
    return TRUE;
}

gchar* dR_pooling_createKernelName(dR_Node* layer)
{
    dR_Pooling_Data* poollayer = (dR_Pooling_Data*)(layer->layer);
    gint length = 0;
	gint maxsize = 100;
    gchar* string = g_malloc(sizeof(char)*maxsize);

    length += g_snprintf(string+length,maxsize-length,"Pool%dx%d_%d_", poollayer->shape.s1, poollayer->shape.s2,layer->layerID);
    if(!poollayer->useLMEM)
        length += g_snprintf(string+length,maxsize-length,"nocopy");
    else
        length += g_snprintf(string+length,maxsize-length,"copy");
    switch(poollayer->poolingType){
    case tMax:
        length += g_snprintf(string+length,maxsize-length,"Max");
        break;
    case tAverage:
        length += g_snprintf(string+length,maxsize-length,"Average");
        break;
    case tl2norm:
        length += g_snprintf(string+length,maxsize-length,"L2norm");
        break;
    }

    return string;
}

gchar* dR_pooling_printLayer(dR_Node* layer)
{
    gchar* out;
    gchar* typestr;
    dR_Pooling_Data* poollayer = (dR_Pooling_Data*)(layer->layer);
    if(poollayer->poolingType == tMax)
        typestr = "Max Pooling";
    else if(poollayer->poolingType == tAverage)
        typestr = "Average pooling";
    else if(poollayer->poolingType == tl2norm)
        typestr = "L2 norm pooling";
    out = g_strdup_printf("%s%d%s%d%s%d%s%d%s%d%s%s%s%d%s%d%s%d%s%d%s","Pooling Layer: ",layer->layerID, "\n Shape: ", poollayer->shape.s0,", ",poollayer->shape.s1,", ", poollayer->shape.s2,", ",poollayer->shape.s3,
            "\n Pooling Type: ",typestr,
            "\n Stride: ", poollayer->stride.s0, ", ", poollayer->stride.s1, ", ", poollayer->stride.s2,", ",poollayer->stride.s3,"\n");
    return out;
}
