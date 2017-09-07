#include "dR_nodes_conv2d.h"
#include "dR_core.h"


// ///////////////////////////////////
// Standard 2D Convolutional Layer  //
// ///////////////////////////////////


// Mandatory

dR_Node* dR_Conv2d(dR_Graph* net, dR_Node* inputLayer, dR_Shape4* sh, dR_Shape4* st, dR_ActivationType acttype, gboolean useB){
    dR_Conv2d_Data* conv2d = g_malloc(sizeof(dR_Conv2d_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    conv2d->shape.s0 = sh->s0;
    conv2d->shape.s1 = sh->s1;
    conv2d->shape.s2 = sh->s2;
    conv2d->shape.s3 = sh->s3;
    conv2d->stride.s0 = st->s0;
    conv2d->stride.s1 = st->s1;
    conv2d->stride.s2 = st->s2;
    conv2d->stride.s3 = st->s3;
    conv2d->activation = acttype;
    conv2d->useBias = useB;
    conv2d->hasVariables = FALSE;
    l->layer = conv2d;
    l->type = tConv2d;

    l->propagateShape = dR_conv2d_propagateShape;
    l->getRequiredOutputBufferSize = dR_conv2d_getRequiredOutputBufferSize;
    l->printLayer = dR_conv2d_printLayer;
    l->setVariables = dR_Conv2d_setVariables;
    l->serializeNode = dR_conv2d_serializeNode;
    l->parseAppendNode = dR_conv2d_parseAppendNode;

    //If filter is 3x3 and stride 1, use Winograd implementation
    if(sh->s0==3 && sh->s1==3 && st->s1==1 && st->s2==1)
    {
        l->compute = dR_conv2d_winograd_compute;
        l->schedule = dR_conv2d_winograd_schedule;
        l->createKernel = dR_conv2d_winograd_createKernel;
        l->allocateBuffers = dR_conv2d_winograd_allocateBuffers;
        l->fillBuffers = dR_conv2d_winograd_fillBuffers;
        l->cleanupBuffers = dR_conv2d_winograd_cleanupBuffers;
        l->cleanupLayer = dR_conv2d_winograd_cleanupLayer;


        conv2d->winoParallelizationPattern = 0;
        //l->generateKernel = dR_conv2d_winograd_generateKernel;
        //l->createKernelName = dR_conv2d_winograd_createKernelName;
        l->generateKernel = NULL;
        l->createKernelName = NULL;
    }
    // If filter is 1x1, use special conv2d_1x1 algorithm ("local fully connected")
    else if(sh->s0==1 && sh->s1==1 && st->s1==1 && st->s2==1)
    //else if(FALSE)
    {
        l->compute = dR_conv2d_1x1_compute;
        l->schedule = dR_conv2d_1x1_schedule;
        l->createKernel = dR_conv2d_1x1_createKernel;
        l->allocateBuffers = dR_conv2d_1x1_allocateBuffers;
        l->fillBuffers = dR_conv2d_1x1_fillBuffers;
        l->cleanupBuffers = dR_conv2d_1x1_cleanupBuffers;
        l->cleanupLayer = dR_conv2d_1x1_cleanupLayer;

        l->generateKernel = NULL;
        l->createKernelName = NULL;
    }
    else
    {
        l->compute = dR_conv2d_direct_compute;
        l->schedule = dR_conv2d_direct_schedule;
        l->createKernel = dR_conv2d_direct_createKernel;
        l->allocateBuffers = dR_conv2d_direct_allocateBuffers;
        l->fillBuffers = dR_conv2d_direct_fillBuffers;
        l->cleanupBuffers = dR_conv2d_direct_cleanupBuffers;
        l->cleanupLayer = dR_conv2d_direct_cleanupLayer;

        l->generateKernel = dR_conv2d_direct_generateKernel;
        l->createKernelName = dR_conv2d_direct_createKernelName;
    }

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: Conv2d needs an appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}


gboolean dR_conv2d_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
         if(!net->config->silent)
         {
            g_print("Conv2d Layer with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
         }
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    convlayer->ishape = lastlayer->oshape;

    layer->oshape.s0 = convlayer->ishape.s0/convlayer->stride.s1;
    layer->oshape.s1 = convlayer->ishape.s1/convlayer->stride.s2;
    layer->oshape.s2 = convlayer->shape.s3;

    if(layer->oshape.s0<=0 || layer->oshape.s1<=0 || layer->oshape.s2<=0)
    {
        g_print("Conv2d shape propagates to Zero. False shape parameters! \n");
        return FALSE;
    }
    return TRUE;
}

gint32 dR_conv2d_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;

}

gchar* dR_conv2d_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_Conv2d_Data* convlayer = (dR_Conv2d_Data*)(layer->layer);
    gchar* desc = "Conv2D";
    gint numNodeParams = 10;
    gint numNodeVariables;
    if(convlayer->useBias)
    {
        numNodeVariables = 2;
    }
    else
    {
        numNodeVariables = 1;
    }
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",convlayer->activation);
    params[1] = g_strdup_printf("%d",convlayer->useBias?1:0);
    params[2] = g_strdup_printf("%d",convlayer->shape.s0);
    params[3] = g_strdup_printf("%d",convlayer->shape.s1);
    params[4] = g_strdup_printf("%d",convlayer->shape.s2);
    params[5] = g_strdup_printf("%d",convlayer->shape.s3);
    params[6] = g_strdup_printf("%d",convlayer->stride.s0);
    params[7] = g_strdup_printf("%d",convlayer->stride.s1);
    params[8] = g_strdup_printf("%d",convlayer->stride.s2);
    params[9] = g_strdup_printf("%d",convlayer->stride.s3);

    *numVariables = numNodeVariables;
    variableSizes[0] = convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s2*convlayer->shape.s3;
    variables[0] = convlayer->weights;
    if(convlayer->useBias)
    {
        variableSizes[1] = convlayer->shape.s3;
        variables[1] = convlayer->biases;
    }
    return desc;
}

dR_Node* dR_conv2d_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 10;
    gint numNodeVariables = 2;
    dR_Shape4* shape;
    dR_Shape4* stride;
    dR_Node* out;
    shape = g_malloc(sizeof(dR_Shape4));
    stride = g_malloc(sizeof(dR_Shape4));
    if(numINodes!=1)
    {
        g_print("Parsing Error: Conv2d Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables<numNodeVariables-1||numVariables>numNodeVariables)
    {
        g_print("Parsing Error: Conv2d Node needs %d Parameters and %d or %d Variables!\n",numNodeParams,numNodeVariables-1,numNodeVariables);
        return NULL;
    }
    shape->s0 = atoi(params[2]);
    shape->s1 = atoi(params[3]);
    shape->s2 = atoi(params[4]);
    shape->s3 = atoi(params[5]);
    stride->s0 = atoi(params[6]);
    stride->s1 = atoi(params[7]);
    stride->s2 = atoi(params[8]);
    stride->s3 = atoi(params[9]);
    out = dR_Conv2d(net, iNodes[0], shape, stride ,atoi(params[0]), atoi(params[1]));
    dR_Conv2d_setVariables(out,variables[0],atoi(params[1])?variables[1]:NULL);
    g_free(shape);
    g_free(stride);
    return out;
}



gchar* dR_conv2d_printLayer(dR_Node* layer)
{
    dR_Conv2d_Data* convlayer = (dR_Conv2d_Data*)(layer->layer);
    gchar* actstr;
    gchar* biasstr;
    gchar* out;
    if(convlayer->activation == tReLU)
        actstr = "ReLU";
    else if(convlayer->activation == tLinear)
        actstr = "Linear";
    else if(convlayer->activation == tSigmoid)
        actstr = "Sigmoid";
    else if(convlayer->activation == tTan)
        actstr = "Tan";
    if(convlayer->useBias)
        biasstr = "Yes";
    else
        biasstr = "No";
    out = g_strdup_printf("%s%d%s%d%s%d%s%d%s%d%s%s%s%d%s%d%s%d%s%d%s%s%s","Conv2d Layer: ", layer->layerID,"\n Filter dimensions: ", convlayer->shape.s0,"x",convlayer->shape.s1,
            "\n Input channels: ",convlayer->shape.s2,"\n Output channels: ", convlayer->shape.s3,
            "\n Activation: ",actstr,"\n Stride: ", convlayer->stride.s0, ", ", convlayer->stride.s1, ", ", convlayer->stride.s2,", ",convlayer->stride.s3,
            "\n Use bias: ", biasstr,"\n");
    return out;
}


void dR_Conv2d_setVariables(dR_Node* layer, gfloat* weights, gfloat* biases)
{
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    int i;
    convlayer->weights = g_malloc(convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s2*convlayer->shape.s3*sizeof(gfloat));
    for(i=0; i<convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s2*convlayer->shape.s3;i++)
    {
        convlayer->weights[i] = weights[i];
    }
    if(convlayer->useBias)
    {
        convlayer->biases = g_malloc(convlayer->shape.s3*sizeof(gfloat));
        for(i=0; i<convlayer->shape.s3;i++)
        {
            convlayer->biases[i] = biases[i];
        }
    }
    convlayer->hasVariables = TRUE;
}



// Direct Conv2D Implementation //


gboolean dR_conv2d_direct_schedule(dR_Graph* net, dR_Node* layer){

    // Trying to calculate [1,1,inputDepth,z] in one Thread and
    // [x,y,inputDepth,z] in one Workgroup

    // min f(x,y,z)= (width/x)*(height/y)*(outputDepth/z)*((x+filterx-1)*(y+filtery-1)*inputDepth + filterx*filtery*inputDepth*z)   // number of global memory accesses
    // subject to:
    // 8<=x<=width, 8<=y<=height, 1<=z<=outputDepth,
    // ((x+filterx-1)*(y+filtery-1) + filterx*filtery*z)*4<localmemorysize  // restriction based on available local memory
    // x*y = 32*k for k € N
    // x*y <= maxworkgroupsize
    // (width/x)*(height/y)*(outputDepth/z)>60 Minimum 60 Workgroups

    // Minimizing global memory accesses subject to available local memory


    //size_t maxwgs;
    //clGetKernelWorkGroupInfo(layer->clKernel, net->clConfig->clDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxwgs,NULL);

    dR_Conv2d_Data* convlayer = (dR_Conv2d_Data*)(layer->layer);
    gint lws = 16;
    cl_ulong lms;
	size_t mws;
    gint numberofwgs;
    gint localMemoryConFilter = (convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s3*4);
    gint localMemoryConInput = ((lws+(convlayer->shape.s0-1))*(lws+(convlayer->shape.s0-1))*4);
    int z=1;
    if(convlayer->shape.s0<=convlayer->stride.s1&&convlayer->shape.s1<=convlayer->stride.s2)
        convlayer->useLMEM = FALSE;
    else
        convlayer->useLMEM = TRUE;



    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lms, NULL);
    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &mws, NULL);

    // Find maximum size of workgroup in spatial dimension based on memory constraints
    if(!net->config->silent&&net->config->debugInfo)
        g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);
    while(localMemoryConInput+localMemoryConFilter<(gint)lms&&lws*lws<=(gint)mws){
        lws*=2;
        localMemoryConInput = ((lws+(convlayer->shape.s0-1))*(lws+(convlayer->shape.s0-1))*4);
    }
    lws/=2;


    if(layer->oshape.s0%lws==0)
    {
        convlayer->globalWorkSizeX = layer->oshape.s0;
    }
    else
    {
        convlayer->globalWorkSizeX = ((layer->oshape.s0/lws)+1)*lws;
    }


    if(layer->oshape.s1%lws==0)
    {
        convlayer->globalWorkSizeY = layer->oshape.s1;
    }
    else
    {
        convlayer->globalWorkSizeY = ((layer->oshape.s1/lws)+1)*lws;
    }

    convlayer->gidXGap = layer->oshape.s0 - convlayer->globalWorkSizeX;
    convlayer->gidYGap = layer->oshape.s1 - convlayer->globalWorkSizeY;

    numberofwgs = ((convlayer->globalWorkSizeX/lws) * (convlayer->globalWorkSizeY/lws))*z;


    // If Number of Workgroups based on spatial division is to small -> parallize in filter dimension
    while(numberofwgs<32&&z<convlayer->shape.s3)
    {
        z++;
        while(convlayer->shape.s3%z!=0)
            z++;
        numberofwgs = ((convlayer->globalWorkSizeX/lws) * (convlayer->globalWorkSizeY/lws))*z;
    }
    //z = 1;
    convlayer->localWorkSizexy = lws;
    convlayer->numberOfDepthPartitions = z;

    if(!net->config->silent&&net->config->debugInfo)
        g_print("LocalWorkSizexy: %d \n", (gint)convlayer->localWorkSizexy);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Number of Depth-Partitions: %d \n", (gint)convlayer->numberOfDepthPartitions);


    if(!net->config->silent&&net->config->debugInfo)
        g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Scheduled Conv2D Layer \n");

    return TRUE;
 }


gboolean dR_conv2d_direct_compute(dR_Graph* net, dR_Node* layer){
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    cl_int filterWindowWidth = convlayer->shape.s0;
    cl_int filterWindowHeight = convlayer->shape.s1;
    cl_int oDepth = convlayer->shape.s3;
    cl_int lMemImageSize, lMemFilterSize;
    size_t localWorkSize[3];
    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;

    localWorkSize[0] = convlayer->localWorkSizexy;
    localWorkSize[1] = convlayer->localWorkSizexy;;
    localWorkSize[2] = 1;
    lMemImageSize = (localWorkSize[0] + filterWindowWidth - 1) * (localWorkSize[1] + filterWindowHeight - 1);
    lMemFilterSize = filterWindowHeight*filterWindowWidth*(oDepth/convlayer->numberOfDepthPartitions);
    globalWorkSize[0] = convlayer->globalWorkSizeX;
    globalWorkSize[1] = convlayer->globalWorkSizeY;
    globalWorkSize[2] = convlayer->numberOfDepthPartitions;

    if ((filterWindowWidth-1)/2 > (cl_int)localWorkSize[0]
    || (filterWindowHeight-1) > (cl_int)localWorkSize[1])
    {
        g_print("Error: Kernel size for convolution is too big (is %ix%i; max %ix%i)\n", filterWindowWidth, filterWindowHeight, (gint)(localWorkSize[0]), (gint)(localWorkSize[1]));
        return FALSE;
    }
    net->clConfig->clError = FALSE;
    if(convlayer->useBias)
    {
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)&convlayer->biasBuf);        paramid++;
    }
    if(convlayer->useLMEM)
    {
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemImageSize * sizeof(cl_float), NULL);         paramid++;
    }
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)&convlayer->weightsBuf);     paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemFilterSize * sizeof(cl_float), NULL);            paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&iwidth);                    paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&iheight);                   paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&idepth);                    paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&filterWindowWidth);         paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&filterWindowHeight);        paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&oDepth);                    paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Conv2d Kernel"))
        return FALSE;
    //execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        localWorkSize, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:conv2d");

}



gboolean dR_conv2d_direct_createKernel(dR_Graph* net, dR_Node* layer)
{
    //dR_Conv2d_Data* convlayer = (dR_Conv2d_Data*)(layer->layer);
    return dR_createKernel(net,dR_conv2d_direct_createKernelName(layer),&(layer->clKernel));
}


gboolean dR_conv2d_direct_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_Shape4 shape;
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    shape = convlayer->shape;
    ret &= dR_createFloatBuffer(net, &(((dR_Conv2d_Data*)(layer->layer))->weightsBuf),shape.s0*shape.s1*shape.s2*shape.s3, CL_MEM_READ_ONLY);
    if(convlayer->useBias)
        ret &= dR_createFloatBuffer(net, &(((dR_Conv2d_Data*)(layer->layer))->biasBuf),shape.s3, CL_MEM_READ_ONLY);
    return ret;
}

gboolean dR_conv2d_direct_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    ret &= dR_uploadArray(net,"",convlayer->weights,0,convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s2*convlayer->shape.s3*sizeof(cl_float),convlayer->weightsBuf);
    if(convlayer->useBias)
        ret &= dR_uploadArray(net,"",convlayer->biases,0,convlayer->shape.s3*sizeof(cl_float),convlayer->biasBuf);
    return ret;
}

gboolean dR_conv2d_direct_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    ret &= dR_clMemoryBufferCleanup(net, ((dR_Conv2d_Data*)(layer->layer))->weightsBuf);
    ret &= dR_clMemoryBufferCleanup(net, ((dR_Conv2d_Data*)(layer->layer))->biasBuf);
    ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_conv2d_direct_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    if(convlayer->hasVariables)
    {
        g_free(convlayer->weights);
        if(convlayer->useBias)
        {
            g_free(convlayer->biases);
        }
    }
    g_free((dR_Conv2d_Data*)(layer->layer));
    return TRUE;
}


// Optional


gboolean dR_conv2d_direct_generateKernel(dR_Graph* net, dR_Node* layer)
{
    dR_Conv2d_Data* convlayer = (dR_Conv2d_Data*)(layer->layer);
    gint length = 0;

    gchar* kernelname;
    gchar *temp, *source;
    gchar * filePath;
    // Variables that get hardcoded into the shader
    gint width =convlayer->ishape.s0;
    gint height =convlayer->ishape.s1;
    gint inputDepth =convlayer->ishape.s2;
    gint widthheight = width*height;
    gint filterWidth = convlayer->shape.s0;
    gint filterHeight = convlayer->shape.s1;
    gint outputDepth = convlayer->shape.s3;
    gint x = -filterWidth/2;
    gint y = -filterHeight/2;
    gint lengthindepth = outputDepth/convlayer->numberOfDepthPartitions;
    gint filterxy = filterWidth*filterHeight;
    gint floatsToReadInThisWG = (convlayer->localWorkSizexy+filterWidth-1)*(convlayer->localWorkSizexy+filterHeight-1);
    gint numberOfWorkItems = convlayer->localWorkSizexy*convlayer->localWorkSizexy;
    gint copyWidth = convlayer->localWorkSizexy+ filterWidth-1;
    gint jumpint = convlayer->localWorkSizexy-1;
    gint fid = 0;

    kernelname = dR_conv2d_direct_createKernelName(layer);
    source = NULL;

    //Creating Shader Name and Return type

    source = g_strdup_printf(
    "#ifndef kerneldef_%s \n"
    "#define kerneldef_%s \n\n"
    "__kernel void %s(\n"
                ,kernelname,kernelname,kernelname);

    // Rest of the Header and first part
    if(convlayer->useBias)
    {
        temp = g_strdup_printf(
    "    const __global float * biases,\n");
        source = concat_and_free_old(source,temp);
    }
    if(convlayer->useLMEM)
    {
        temp = g_strdup_printf(
    "    __local float* lImage,\n");
        source = concat_and_free_old(source,temp);
    }
    temp = g_strdup_printf(
    "    const __global  float * gInputImage,\n"
    "    const __global float * gFilter,\n"
    "    __global  float * gOutputImage,\n"
    "    __local float* lFilter\n"
    "    )\n"
    "{\n"
    "    const int depthstart = %d*get_global_id(2);\n"
    "    int biasindex = depthstart;\n"
    //"    if(get_global_id(0)<200) return;\n"
    "    float sum = 0.0f;\n"
                ,lengthindepth);
    source = concat_and_free_old(source,temp);


    if(convlayer->useLMEM)
    {
        temp = g_strdup_printf(
    "    int index;\n");
        source = concat_and_free_old(source,temp);
    }

    temp = g_strdup_printf(
    "    const int indexstart = get_local_id(1)*%d + get_local_id(0);\n"
    "    int inputindex = 0; \n"
    "    int outputindex = 0; \n"
    "    const int inImageOffset = %d*get_global_id(1)+get_global_id(0);\n"
    "    const int blx = get_group_id(0)*%d-%d;\n"
    "    const int bly = get_group_id(1)*%d-%d;\n"
    "    if(get_global_id(0)<%d&&get_global_id(1)<%d)\n"
    "    {\n"
    "        for(outputindex=0; outputindex<%d;outputindex++)\n"
    "        {\n"
    "            gOutputImage[biasindex*%d+inImageOffset] = 0.0;\n"
    "            biasindex++;\n"
    "        }\n"
    "    }\n"
    "    biasindex=depthstart;\n",(gint)convlayer->localWorkSizexy + filterWidth -1,width,(gint)convlayer->localWorkSizexy,
                filterWidth/2,(gint)convlayer->localWorkSizexy,filterHeight/2, width, height, lengthindepth,widthheight);
    source = concat_and_free_old(source,temp);

    // Inputloop

    temp = g_strdup_printf(
    "    for(inputindex = 0; inputindex<%d; inputindex++)\n"
    "    { \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                ,inputDepth);
    source = concat_and_free_old(source,temp);

    // Copy ImagePart to LocalMem
    if(convlayer->useLMEM)
    {
        temp = g_strdup_printf(
    "        for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<%d;i+=%d)\n"
    "        {\n"
    "            lImage[i] = sampleZeroPaddedFloat(gInputImage, (inputindex* %d), blx + (i %% %d), bly + (i / %d), %d, %d);\n"
    "        }\n"
                    ,floatsToReadInThisWG, numberOfWorkItems,widthheight, copyWidth, copyWidth, width, height);
        source = concat_and_free_old(source,temp);
    }
    fid = 0;
    // Copy Filter to LocalMem
    temp = g_strdup_printf(
    "        for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<%d;i+=%d)\n"
    "            lFilter[i] = gFilter[(inputindex* %d+ (depthstart*%d)) + i];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        if(get_global_id(0)<%d&&get_global_id(1)<%d)\n"
    "        {\n"
                , lengthindepth*filterxy, (gint)convlayer->localWorkSizexy*(gint)convlayer->localWorkSizexy,filterxy*outputDepth, filterxy, width, height);
    source = concat_and_free_old(source,temp);

    // OutputLoop
    temp = g_strdup_printf(
    "            int findex = 0;\n"
    "            for(outputindex=0; outputindex<%d;outputindex++)\n"
    "            {\n"
                , lengthindepth);
    source = concat_and_free_old(source,temp);
    if(convlayer->useLMEM)
    {
        temp = g_strdup_printf(
    "                index = indexstart;\n");
        source = concat_and_free_old(source,temp);
    }
    x = -filterWidth/2;
    y = -filterHeight/2;

    //ConvolutionLoop
    temp = g_strdup_printf(
    "                for(int yid= 0; yid<%d;yid++)\n"
    "                {\n"
    "                    for(int xid = 0; xid<%d;xid++)\n"
    "                    {\n",
                filterHeight,filterWidth);
    source = concat_and_free_old(source,temp);
    if(convlayer->useLMEM)
    {
        temp = g_strdup_printf(
    "                        sum = mad(lImage[index], lFilter[findex],sum); index++; findex++;\n");
        source = concat_and_free_old(source,temp);
        fid++;
    }
    else
    {
        if(filterWidth==1 && filterHeight==1)
        {
            temp = g_strdup_printf(
        "                        sum +=  gInputImage[(inputindex*%d) + (get_global_id(1)*%d+get_global_id(0))] * lFilter[findex]; findex++;\n"
                        ,widthheight,width);
        }
        else
        {
            temp = g_strdup_printf(
        "                        sum +=  sampleZeroPaddedFloat(gInputImage, (inputindex*%d),get_global_id(0)+%d,get_global_id(1)+%d,%d,%d) * lFilter[findex]; findex++;\n"
                        ,widthheight,x,y, width, height);
        }
        source = concat_and_free_old(source,temp);
        fid++;x++;
    }
    temp = g_strdup_printf(
    "                    }\n");
    source = concat_and_free_old(source,temp);
    if(convlayer->useLMEM)
    {
        temp = g_strdup_printf(
    "                        index+=%d;\n"
                    ,jumpint);
        source = concat_and_free_old(source,temp);
    }
    // Output

    temp = g_strdup_printf(
    "                }\n"
    "                gOutputImage[biasindex*%d+inImageOffset] += sum; \n"
    "                sum = 0.0f; biasindex++;\n"
    "            }\n"
                , widthheight);
    source = concat_and_free_old(source,temp);

    temp = g_strdup_printf(
    "            biasindex = depthstart;\n"
    "        }\n"
    "    }\n");
    source = concat_and_free_old(source,temp);
    if(convlayer->activation!=tLinear||convlayer->useBias)
    {
        temp = g_strdup_printf(
    "    if(get_global_id(0)<%d&&get_global_id(1)<%d)\n"
    "    {\n"
                , width, height);
        source = concat_and_free_old(source,temp);
        temp = g_strdup_printf(
    "        for(outputindex=0; outputindex<%d;outputindex++)\n"
    "        {\n"
                , lengthindepth);
        source = concat_and_free_old(source,temp);
        if(convlayer->useBias)
        {
            switch(convlayer->activation){
            case tReLU:
                temp = g_strdup_printf(
"            gOutputImage[biasindex*%d+inImageOffset] = clamp(gOutputImage[biasindex*%d+inImageOffset]+biases[biasindex], 0.0f, FLT_MAX);\n"
"            biasindex++;\n"
                        ,widthheight, widthheight);
                source = concat_and_free_old(source,temp);
                break;
            case tLinear:
                temp = g_strdup_printf(
"            gOutputImage[biasindex*%d+inImageOffset] = gOutputImage[biasindex*%d+inImageOffset]+biases[biasindex];\n"
"            biasindex++;\n"
                        ,widthheight, widthheight);
                source = concat_and_free_old(source,temp);
                break;
            case tSigmoid:
                temp = g_strdup_printf(
"            gOutputImage[biasindex*%d+inImageOffset] += biases[biasindex];\n"
"            biasindex++;\n"
                        ,widthheight); //Not implemented
                source = concat_and_free_old(source,temp);
                break;
            case tTan:
                temp = g_strdup_printf(
"            gOutputImage[biasindex*%d+inImageOffset] += biases[biasindex];\n"
"            biasindex++;\n"
                        ,widthheight); //Not implemented
                source = concat_and_free_old(source,temp);
                break;
            }

        }
        else
        {
            switch(convlayer->activation){
            case tReLU:
                temp = g_strdup_printf(
"            gOutputImage[biasindex*%d+inImageOffset] = clamp(gOutputImage[biasindex*%d+inImageOffset], 0.0f, FLT_MAX);\n"
"            biasindex++;\n"
                        ,widthheight, widthheight);
                source = concat_and_free_old(source,temp);
                break;
            case tLinear:
                temp = g_strdup_printf(
"            gOutputImage[biasindex*%d+inImageOffset] = gOutputImage[biasindex*%d+inImageOffset];\n"
"            biasindex++;\n"
                        ,widthheight, widthheight);
                source = concat_and_free_old(source,temp);
                break;
            case tSigmoid: // Not Implemented
                temp = g_strdup_printf(
    "        gOutputImage[biasindex*%d+inImageOffset] += biases[biasindex];\n"
    "        biasindex++;\n"
                            ,widthheight);
                source = concat_and_free_old(source,temp);
                break;
            case tTan: // Not Implemented
                temp = g_strdup_printf(
    "        gOutputImage[biasindex*%d+inImageOffset] += biases[biasindex];\n"
    "        biasindex++;\n"
                            ,widthheight);
                source = concat_and_free_old(source,temp);
            break;
            }
        }
        temp = g_strdup_printf(
    "        }\n"
    "    }\n");
        source = concat_and_free_old(source,temp);
    }

    temp = g_strdup_printf(
    "}\n"
    "#endif \n\n");
    source = concat_and_free_old(source,temp);

    length = strlen(source);
    filePath = g_build_filename(net->config->modelPath, dR_conv2d_direct_createKernelName(layer), NULL);
    g_file_set_contents(filePath,source,length,NULL);

    g_free(source);

    return TRUE;
}

gchar* dR_conv2d_direct_createKernelName(dR_Node* layer)
{
    dR_Conv2d_Data* convlayer = (dR_Conv2d_Data*)(layer->layer);
    gchar* string, *temp;
    string = g_strdup_printf("conv2d%dx%dx%d_%d_", convlayer->shape.s0, convlayer->shape.s1,convlayer->shape.s3,layer->layerID);
    if(convlayer->useLMEM)
    {
        temp = g_strdup_printf("lmem");
        string = concat_and_free_old(string,temp);
    }
    switch(convlayer->activation){
    case tReLU:
        temp = g_strdup_printf("ReLU");
        break;
    case tLinear:
        temp = g_strdup_printf("Linear");
        break;
    case tSigmoid:
        temp = g_strdup_printf("Sigmoid");
        break;
    case tTan:
        temp = g_strdup_printf("Tan");
        break;
    }
    string = concat_and_free_old(string,temp);
    return string;
}



// Winograd Implementation //



gboolean dR_conv2d_winograd_schedule(dR_Graph* net, dR_Node* layer){
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    cl_ulong lms;
	size_t mws;
    gint numberofwgs;
    gint localMemoryConFilter;
    gint localMemoryConInput;
    gint width, height;
    gint z=1;
    gint zInput=1;
    gint NperDim = 1;
    // Determine size of superblocks

    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lms, NULL);
    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &mws, NULL);
    if(convlayer->shape.s3*convlayer->shape.s2*16*4<(gint)lms-(16*16*convlayer->shape.s2)&&convlayer->ishape.s0>=256&&convlayer->ishape.s1>=256)
    {
        convlayer->winogradWide = TRUE;
    }
    else
    {
        convlayer->winogradWide = FALSE;
    }
    if(!convlayer->winogradWide)
    {
        localMemoryConFilter = 1;


        localMemoryConInput = (16)*NperDim*NperDim*convlayer->shape.s2*4;

        // Find N based on memory constraints and wasted space
        if(!net->config->silent&&net->config->debugInfo)
            g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);

        while(localMemoryConInput+localMemoryConFilter>=(gint)lms)
        {
            z*=2;
            localMemoryConFilter = 16*(convlayer->shape.s3/z)*4;
        }


        while(convlayer->shape.s3/z<convlayer->shape.s2/zInput)
        {
            zInput*=2;
        }

        numberofwgs = NperDim*NperDim*(convlayer->shape.s3/z>convlayer->shape.s2/zInput?convlayer->shape.s3/z:convlayer->shape.s2/zInput);
        //int maxdepth = (convlayer->shape.s3/z>convlayer->shape.s2?convlayer->shape.s3/z:convlayer->shape.s2/zInput);

        while(localMemoryConInput+localMemoryConFilter<(gint)lms&&numberofwgs<=(gint)mws/2&&NperDim*2<convlayer->ishape.s0&&NperDim*2<convlayer->ishape.s1){
            NperDim+=1;
            localMemoryConInput = (16)*NperDim*NperDim*convlayer->shape.s2*4;
            numberofwgs = NperDim*NperDim*(convlayer->shape.s3/z>convlayer->shape.s2/zInput?convlayer->shape.s3/z:convlayer->shape.s2/zInput);
        }
        if(NperDim>1)
        {
            NperDim--;
        }
        while((convlayer->ishape.s0%(NperDim*2)>(convlayer->ishape.s0/10)||convlayer->ishape.s1%(NperDim*2)>(convlayer->ishape.s1/10))&&NperDim>4)
        {
            NperDim--;
        }
        localMemoryConInput = (16)*NperDim*NperDim*convlayer->shape.s2*4;
        numberofwgs = NperDim*NperDim*(convlayer->shape.s3/z>convlayer->shape.s2?convlayer->shape.s3/z:convlayer->shape.s2/zInput);
		while(numberofwgs>(gint)mws/2)
		{
			zInput*=2;
			z*=2;
			localMemoryConFilter = 16*(convlayer->shape.s3/z)*4;
			numberofwgs = NperDim*NperDim*(convlayer->shape.s3/z>convlayer->shape.s2/zInput?convlayer->shape.s3/z:convlayer->shape.s2/zInput);
		}
        // Find dimensions of input that need to processed (>ishape but multiple of N)
        if(layer->oshape.s0%(NperDim*2)==0)
        {
            width = layer->oshape.s0;
        }
        else
        {
            width = ((layer->oshape.s0/(NperDim*2))+1)*(NperDim*2);
        }


        if(layer->oshape.s1%(NperDim*2)==0)
        {
            height = layer->oshape.s1;
        }
        else
        {
            height = ((layer->oshape.s1/(NperDim*2))+1)*(NperDim*2);
        }
		/*
        if(NperDim==1)
        {
            while(numberofwgs<=(gint)mws/2&&z>1&&zInput>1){
                z/=2;
                zInput/=2;
                numberofwgs = NperDim*NperDim*(convlayer->shape.s3/z>convlayer->shape.s2/zInput?convlayer->shape.s3/z:convlayer->shape.s2/zInput);
            }
        }
		*/
        height = height / (2*NperDim);
        width = width / (2*NperDim);
    }
    else
    {
        localMemoryConFilter = 16*convlayer->shape.s2*(convlayer->shape.s3/z)*4;
        localMemoryConInput = (16)*NperDim*convlayer->shape.s2*4;

        // Find N based on memory constraints and wasted space
        if(!net->config->silent&&net->config->debugInfo)
            g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);

        while(localMemoryConInput+localMemoryConFilter>=(gint)lms)
        {
            z*=2;
            localMemoryConFilter = 16*(convlayer->shape.s3/z)*4;
        }


        while(convlayer->shape.s3/z<convlayer->shape.s2/zInput)
        {
            zInput*=2;
        }

        numberofwgs = NperDim*(convlayer->shape.s3/z>convlayer->shape.s2/zInput?convlayer->shape.s3/z:convlayer->shape.s2/zInput);
        //int maxdepth = (convlayer->shape.s3/z>convlayer->shape.s2?convlayer->shape.s3/z:convlayer->shape.s2/zInput);

        while(localMemoryConInput+localMemoryConFilter<(gint)lms&&numberofwgs<=(gint)mws&&NperDim*2<convlayer->ishape.s0&&NperDim*2<convlayer->ishape.s1){
            NperDim+=1;
            localMemoryConInput = (16)*NperDim*convlayer->shape.s2*4;
            numberofwgs = NperDim*(convlayer->shape.s3/z>convlayer->shape.s2/zInput?convlayer->shape.s3/z:convlayer->shape.s2/zInput);
        }
        if(NperDim>1)
        {
            NperDim--;
        }
        while((convlayer->ishape.s0%(NperDim*2)>(convlayer->ishape.s0/10)||convlayer->ishape.s1%(NperDim*2)>(convlayer->ishape.s1/10))&&NperDim>4)
        {
            NperDim--;
        }
        localMemoryConInput = (16)*NperDim*convlayer->shape.s2*4;
        numberofwgs = NperDim*(convlayer->shape.s3/z>convlayer->shape.s2?convlayer->shape.s3/z:convlayer->shape.s2/zInput);

        // Find dimensions of input that need to processed (>ishape but multiple of N)
        if(layer->oshape.s0%(NperDim*2)==0)
        {
            width = layer->oshape.s0;
        }
        else
        {
            width = ((layer->oshape.s0/(NperDim*2))+1)*(NperDim*2);
        }


        if(layer->oshape.s1%(NperDim*2)==0)
        {
            height = layer->oshape.s1;
        }
        else
        {
            height = ((layer->oshape.s1/(NperDim*2))+1)*(NperDim*2);
        }

        height = height / (2*NperDim);
        width = width / (2*NperDim);
    }
    convlayer->winogradn = 2; // Currently only F(2x2,3x3) supported
    convlayer->winogradN = NperDim*NperDim;
    convlayer->winogradNperDim = NperDim;
    convlayer->globalWorkSizeX = width;
    convlayer->globalWorkSizeY = height;
    convlayer->numberOfDepthPartitions = z;
    convlayer->numberOfDepthPartitionsInput = zInput;
    convlayer->lmemInputSize = localMemoryConInput/4;
    convlayer->lmemFilterSize = localMemoryConFilter/4;

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Wide: %s \n", convlayer->winogradWide?"True":"False");

    if(!net->config->silent&&net->config->debugInfo)
        g_print("N: %d \n", convlayer->winogradN);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Number of WG in X: %d \n", (gint)convlayer->globalWorkSizeX);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Number of WG in Y: %d \n", (gint)convlayer->globalWorkSizeY);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Workgroupsize: %d \n", numberofwgs);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Filter partitions: %d \n", convlayer->numberOfDepthPartitions);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Input partitions: %d \n", convlayer->numberOfDepthPartitionsInput);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Used Local Memory: %d \n", localMemoryConInput+localMemoryConFilter);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Scheduled Conv2D Layer \n");



    return TRUE;
 }


gboolean dR_conv2d_winograd_compute(dR_Graph* net, dR_Node* layer){
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    cl_int oDepth = convlayer->shape.s3;
    cl_int iDepth = convlayer->shape.s2;
    cl_int iwidth = convlayer->ishape.s0;
    cl_int iheight = convlayer->ishape.s1;
    cl_int maxDepth = 0;
    cl_int activation, numWGinX;
    //cl_int numO,numI,numOPerPartition,numIPerPartition;
    cl_int lMemImageSize, lMemFilterSize;
    size_t localWorkSize[3];
    size_t globalWorkSize[3];
    gint numOPerPartition, numIPerPartition, numO, numI;
    gint paramid = 0;
    cl_int usebias;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;

    if(!convlayer->winogradWide)
    {
        if(oDepth/convlayer->numberOfDepthPartitions>iDepth)
        {
            maxDepth = oDepth/convlayer->numberOfDepthPartitions;
        }
        else
        {
            maxDepth = iDepth/convlayer->numberOfDepthPartitionsInput;
        }
        localWorkSize[0] = convlayer->winogradN;
        localWorkSize[1] = maxDepth;
        localWorkSize[2] = 1;
        if(convlayer->winogradN==1)
            lMemImageSize = (16)*convlayer->winogradN*convlayer->shape.s2;
        else
            lMemImageSize = 16*convlayer->winogradN*convlayer->shape.s2;
        lMemFilterSize = 1;
        globalWorkSize[0] = convlayer->winogradN*convlayer->globalWorkSizeX*convlayer->globalWorkSizeY;
        globalWorkSize[1] = maxDepth*convlayer->numberOfDepthPartitions;
        globalWorkSize[2] = 1;
    }
    else
    {
        if(oDepth/convlayer->numberOfDepthPartitions>iDepth)
        {
            maxDepth = oDepth/convlayer->numberOfDepthPartitions;
        }
        else
        {
            maxDepth = iDepth/convlayer->numberOfDepthPartitionsInput;
        }
        localWorkSize[0] = convlayer->winogradNperDim;
        localWorkSize[1] = maxDepth;
        localWorkSize[2] = 1;
        lMemImageSize = convlayer->lmemInputSize;
        lMemFilterSize = convlayer->lmemFilterSize;
        globalWorkSize[0] = convlayer->winogradNperDim*convlayer->globalWorkSizeX*convlayer->globalWorkSizeY;
        globalWorkSize[1] = maxDepth*convlayer->numberOfDepthPartitions;
        globalWorkSize[2] = 1;
    }


    usebias = (convlayer->useBias?1:0);
    numWGinX = convlayer->globalWorkSizeX;
    //For non-generated kernels
    numOPerPartition = convlayer->shape.s3/convlayer->numberOfDepthPartitions;
    numIPerPartition = convlayer->numberOfDepthPartitionsInput;
    numO = convlayer->shape.s3;
    numI = convlayer->shape.s2;

    net->clConfig->clError = FALSE;
    if(convlayer->activation==tReLU)
    {
        activation = 1;
    }
    else
    {
        activation = 0;
    }
    /* Kernel Header:
    const __global float * biases,
    const __global float * gI,
    const __global float * gTransformedFilters,
    __global  float * gO,
    __local float* lTransformedFilters,
    __local float* lInput,
    int activation,     // 0 linear, 1 relu
    int bias,           // 0 no bias, 1 bias
    int NperDim,        // Number of blocks per Dimensions (expected to be 1-5, for N = 1, 4, 9, 16, 25)
    int OPartitionSize, // Number of Filterapplications per WorkGroup
    int IPartitionSize, // Number of Inputtransforms per WorkItem
    int O,              // Number of Filters
    int I,              // Number of Input Channels
    int bW,             // Input Buffer Width
    int bH,             // Input Buffer Height
    int numWGsInX,      // Number of Workgroups in X Dimension
    int cW,             // Clipped Input Width
    int cH              // Clipped Input Height*/

    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)&convlayer->biasBuf);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                        paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)&convlayer->weightsBuf);       paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);     paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemFilterSize * sizeof(cl_float), NULL);              paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemImageSize  * sizeof(cl_float), NULL);              paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&activation);                  paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&usebias);                     paramid++;
    //For non-generated kernels
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&convlayer->winogradNperDim);  paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&numOPerPartition);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&numIPerPartition);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&numO);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&numI);                      paramid++;

    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&iwidth);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&iheight);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&numWGinX);                      paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Conv2dWino Kernel"))
        return FALSE;
    //execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 2, NULL, globalWorkSize,
        localWorkSize, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:conv2dwino");

}



gboolean dR_conv2d_winograd_createKernel(dR_Graph* net, dR_Node* layer)
{
    //return dR_createKernel(net,dR_conv2d_winograd_createKernelName(layer),&(layer->clKernel));
    dR_Conv2d_Data* convlayer = (dR_Conv2d_Data*)(layer->layer);
    if(!convlayer->winogradWide)
        return dR_createKernel(net,"conv2dwinograd2",&(layer->clKernel));
    else
        return dR_createKernel(net,"conv2dwinograd2_wide",&(layer->clKernel));
}


gboolean dR_conv2d_winograd_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    dR_Shape4 shape;
    gboolean ret = TRUE;
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    shape = convlayer->shape;
    // Create Buffer for Transformed Filters
    ret &= dR_createFloatBuffer(net, &convlayer->weightsBuf,(convlayer->winogradn+2)*(convlayer->winogradn+2)*shape.s3*shape.s2, CL_MEM_READ_ONLY);
    if(convlayer->useBias)
        ret &= dR_createFloatBuffer(net, &convlayer->biasBuf,shape.s3, CL_MEM_READ_ONLY);
    return ret;
}



gboolean dR_conv2d_winograd_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    gint i,o,k;
    gfloat* transformedFilters, *temp, *transformedFiltersTemp;
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    dR_Shape4 shape = convlayer->shape;
    gboolean ret = TRUE;
    gfloat filtersTemp[9];

    // Since it is always F(2x2,3x3) or F(4x4,3x3), matrix transforms are hardcoded in Kernel
    // Create Transformed Filters (with Matrix G)
    gfloat filterTransform4x3[4*3] = { 1.0, 0.0, 0.0,
                                       0.5, 0.5, 0.5,
                                       0.5,-0.5, 0.5,
                                       0.0, 0.0, 1.0};
    /*gfloat filterTransform4x3[4*3] = { 0.0, 0.0, 1.0,
                                       0.5,-0.5, 0.5,
                                       0.5, 0.5, 0.5,
                                       1.0, 0.0, 0.0};*/

    gfloat filterTransform6x3[6*3] = { 1.,  0.,  0.,
                                       1., -1.,  1.,
                                       1.,  1.,  1.,
                                       1., -2.,  4.,
                                       1.,  2.,  4.,
                                       0.,  0.,  1.};
    transformedFilters = g_malloc((convlayer->winogradn+2)*(convlayer->winogradn+2)*shape.s2*shape.s3*sizeof(gfloat));
    transformedFiltersTemp = g_malloc((convlayer->winogradn+2)*(convlayer->winogradn+2)*shape.s2*shape.s3*sizeof(gfloat));

    temp = g_malloc((convlayer->winogradn+2)*3*sizeof(gfloat));

    for(i = 0; i<shape.s2;i++)
    {
        for(o = 0; o<shape.s3;o++)
        {
            if(convlayer->winogradn==2)
            {
                //swap filters
                for(k = 0; k<3;k++)
                {
                    filtersTemp[k*3+0] = convlayer->weights[(i*shape.s3*9+o*9)+k*3+0];
                    filtersTemp[k*3+1] = convlayer->weights[(i*shape.s3*9+o*9)+k*3+1];
                    filtersTemp[k*3+2] = convlayer->weights[(i*shape.s3*9+o*9)+k*3+2];
                }
                dR_matmul(filterTransform4x3,4,3,filtersTemp,3,3,temp);
                dR_matmulT(temp,4,3,filterTransform4x3,4,3,transformedFiltersTemp+(i*shape.s3*16+o*16));

            }
            else
            {
                dR_matmul(filterTransform6x3,6,3,convlayer->weights+(i*shape.s3*9+o*9),3,3,temp);
                dR_matmulT(temp,6,3,filterTransform6x3,6,3,transformedFiltersTemp+(i*shape.s3*36+i*36));

            }
        }
    }
    // [I,O,h,w] -> [I,h,w,O]
    for(i = 0; i<shape.s2;i++)
    {
        for(o = 0; o<shape.s3;o++)
        {
            for(k = 0; k<16;k++)
            {
                transformedFilters[i*shape.s3*16+k*shape.s3+o] = transformedFiltersTemp[i*shape.s3*16+o*16+k];
            }
        }
    }//TODO transpose for 4x4

    ret &= dR_uploadArray(net,"",transformedFilters,0,((convlayer->winogradn+2)*(convlayer->winogradn+2)*shape.s2*shape.s3)*sizeof(cl_float),convlayer->weightsBuf);

    if(convlayer->useBias)
        ret &= dR_uploadArray(net,"",convlayer->biases,0,convlayer->shape.s3*sizeof(cl_float),convlayer->biasBuf);

    g_free(temp);
    g_free(transformedFilters);
    g_free(transformedFiltersTemp);

    return ret;
}

gboolean dR_conv2d_winograd_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    ret &= dR_clMemoryBufferCleanup(net, ((dR_Conv2d_Data*)(layer->layer))->weightsBuf);
    ret &= dR_clMemoryBufferCleanup(net, ((dR_Conv2d_Data*)(layer->layer))->biasBuf);
    ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_conv2d_winograd_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    if(convlayer->hasVariables)
    {
        g_free(convlayer->weights);
        if(convlayer->useBias)
        {
            g_free(convlayer->biases);
        }
    }
    g_free((dR_Conv2d_Data*)(layer->layer));
    return TRUE;
}


// Optional


gboolean dR_conv2d_winograd_generateKernel(dR_Graph* net, dR_Node* layer)
{
    dR_Conv2d_Data* convlayer = (dR_Conv2d_Data*)(layer->layer);
    gint length = 0;
    gchar* kernelname;
    gchar *temp, *source;
    gchar * filePath;
    // Variables that get hardcoded into the shader
    gint O =convlayer->shape.s3;
    gint I =convlayer->shape.s2;
    gint OPartitionSize = O/convlayer->numberOfDepthPartitions;
    gint INumPartitions = convlayer->numberOfDepthPartitionsInput;
    gint NPerDim = convlayer->winogradNperDim;
    gint N = NPerDim*NPerDim;
    gint N_or_NperDim = convlayer->winogradWide?NPerDim:N;
    gchar* g_or_l = convlayer->winogradWide?"l":"g";
    kernelname = dR_conv2d_winograd_createKernelName(layer);
    source = NULL;

    //Creating Shader Name and Return type
    // See prototype kernel dR_winograd2.cl for comments
    source = g_strdup_printf(
    "#ifndef kerneldef_%s \n"
    "#define kerneldef_%s \n\n"
    "__kernel void %s(\n"
                ,kernelname,kernelname,kernelname);
    // Rest of the Header and first part
    temp = g_strdup_printf(
    "    const __global float * biases,\n"
    "    const __global float * gI,\n"
    "    const __global float * gTransformedFilters,\n"
    "    __global  float * gO,\n"
    "    __local float* lTransformedFilters,\n"
    "    __local float* lInput,\n"
    "    int activation,    \n"
    "    int bias,          \n"
    "    int bW,            \n"
    "    int bH,            \n"
    "    int numWGsInX     \n"
    "    )\n"
    "{\n");
    source = concat_and_free_old(source,temp);
    if(convlayer->winogradWide)
    {
        temp = g_strdup_printf(
        "   for(int j = get_local_id(1)*get_local_size(0)+get_local_id(0); j<16*%d*%d;j+=get_local_size(0)*get_local_size(1))\n"
        "   {\n"
        "       lTransformedFilters[j] = gTransformedFilters[j];\n"
        "   }\n"
                    ,OPartitionSize,I);
        source = concat_and_free_old(source,temp);
    }
    temp = g_strdup_printf(
    "    const int x_p = %d*(get_group_id(0)%%numWGsInX)-1;\n"
    "    const int y_p = %d*(get_group_id(0)/numWGsInX)-1;\n"
                ,NPerDim*2,NPerDim*2);
    source = concat_and_free_old(source,temp);
    if(convlayer->winogradWide)
    {
        temp = g_strdup_printf(
        "    for(int n = 0; n<%d; n++)\n"
        "    {\n"
        "    int x_o = get_local_id(0)*2;\n"
        "    int y_o = n*2;\n"
                    ,NPerDim);
        source = concat_and_free_old(source,temp);
    }
    else
    {
        temp = g_strdup_printf(
        "    int x_o = (get_local_id(0)%% %d)*2;\n"
        "    int y_o = (get_local_id(0)/%d)*2;\n"
                    ,NPerDim,NPerDim);
        source = concat_and_free_old(source,temp);
    }
    temp = g_strdup_printf(
    "    int ox = x_p+1+x_o;\n"
    "    int oy = y_p+1+y_o;\n"
    "    int index = (get_group_id(1)*%d+get_local_id(1))*(bW*bH)+oy*bW+ox;\n"
                ,OPartitionSize);
    source = concat_and_free_old(source,temp);

    temp = g_strdup_printf(
    "    if(get_local_id(1)*%d<%d)\n"
    "    {\n"
    "        for(int i = 0;i< %d;i++)\n"
    "        {\n"
    "            float m[16];\n"
    "            float temp[8];\n"
    "            int z_o = ((get_local_id(1)*%d+i)* bW*bH);\n"
                ,INumPartitions,I,INumPartitions,INumPartitions);
    source = concat_and_free_old(source,temp);

    temp = g_strdup_printf(
    "            temp[0]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+1, bW, bH);\n"
    "            temp[1]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+1, bW, bH);\n"
    "            temp[2]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+1, bW, bH);\n"
    "            temp[3]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+1, bW, bH);\n"
    "\n"
    "            temp[4]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+2, bW, bH);\n"
    "            temp[5]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+2, bW, bH);\n"
    "            temp[6]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+2, bW, bH);\n"
    "            temp[7]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+2, bW, bH);\n"
    "\n"
    "            m[0]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o, bW, bH)   - temp[4];\n"
    "            m[1]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o, bW, bH) - temp[5];\n"
    "            m[2]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o, bW, bH) - temp[6];\n"
    "            m[3]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o, bW, bH) - temp[7];\n"
    "\n"
    "            m[4]  =  temp[0] + temp[4];\n"
    "            m[5]  =  temp[1] + temp[5];\n"
    "            m[6]  =  temp[2] + temp[6];\n"
    "            m[7]  =  temp[3] + temp[7];\n"
    "\n"
    "            m[8]  = -temp[0] + temp[4];\n"
    "            m[9]  = -temp[1] + temp[5];\n"
    "            m[10] = -temp[2] + temp[6];\n"
    "            m[11] = -temp[3] + temp[7];\n"
    "\n"
    "            m[12] = -temp[0] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+3, bW, bH);\n"
    "            m[13] = -temp[1] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+3, bW, bH);\n"
    "            m[14] = -temp[2] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+3, bW, bH);\n"
    "            m[15] = -temp[3] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+3, bW, bH);\n"
    "\n"
    "            temp[0] = m[1];\n"
    "            m[0]  =  m[0]     - m[2];\n"
    "            m[1]  =  m[1]     + m[2];\n"
    "            m[2]  = -temp[0]  + m[2];\n"
    "            m[3]  = -temp[0]  + m[3];\n"
    "\n"
    "            temp[0] = m[5];\n"
    "            m[4]  =  m[4]     - m[6];\n"
    "            m[5]  =  m[5]     + m[6];\n"
    "            m[6]  = -temp[0]  + m[6];\n"
    "            m[7]  = -temp[0]  + m[7];\n"
    "\n"
    "            temp[0] = m[9];\n"
    "            m[8]  =  m[8]     - m[10];\n"
    "            m[9]  =  m[9]     + m[10];\n"
    "            m[10] = -temp[0]  + m[10];\n"
    "            m[11] = -temp[0]  + m[11];\n"
    "\n"
    "            temp[0] = m[13];\n"
    "            m[12] =  m[12]    - m[14];\n"
    "            m[13] =  m[13]    + m[14];\n"
    "            m[14] = -temp[0]  + m[14];\n"
    "            m[15] = -temp[0]  + m[15];\n"
    "\n"
    "            #pragma unroll\n"
    "            for(int k = 0; k<16;k++)\n"
    "            {\n"
    "                lInput[(get_local_id(1)*%d+i)*%d+k*%d+get_local_id(0)] = m[k];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                ,INumPartitions,N_or_NperDim*16,N_or_NperDim);
    source = concat_and_free_old(source,temp);
    temp = g_strdup_printf(
    "    if(get_local_id(1)<%d&&ox<bW&&oy<bH)\n"
    "    {\n"

    "        float m[16] = {0.0};\n"
    "        int f_ind = get_group_id(1)*%d;\n"
    "        int i_ind = 0;\n"
    "        //#pragma unroll\n"
    "        for(int i = 0; i<%d;i++)\n"
    "        {\n"
    "            #pragma unroll\n"
    "            for(int j = 0; j<16; j++)\n"
    "            {\n"
    "                m[j] = mad(lInput[i_ind+get_local_id(0)],%sTransformedFilters[f_ind+get_local_id(1)],m[j]);\n"
    "                f_ind += %d;\n"
    "                i_ind += %d;\n"
    "            }\n"
    "        }\n"
                ,OPartitionSize,OPartitionSize,I,g_or_l,O,N_or_NperDim);
    source = concat_and_free_old(source,temp);
    temp = g_strdup_printf(
    "        m[0] =  m[0]  + m[1]  + m[2];\n"
    "        m[1] =  m[1]  - m[2]  + m[3];\n"
    "        m[2] =  m[4]  + m[5]  + m[6];\n"
    "        m[3] =  m[5]  - m[6]  + m[7];\n"
    "        m[4] =  m[8]  + m[9]  + m[10];\n"
    "        m[5] =  m[9]  - m[10] + m[11];\n"
    "        m[6] =  m[12] + m[13] + m[14];\n"
    "        m[7] =  m[13] - m[14] + m[15];\n"

    "        m[0] =  m[0] + m[2] + m[4];\n"
    "        m[1] =  m[1] + m[3] + m[5];\n"
    "        m[2] =  m[2] - m[4] + m[6];\n"
    "        m[3] =  m[3] - m[5] + m[7];\n"

    "        if(bias)\n"
    "        {\n"
    "            float bias = biases[get_group_id(1)*%d+get_local_id(1)];\n"
    "            m[0] += bias;\n"
    "            m[1] += bias;\n"
    "            m[2] += bias;\n"
    "            m[3] += bias;\n"
    "        }\n"
                ,OPartitionSize);
    source = concat_and_free_old(source,temp);
    temp = g_strdup_printf(
    "        if(activation)\n"
    "        {\n"
    "            m[0] = clamp(m[0], 0.0f, FLT_MAX);\n"
    "            m[1] = clamp(m[1], 0.0f, FLT_MAX);\n"
    "            m[2] = clamp(m[2], 0.0f, FLT_MAX);\n"
    "            m[3] = clamp(m[3], 0.0f, FLT_MAX);\n"
    "        }\n"
    "        gO[index] = m[0];\n"
    "        gO[index+1] = m[1];\n"
    "        gO[index+bW] = m[2];\n"
    "        gO[index+bW+1] = m[3];\n"
    "    }\n");
    source = concat_and_free_old(source,temp);
    if(convlayer->winogradWide)
    {
        temp = g_strdup_printf(
        "    }\n");
        source = concat_and_free_old(source,temp);
    }
    temp = g_strdup_printf(
    "}\n"
    "#endif \n\n");
    source = concat_and_free_old(source,temp);

    length = strlen(source);
    filePath = g_build_filename(net->config->modelPath, dR_conv2d_winograd_createKernelName(layer), NULL);
    g_file_set_contents(filePath,source,length,NULL);

    g_free(source);

    return TRUE;
}

gchar* dR_conv2d_winograd_createKernelName(dR_Node* layer)
{
    dR_Conv2d_Data* convlayer = (dR_Conv2d_Data*)(layer->layer);

    gint O =convlayer->shape.s3;
    gint I =convlayer->shape.s2;
    gint OPartitionSize = O/convlayer->numberOfDepthPartitions;
    gint INumPartitions = convlayer->numberOfDepthPartitionsInput;
    gint N = convlayer->winogradN;
    gchar* wide = convlayer->winogradWide?"_wide":"";
    gchar* string = g_strdup_printf("conv2dwino%dx%d_%dx%d_%d%s", I, O, INumPartitions, OPartitionSize, N, wide);
    return string;
}



// 1x1 Implementation


gboolean dR_conv2d_1x1_schedule(dR_Graph* net, dR_Node* layer){

    // Trying to calculate [1,1,inputDepth,z] in one Thread and
    // [x,y,inputDepth,z] in one Workgroup

    // min f(x,y,z)= (width/x)*(height/y)*(outputDepth/z)*((x+filterx-1)*(y+filtery-1)*inputDepth + filterx*filtery*inputDepth*z)   // number of global memory accesses
    // subject to:
    // 8<=x<=width, 8<=y<=height, 1<=z<=outputDepth,
    // ((x+filterx-1)*(y+filtery-1) + filterx*filtery*z)*4<localmemorysize  // restriction based on available local memory
    // x*y = 32*k for k € N
    // x*y <= maxworkgroupsize
    // (width/x)*(height/y)*(outputDepth/z)>60 Minimum 60 Workgroups

    // Minimizing global memory accesses subject to available local memory


    //size_t maxwgs;
    //clGetKernelWorkGroupInfo(layer->clKernel, net->clConfig->clDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxwgs,NULL);

    dR_Conv2d_Data* convlayer = (dR_Conv2d_Data*)(layer->layer);
    gint lws = 16;
    cl_ulong lms;
    size_t mws;
    gint numberofwgs;
    convlayer->useLMEM = FALSE;



    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lms, NULL);
    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &mws, NULL);

    // Find maximum size of workgroup in spatial dimension based on memory constraints
    if(!net->config->silent&&net->config->debugInfo)
        g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);
    while(lws<(gint)layer->oshape.s0&&lws<(gint)layer->oshape.s1&&lws*lws<=(gint)mws){
        lws*=2;
    }
    lws/=2;


    if(layer->oshape.s0%lws==0)
    {
        convlayer->globalWorkSizeX = layer->oshape.s0;
    }
    else
    {
        convlayer->globalWorkSizeX = ((layer->oshape.s0/lws)+1)*lws;
    }


    if(layer->oshape.s1%lws==0)
    {
        convlayer->globalWorkSizeY = layer->oshape.s1;
    }
    else
    {
        convlayer->globalWorkSizeY = ((layer->oshape.s1/lws)+1)*lws;
    }

    convlayer->gidXGap = layer->oshape.s0 - convlayer->globalWorkSizeX;
    convlayer->gidYGap = layer->oshape.s1 - convlayer->globalWorkSizeY;

    numberofwgs = ((convlayer->globalWorkSizeX/lws) * (convlayer->globalWorkSizeY/lws))*convlayer->shape.s3;

    convlayer->localWorkSizexy = lws;

    if(!net->config->silent&&net->config->debugInfo)
        g_print("LocalWorkSizexy: %d \n", (gint)convlayer->localWorkSizexy);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Scheduled Conv2D 1x1 Layer \n");

    return TRUE;
 }


gboolean dR_conv2d_1x1_compute(dR_Graph* net, dR_Node* layer){

    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    cl_int lMemFilterSize;
    cl_int numO, numI, activation, usebias, bW, hW;
    size_t localWorkSize[3];
    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;

    localWorkSize[0] = convlayer->localWorkSizexy;
    localWorkSize[1] = convlayer->localWorkSizexy;;
    localWorkSize[2] = 1;
    lMemFilterSize = (convlayer->shape.s2);
    globalWorkSize[0] = convlayer->globalWorkSizeX;
    globalWorkSize[1] = convlayer->globalWorkSizeY;
    globalWorkSize[2] = convlayer->shape.s3;
    usebias = (convlayer->useBias?1:0);
    numO = convlayer->shape.s3;
    numI = convlayer->shape.s2;
    bW = convlayer->ishape.s0;
    hW = convlayer->ishape.s1;
    if(convlayer->activation==tReLU)
    {
        activation = 1;
    }
    else
    {
        activation = 0;
    }

    /* Kernel Parameters
    const __global float * biases,
    const __global  float * gInputImage,
    const __global float * gFilter,
    __global  float * gOutputImage,
    __local float* lFilter,
    int bias,           // 0 no bias, 1 bias
    int activation,     // 0 linear, 1 relu
    int numO,              // Number of Filters
    int numI ,             // Number of Input Channels
    int bW,             // Input Buffer Width
    int bH             // Input Buffer Height
    */
    net->clConfig->clError = FALSE;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)&convlayer->biasBuf);        paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)&convlayer->weightsBuf);     paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);   paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemFilterSize * sizeof(cl_float), NULL);            paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&usebias);                   paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&activation);                paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&numO);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&numI);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&bW);                        paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&hW);                        paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Conv2d_1x1 Kernel"))
        return FALSE;
    //execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        localWorkSize, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:conv2d1x1");

}



gboolean dR_conv2d_1x1_createKernel(dR_Graph* net, dR_Node* layer)
{
    return dR_createKernel(net,"conv2d1x1",&(layer->clKernel));
}


gboolean dR_conv2d_1x1_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_Shape4 shape;
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    shape = convlayer->shape;
    ret &= dR_createFloatBuffer(net, &(((dR_Conv2d_Data*)(layer->layer))->weightsBuf),shape.s0*shape.s1*shape.s2*shape.s3, CL_MEM_READ_ONLY);
    if(convlayer->useBias)
        ret &= dR_createFloatBuffer(net, &(((dR_Conv2d_Data*)(layer->layer))->biasBuf),shape.s3, CL_MEM_READ_ONLY);
    else
        ret &= dR_createFloatBuffer(net, &(((dR_Conv2d_Data*)(layer->layer))->biasBuf),1, CL_MEM_READ_ONLY);
    return ret;
}

gboolean dR_conv2d_1x1_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    gint i, o;
    dR_Shape4 shape = convlayer->shape;
    gfloat* transformedWeights = g_malloc(sizeof(float)*shape.s2*shape.s3);
    // Filter-layout [I,O] -> [O,I] for specialized 1x1 kernel
    for(i = 0; i<shape.s2;i++)
    {
        for(o = 0; o<shape.s3;o++)
        {
            transformedWeights[o*shape.s2+i] = convlayer->weights[i*shape.s3+o];
        }
    }
    ret &= dR_uploadArray(net,"",transformedWeights,0,convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s2*convlayer->shape.s3*sizeof(cl_float),convlayer->weightsBuf);
    if(convlayer->useBias)
        ret &= dR_uploadArray(net,"",convlayer->biases,0,convlayer->shape.s3*sizeof(cl_float),convlayer->biasBuf);
    g_free(transformedWeights);
    return ret;
}

gboolean dR_conv2d_1x1_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    ret &= dR_clMemoryBufferCleanup(net, ((dR_Conv2d_Data*)(layer->layer))->weightsBuf);
    ret &= dR_clMemoryBufferCleanup(net, ((dR_Conv2d_Data*)(layer->layer))->biasBuf);
    ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_conv2d_1x1_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    dR_Conv2d_Data* convlayer = ((dR_Conv2d_Data*)(layer->layer));
    if(convlayer->hasVariables)
    {
        g_free(convlayer->weights);
        if(convlayer->useBias)
        {
            g_free(convlayer->biases);
        }
    }
    g_free((dR_Conv2d_Data*)(layer->layer));
    return TRUE;
}






// ////////////////////////////////////
//  2D Convolutional Layer Transpose //
// ////////////////////////////////////


// Mandatory

dR_Node* dR_Conv2dtransposed(dR_Graph* net, dR_Node* inputLayer, dR_Shape4 sh, dR_Shape4 st, dR_ActivationType acttype, gboolean useB){
    dR_Conv2dTranspose_Data* conv2dt = g_malloc(sizeof(dR_Conv2dTranspose_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    conv2dt->shape = sh;
    conv2dt->stride = st;
    conv2dt->activation = acttype;
    conv2dt->useBias = useB;
    conv2dt->hasVariables = FALSE;
    l->layer = conv2dt;
    l->type = tConv2d;

    l->compute = dR_conv2dtranspose_compute;
    l->schedule = dR_conv2dtranspose_schedule;
    l->propagateShape = dR_conv2dtranspose_propagateShape;
    l->getRequiredOutputBufferSize = dR_conv2dtranspose_getRequiredOutputBufferSize;
    l->createKernel = dR_conv2dtranspose_createKernel;
    l->allocateBuffers = dR_conv2dtranspose_allocateBuffers;
    l->fillBuffers = dR_conv2dtranspose_fillBuffers;
    l->cleanupBuffers = dR_conv2dtranspose_cleanupBuffers;
    l->cleanupLayer = dR_conv2dtranspose_cleanupLayer;
    l->serializeNode = dR_conv2dtranspose_serializeNode;
    l->parseAppendNode = dR_conv2dtranspose_parseAppendNode;

    l->generateKernel = dR_conv2dtranspose_generateKernel;
    l->setVariables = dR_Conv2dtransposed_setVariables;
    l->createKernelName = dR_conv2dtranspose_createKernelName;
    l->printLayer = dR_conv2dtranspose_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: Conv2d needs na appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}


gchar* dR_conv2dtranspose_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_Conv2d_Data* convlayer = (dR_Conv2d_Data*)(layer->layer);
    gchar* desc = "Conv2D";
    gint numNodeParams = 10;
    gint numNodeVariables;
    if(convlayer->useBias)
    {
        numNodeVariables = 2;
    }
    else
    {
        numNodeVariables = 1;
    }
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",convlayer->activation);
    params[1] = g_strdup_printf("%d",convlayer->useBias?1:0);
    params[2] = g_strdup_printf("%d",convlayer->shape.s0);
    params[3] = g_strdup_printf("%d",convlayer->shape.s1);
    params[4] = g_strdup_printf("%d",convlayer->shape.s2);
    params[5] = g_strdup_printf("%d",convlayer->shape.s3);
    params[6] = g_strdup_printf("%d",convlayer->stride.s0);
    params[7] = g_strdup_printf("%d",convlayer->stride.s1);
    params[8] = g_strdup_printf("%d",convlayer->stride.s2);
    params[9] = g_strdup_printf("%d",convlayer->stride.s3);

    *numVariables = numNodeVariables;
    variableSizes[0] = convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s2*convlayer->shape.s3;
    variables[0] = convlayer->weights;
    if(convlayer->useBias)
    {
        variableSizes[1] = convlayer->shape.s3;
        variables[1] = convlayer->biases;
    }
    return desc;
}

dR_Node* dR_conv2dtranspose_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 10;
    gint numNodeVariables = 2;
    dR_Shape4 shape;
    dR_Shape4 stride;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: Conv2d Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables<numNodeVariables-1||numVariables>numNodeVariables)
    {
        g_print("Parsing Error: Conv2d Node needs %d Parameters and %d or %d Variables!\n",numNodeParams,numNodeVariables-1,numNodeVariables);
        return NULL;
    }
    shape.s0 = atoi(params[2]);
    shape.s1 = atoi(params[3]);
    shape.s2 = atoi(params[4]);
    shape.s3 = atoi(params[5]);
    stride.s0 = atoi(params[6]);
    stride.s1 = atoi(params[7]);
    stride.s2 = atoi(params[8]);
    stride.s3 = atoi(params[9]);
    out = dR_Conv2dtransposed(net, iNodes[0], shape, stride ,atoi(params[0]), atoi(params[1]));
    dR_Conv2dtransposed_setVariables(out,variables[0],atoi(params[1])?variables[1]:NULL);
    return out;
}


gboolean dR_conv2dtranspose_schedule(dR_Graph* net, dR_Node* layer){

    // Trying to calculate [1,1,inputDepth,z] in one Thread and
    // [x,y,inputDepth,z] in one Workgroup

    // min f(x,y,z)= (width/x)*(height/y)*(outputDepth/z)*((x+filterx-1)*(y+filtery-1)*inputDepth + filterx*filtery*inputDepth*z)   // number of global memory accesses
    // subject to:
    // 8<=x<=width, 8<=y<=height, 1<=z<=outputDepth,
    // ((x+filterx-1)*(y+filtery-1) + filterx*filtery*z)*4<localmemorysize  // restriction based on available local memory
    // x*y = 32*k for k € N
    // x*y <= maxworkgroupsize
    // (width/x)*(height/y)*(outputDepth/z)>60 Minimum 60 Workgroups

    // Minimizing global memory accesses subject to available local memory


    //size_t maxwgs;
    //clGetKernelWorkGroupInfo(layer->clKernel, net->clConfig->clDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxwgs,NULL);

    dR_Conv2dTranspose_Data* convlayer = (dR_Conv2dTranspose_Data*)(layer->layer);
    gint lws = 16;
    cl_ulong lms;
	size_t mws;
    gint numberofwgs;
    gint localMemoryConFilter = (convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s3*4);
    gint localMemoryConInput = ((lws+(convlayer->shape.s0-1))*(lws+(convlayer->shape.s0-1))*4);
    gint z=1;
    if(convlayer->shape.s0<=convlayer->stride.s1&&convlayer->shape.s1<=convlayer->stride.s2)
        convlayer->useLMEM = FALSE;
    else
        convlayer->useLMEM = TRUE;



    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lms, NULL);
    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &mws, NULL);

    // Find maximum size of workgroup in spatial dimension based on memory constraints
    if(!net->config->silent)
        g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);
    while(localMemoryConInput+localMemoryConFilter<(gint)lms&&(gint)lws*(gint)lws<=(gint)mws){
        lws*=2;
        localMemoryConInput = ((lws+(convlayer->shape.s0-1))*(lws+(convlayer->shape.s0-1))*4);
    }
    lws/=2;


    if(layer->oshape.s0%lws==0)
    {
        convlayer->globalWorkSizeX = layer->oshape.s0;
    }
    else
    {
        convlayer->globalWorkSizeX = ((layer->oshape.s0/lws)+1)*lws;
    }


    if(layer->oshape.s1%lws==0)
    {
        convlayer->globalWorkSizeY = layer->oshape.s1;
    }
    else
    {
        convlayer->globalWorkSizeY = ((layer->oshape.s1/lws)+1)*lws;
    }

    convlayer->gidXGap = layer->oshape.s0 - convlayer->globalWorkSizeX;
    convlayer->gidYGap = layer->oshape.s1 - convlayer->globalWorkSizeY;

    numberofwgs = ((convlayer->globalWorkSizeX/lws) * (convlayer->globalWorkSizeY/lws))*z;


    // If Number of Workgroups based on spatial division is to small -> parallize in filter dimension
    while(numberofwgs<32&&z<convlayer->shape.s3)
    {
        z++;
        while(convlayer->shape.s3%z!=0)
            z++;
        numberofwgs = ((convlayer->globalWorkSizeX/lws) * (convlayer->globalWorkSizeY/lws))*z;
    }
    //z = 1;
    convlayer->localWorkSizexy = lws;
    convlayer->numberOfDepthPartitions = z;

    if(!net->config->silent&&net->config->debugInfo)
        g_print("LocalWorkSizexy: %d \n", (gint)convlayer->localWorkSizexy);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Number of Depth-Partitions: %d \n", convlayer->numberOfDepthPartitions);


    if(!net->config->silent&&net->config->debugInfo)
        g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Scheduled Conv2D Layer \n");

    return TRUE;
 }


gboolean dR_conv2dtranspose_compute(dR_Graph* net, dR_Node* layer){
    dR_Conv2dTranspose_Data* convlayer = ((dR_Conv2dTranspose_Data*)(layer->layer));
    cl_int filterWindowWidth = convlayer->shape.s0;
    cl_int filterWindowHeight = convlayer->shape.s1;
    cl_int oDepth = convlayer->shape.s3;
    //cl_int iwidth = convlayer->ishape.s0;
    //cl_int iheight = convlayer->ishape.s1;
    //cl_int idepth = convlayer->ishape.s2;
    cl_int lMemImageSize, lMemFilterSize;
    size_t localWorkSize[3];
    size_t globalWorkSize[3];
    gint paramid = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;

    localWorkSize[0] = convlayer->localWorkSizexy;
    localWorkSize[1] = convlayer->localWorkSizexy;;
    localWorkSize[2] = 1;
    lMemImageSize = (localWorkSize[0] + filterWindowWidth - 1) * (localWorkSize[1] + filterWindowHeight - 1);
    lMemFilterSize = filterWindowHeight*filterWindowWidth*(oDepth/convlayer->numberOfDepthPartitions);
    globalWorkSize[0] = convlayer->globalWorkSizeX;
    globalWorkSize[1] = convlayer->globalWorkSizeY;
    globalWorkSize[2] = convlayer->numberOfDepthPartitions;

    if ((filterWindowWidth-1)/2 > (cl_int)localWorkSize[0]
    || (filterWindowHeight-1) > (cl_int)localWorkSize[1])
    {
        g_print("Error: Kernel size for convolution is too big (is %ix%i; max %ix%i)\n", filterWindowWidth, filterWindowHeight, (gint)(localWorkSize[0]), (gint)(localWorkSize[1]));
        return FALSE;
    }

    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)&convlayer->weightsBuf);     paramid++;
    if(convlayer->useBias)
    {
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)&convlayer->biasBuf);        paramid++;
    }
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);          paramid++;
    if(convlayer->useLMEM)
    {
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemImageSize * sizeof(cl_float), NULL);         paramid++;
    }
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemFilterSize * sizeof(cl_float), NULL);            paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&iwidth);                    paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&iheight);                   paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&idepth);                    paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&filterWindowWidth);         paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&filterWindowHeight);        paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&oDepth);                    paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Conv2d Kernel"))
        return FALSE;
    //execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        localWorkSize, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:conv2d");

}

gboolean dR_conv2dtranspose_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_Conv2dTranspose_Data* convlayer = ((dR_Conv2dTranspose_Data*)(layer->layer));
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("Conv2d Layer with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    convlayer->ishape.s0 = lastlayer->oshape.s0;
    convlayer->ishape.s1 = lastlayer->oshape.s1;
    convlayer->ishape.s2 = lastlayer->oshape.s2;

    layer->oshape.s0 = convlayer->ishape.s0/convlayer->stride.s1;
    layer->oshape.s1 = convlayer->ishape.s1/convlayer->stride.s2;
    layer->oshape.s2 = convlayer->shape.s3;

    if(layer->oshape.s0<=0 || layer->oshape.s1<=0 || layer->oshape.s2<=0)
    {
        if(!net->config->silent)
            g_print("Conv2d shape propagates to Zero. False shape parameters! \n");
        return FALSE;
    }
    return TRUE;
}

gint32 dR_conv2dtranspose_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_conv2dtranspose_createKernel(dR_Graph* net, dR_Node* layer)
{
    //dR_Conv2dTranspose_Data* convlayer = (dR_Conv2dTranspose_Data*)(layer->layer);
    return dR_createKernel(net,dR_conv2dtranspose_createKernelName(layer),&(layer->clKernel));
}


gboolean dR_conv2dtranspose_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_Shape4 shape;
    dR_Conv2dTranspose_Data* convlayer = ((dR_Conv2dTranspose_Data*)(layer->layer));
    shape = convlayer->shape;
    ret &= dR_createFloatBuffer(net, &(((dR_Conv2dTranspose_Data*)(layer->layer))->weightsBuf),shape.s0*shape.s1*shape.s2*shape.s3, CL_MEM_READ_ONLY);
    if(convlayer->useBias)
        ret &= dR_createFloatBuffer(net, &(((dR_Conv2dTranspose_Data*)(layer->layer))->biasBuf),shape.s3, CL_MEM_READ_ONLY);
    return ret;
}

gboolean dR_conv2dtranspose_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_Conv2dTranspose_Data* convlayer = ((dR_Conv2dTranspose_Data*)(layer->layer));
    ret &= dR_uploadArray(net,"",convlayer->weights,0,convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s2*convlayer->shape.s3*sizeof(cl_float),convlayer->weightsBuf);
    if(convlayer->useBias)
        ret &= dR_uploadArray(net,"",convlayer->biases,0,convlayer->shape.s3*sizeof(cl_float),convlayer->biasBuf);
    return ret;
}

gboolean dR_conv2dtranspose_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    ret &= dR_clMemoryBufferCleanup(net, ((dR_Conv2dTranspose_Data*)(layer->layer))->weightsBuf);
    ret &= dR_clMemoryBufferCleanup(net, ((dR_Conv2dTranspose_Data*)(layer->layer))->biasBuf);
    ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_conv2dtranspose_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    dR_Conv2dTranspose_Data* convlayer = ((dR_Conv2dTranspose_Data*)(layer->layer));
    if(convlayer->hasVariables)
    {
        g_free(convlayer->weights);
        if(convlayer->useBias)
        {
            g_free(convlayer->biases);
        }
    }
    g_free((dR_Conv2dTranspose_Data*)(layer->layer));
    return TRUE;
}


// Optional

void dR_Conv2dtransposed_setVariables(dR_Node* layer, gfloat* weights, gfloat* biases)
{
    dR_Conv2dTranspose_Data* convlayer = ((dR_Conv2dTranspose_Data*)(layer->layer));
    int i;
    convlayer->weights = g_malloc(convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s2*convlayer->shape.s3*sizeof(gfloat));
    for(i=0; i<convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s2*convlayer->shape.s3;i++)
    {
        convlayer->weights[i] = weights[i];
    }
    if(convlayer->useBias)
    {
        convlayer->biases = g_malloc(convlayer->shape.s3*sizeof(gfloat));
        for(i=0; i<convlayer->shape.s3;i++)
        {
            convlayer->biases[i] = biases[i];
        }
    }
    convlayer->hasVariables = 1;
}


gboolean dR_conv2dtranspose_generateKernel(dR_Graph* net, dR_Node* layer)
{
    //Placeholder to suppress warnings
    net = net;
    layer = layer;
    /*
    dR_Conv2dTranspose_Data* convlayer = (dR_Conv2dTranspose_Data*)(layer->layer);
    gint length = 0;

    gchar* kernelname = dR_conv2dtranspose_createKernelName(net,layer);
    gchar* temp;
    gchar* source;
    gchar * filePath;

    // Variables that get hardcoded into the shader
    gint width =convlayer->ishape.s0;
    gint height =convlayer->ishape.s1;
    gint inputDepth =convlayer->ishape.s2;
    gint widthheight = width*height;
    gint filterWidth = convlayer->shape.s0;
    gint filterHeight = convlayer->shape.s1;
    gint outputDepth = convlayer->shape.s3;
    gint x = -filterWidth/2;
    gint y = -filterHeight/2;
    gint lengthindepth = outputDepth/convlayer->numberOfDepthPartitions;
    gint filterxy = filterWidth*filterHeight;
    gint floatsToReadInThisWG = (convlayer->localWorkSizexy+filterWidth-1)*(convlayer->localWorkSizexy+filterHeight-1);
    gint numberOfWorkItems = convlayer->localWorkSizexy*convlayer->localWorkSizexy;
    gint copyWidth = convlayer->localWorkSizexy+ filterWidth-1;
    gint jumpint = convlayer->localWorkSizexy-1;
    gint fid = 0;
    gint outputindex,yid,xid;

    //Creating Shader Name and Return type


    filePath = g_build_filename(net->config->modelPath, dR_conv2dtranspose_createKernelName(net,layer), NULL);
    g_file_set_contents(filePath,string,length,NULL);

    g_print("Kernel Length Conv: %d \n", length);
    g_free(string);
    g_free(string1);
    g_free(string2);
    g_free(string3);
    */
    return TRUE;
}

gchar* dR_conv2dtranspose_createKernelName(dR_Node* layer)
{
    dR_Conv2dTranspose_Data* convlayer = (dR_Conv2dTranspose_Data*)(layer->layer);
    gchar* string,*temp;
    string = g_strdup_printf("conv2d%dx%dx%d_%d_", convlayer->shape.s0, convlayer->shape.s1,convlayer->shape.s3,layer->layerID);
    if(convlayer->useLMEM)
    {
        temp = g_strdup_printf("lmem");
        string = concat_and_free_old(string,temp);
    }

    switch(convlayer->activation){
    case tReLU:
        temp = g_strdup_printf("ReLU");
        break;
    case tLinear:
        temp = g_strdup_printf("Linear");
        break;
    case tSigmoid:
        temp = g_strdup_printf("Sigmoid");
        break;
    case tTan:
        temp = g_strdup_printf("Tan");
        break;
    }
    string = concat_and_free_old(string,temp);
    return string;
}


gchar* dR_conv2dtranspose_printLayer(dR_Node* layer)
{
    dR_Conv2dTranspose_Data* convlayer = (dR_Conv2dTranspose_Data*)(layer->layer);
    gchar* actstr;
    gchar* biasstr;
    gchar* out;
    if(convlayer->activation == tReLU)
        actstr = "ReLU";
    else if(convlayer->activation == tLinear)
        actstr = "Linear";
    else if(convlayer->activation == tSigmoid)
        actstr = "Sigmoid";
    else if(convlayer->activation == tTan)
        actstr = "Tan";
    if(convlayer->useBias)
        biasstr = "Yes";
    else
        biasstr = "No";
    out = g_strdup_printf("%s%d%s%d%s%d%s%d%s%d%s%s%s%d%s%d%s%d%s%d%s%s%s","Conv2d Layer: ", layer->layerID,"\n Filter dimensions: ", convlayer->shape.s0,"x",convlayer->shape.s1,
            "\n Input channels: ",convlayer->shape.s2,"\n Output channels: ", convlayer->shape.s3,
            "\n Activation: ",actstr,"\n Stride: ", convlayer->stride.s0, ", ", convlayer->stride.s1, ", ", convlayer->stride.s2,", ",convlayer->stride.s3,
            "\n Use bias: ", biasstr,"\n");
    /*if(printVariables&&convlayer->hasVariables)
    {
        char buffer[5];
        char weights[1000000];
        int i;
        for(i = 0; i<convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s2*convlayer->shape.s3;i++)
        {
            sprintf(buffer,"%.3f, ", convlayer->weights[i]);
            strcat(weights,buffer);
        }
        strcat(out,"Weights:\n");
        strcat(out,weights);
        strcat(out,"\n\n");
        if(convlayer->useBias)
        {
            for(i = 0; i<convlayer->shape.s3;i++)
            {
                sprintf(buffer,"%.3f, ", convlayer->biases[i]);
                strcat(weights,buffer);
            }
            strcat(out,"Biases:\n");
            strcat(out,weights);
            strcat(out,"\n\n");
        }
    }*/
    return out;
}


