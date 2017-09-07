#include "dR_nodes_filter.h"
#include "dR_core.h"

// ///////////////////////////////////
// Apply Context-dependent filter   //
// ///////////////////////////////////


// Mandatory

dR_Node* dR_MaskDependentFilter(dR_Graph* net, dR_Node* inputImage, dR_Node* inputFilterMask, dR_Shape3* sh){
    dR_MDFilter_Data* conv2d = g_malloc(sizeof(dR_MDFilter_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    conv2d->shape.s0 = sh->s0;
    conv2d->shape.s1 = sh->s1;
    conv2d->shape.s2 = sh->s2;
    l->layer = conv2d;
    l->type = tMaskDependentFilter;

    l->compute = dR_cdfilter_compute;
    l->schedule = dR_cdfilter_schedule;
    l->propagateShape = dR_cdfilter_propagateShape;
    l->getRequiredOutputBufferSize = dR_cdfilter_getRequiredOutputBufferSize;
    l->createKernel = dR_cdfilter_createKernel;
    l->allocateBuffers = dR_cdfilter_allocateBuffers;
    l->fillBuffers = dR_cdfilter_fillBuffers;
    l->cleanupBuffers = dR_cdfilter_cleanupBuffers;
    l->cleanupLayer = dR_cdfilter_cleanupLayer;
    l->serializeNode = dR_cdfilter_serializeNode;
    l->parseAppendNode = dR_cdfilter_parseAppendNode;

    l->generateKernel = dR_cdfilter_generateKernel;
    l->setVariables = dR_MaskDependentFilter_setVariables;
    l->createKernelName = dR_cdfilter_createKernelName;
    l->printLayer = dR_cdfilter_printLayer;

    if(inputImage&&inputFilterMask)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputImage);
        dR_list_append(l->previous_layers,inputFilterMask);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputImage->next_layers,l);
        dR_list_append(inputFilterMask->next_layers,l);
    }
    else
    {
        g_print("Error: CDFilter Layer needs 2 appropriate Inputnodes");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_cdfilter_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_MDFilter_Data* mdflayer = (dR_MDFilter_Data*)(layer->layer);
    gchar* desc = "MaskDependentFilter";
    gint numNodeParams = 3;
    gint numNodeVariables = 1;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",mdflayer->shape.s0);
    params[1] = g_strdup_printf("%d",mdflayer->shape.s1);
    params[2] = g_strdup_printf("%d",mdflayer->shape.s2);

    *numVariables = numNodeVariables;
    variableSizes[0] = mdflayer->shape.s0*mdflayer->shape.s1*mdflayer->shape.s2;
    variables[0] = mdflayer->filters;
    return desc;
}

dR_Node* dR_cdfilter_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 2;
    gint numNodeParams = 3;
    gint numNodeVariables = 1;
    dR_Shape3* shape;
    dR_Node* out;
    shape = g_malloc(sizeof(dR_Shape3));
    if(numINodes!=1)
    {
        g_print("Parsing Error: MaskDependentFilter Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables<numNodeVariables-1||numVariables>numNodeVariables)
    {
        g_print("Parsing Error: MaskDependentFilter Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    shape->s0 = atoi(params[0]);
    shape->s1 = atoi(params[1]);
    shape->s2 = atoi(params[2]);
    out = dR_MaskDependentFilter(net, iNodes[0], iNodes[1], shape);
    dR_MaskDependentFilter_setVariables(out,variables[0],NULL);
    g_free(shape);
    return out;
}

gboolean dR_cdfilter_schedule(dR_Graph* net, dR_Node* layer){

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

    dR_MDFilter_Data* cdfilterlayer = (dR_MDFilter_Data*)(layer->layer);
    gint lws = 16;
    cl_ulong lms;
	size_t mws;
    gint localMemoryConFilter = (cdfilterlayer->shape.s0*cdfilterlayer->shape.s1*cdfilterlayer->shape.s2*4);
    gint localMemoryConInput = ((lws+(cdfilterlayer->shape.s0-1))*(lws+(cdfilterlayer->shape.s1-1))*4);

    if(cdfilterlayer->shape.s0<=1&&cdfilterlayer->shape.s1<=1)
        cdfilterlayer->useLMEM = FALSE;
    else
        cdfilterlayer->useLMEM = TRUE;



    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lms, NULL);
    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &mws, NULL);

    if(!net->config->silent)
        g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);
    while(localMemoryConInput+localMemoryConFilter<(gint)lms&&lws*lws<=(gint)mws){
        lws*=2;
        localMemoryConInput = ((lws+(cdfilterlayer->shape.s0-1))*(lws+(cdfilterlayer->shape.s1-1))*4);
    }
    lws/=2;
    cdfilterlayer->localWorkSizexy = lws;

    if(layer->oshape.s0%lws==0)
    {
        cdfilterlayer->globalWorkSizeX = layer->oshape.s0;
    }
    else
    {
        cdfilterlayer->globalWorkSizeX = ((layer->oshape.s0/lws)+1)*lws;
    }


    if(layer->oshape.s1%lws==0)
    {
        cdfilterlayer->globalWorkSizeY = layer->oshape.s1;
    }
    else
    {
        cdfilterlayer->globalWorkSizeY = ((layer->oshape.s1/lws)+1)*lws;
    }

    cdfilterlayer->gidXGap = layer->oshape.s0 - cdfilterlayer->globalWorkSizeX;
    cdfilterlayer->gidYGap = layer->oshape.s1 - cdfilterlayer->globalWorkSizeY;

    if(!net->config->silent&&net->config->debugInfo)
        g_print("LocalWorkSizexy: %d \n", (gint)cdfilterlayer->localWorkSizexy);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Scheduled Conv2D with unique weights Layer \n");


    if(!net->config->silent&&net->config->debugInfo)
        g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);

    return TRUE;
 }


gboolean dR_cdfilter_compute(dR_Graph* net, dR_Node* layer){
    dR_MDFilter_Data* cdfilterlayer = ((dR_MDFilter_Data*)(layer->layer));
    cl_int filterWindowWidth = cdfilterlayer->shape.s0;
    cl_int filterWindowHeight = cdfilterlayer->shape.s1;
    cl_int numberOfFilters = cdfilterlayer->shape.s2;
    cl_int lMemImageSize, lMemFilterSize;
    size_t localWorkSize[3];
    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input1;
    cl_mem* input2;
    dR_list_resetIt(layer->previous_layers);
    input1 = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    input2 = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;


    localWorkSize[0] = cdfilterlayer->localWorkSizexy;
    localWorkSize[1] = cdfilterlayer->localWorkSizexy;
    localWorkSize[2] = 1;
    lMemImageSize = (localWorkSize[0] + filterWindowWidth - 1) * (localWorkSize[1] + filterWindowHeight - 1);
    lMemFilterSize = filterWindowHeight*filterWindowWidth*numberOfFilters;
    globalWorkSize[0] = cdfilterlayer->globalWorkSizeX;
    globalWorkSize[1] = cdfilterlayer->globalWorkSizeY;
    globalWorkSize[2] = cdfilterlayer->iimgshape.s2;

    if ((filterWindowWidth-1)/2 > (cl_int)localWorkSize[0]
    || (filterWindowHeight-1) > (cl_int)localWorkSize[1])
    {
        g_print("Error: Kernel size for convolution is too big (is %ix%i; max %ix%i)\n", filterWindowWidth, filterWindowHeight, (gint)(localWorkSize[0]), (gint)(localWorkSize[1]));
        return FALSE;
    }

    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input1);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);   paramid++;
    if(cdfilterlayer->useLMEM)
    {
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemImageSize * sizeof(cl_float), NULL);         paramid++;
    }
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input2);                     paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)&cdfilterlayer->filterBuf);   paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemFilterSize * sizeof(cl_float), NULL);            paramid++;


    if (dR_openCLError(net, "Setting kernel args failed.", "Context-dependent filter Kernel"))
        return FALSE;
    //execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        localWorkSize, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:cdfilter");

}

gboolean dR_cdfilter_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_MDFilter_Data* cdfilterlayer = ((dR_MDFilter_Data*)(layer->layer));
    dR_Node* last_layer;

    dR_list_resetIt(layer->previous_layers);
    last_layer = dR_list_next(layer->previous_layers);

    cdfilterlayer->iimgshape.s0 = last_layer->oshape.s0;
    cdfilterlayer->iimgshape.s1 = last_layer->oshape.s1;
    cdfilterlayer->iimgshape.s2 = last_layer->oshape.s2;

    last_layer = dR_list_next(layer->previous_layers);
    cdfilterlayer->ifiltershape.s0 = last_layer->oshape.s0;
    cdfilterlayer->ifiltershape.s1 = last_layer->oshape.s1;
    cdfilterlayer->ifiltershape.s2 = last_layer->oshape.s2;


    if(cdfilterlayer->iimgshape.s0!= cdfilterlayer->ifiltershape.s0
            || cdfilterlayer->iimgshape.s1!= cdfilterlayer->ifiltershape.s1)
    {
        if(!net->config->silent)
            g_print("Error: Context-dependent Filter: Input shapes do not match: (%d x %d) vs (%d vs %d) \n",
                cdfilterlayer->iimgshape.s0,cdfilterlayer->iimgshape.s1,cdfilterlayer->ifiltershape.s0,cdfilterlayer->ifiltershape.s1);
        return FALSE;
    }


    layer->oshape.s0 = cdfilterlayer->iimgshape.s0;
    layer->oshape.s1 = cdfilterlayer->iimgshape.s1;
    layer->oshape.s2 = cdfilterlayer->iimgshape.s2;


    if(layer->oshape.s0<=0 || layer->oshape.s1<=0 || layer->oshape.s2<=0)
    {
        if(!net->config->silent)
            g_print("Conv2d shape propagates to Zero. False shape parameters! \n");
        return FALSE;
    }
    return TRUE;
}

gint32 dR_cdfilter_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_cdfilter_createKernel(dR_Graph* net, dR_Node* layer)
{
    //dR_MDFilter_Data* cdfilterlayer = (dR_MDFilter_Data*)(layer->layer);
    return dR_createKernel(net,layer->createKernelName(layer),&(layer->clKernel));
}


gboolean dR_cdfilter_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_Shape3 shape;
    shape = ((dR_MDFilter_Data*)(layer->layer))->shape;
    ret &= dR_createFloatBuffer(net, &(((dR_MDFilter_Data*)(layer->layer))->filterBuf),shape.s0*shape.s1*shape.s2, CL_MEM_READ_ONLY);
    return ret;
}

gboolean dR_cdfilter_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_MDFilter_Data* cdfilterlayer = ((dR_MDFilter_Data*)(layer->layer));
    ret &= dR_uploadArray(net,"",cdfilterlayer->filters,0,cdfilterlayer->shape.s0*cdfilterlayer->shape.s1*cdfilterlayer->shape.s2*sizeof(cl_float),cdfilterlayer->filterBuf);
    return ret;
}

gboolean dR_cdfilter_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    ret &= dR_clMemoryBufferCleanup(net, ((dR_MDFilter_Data*)(layer->layer))->filterBuf);
    ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_cdfilter_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
    {
        g_free(((dR_MDFilter_Data*)(layer->layer))->filters);
        g_free((dR_MDFilter_Data*)(layer->layer));
    }
    return TRUE;
}


// Optional

void  dR_MaskDependentFilter_setVariables(dR_Node* layer, gfloat* weights, gfloat* bias)
{
    dR_MDFilter_Data* cdfilterlayer = ((dR_MDFilter_Data*)(layer->layer));
    if(bias!=NULL)
        g_print("Warning: MaskDependentFilter has uses no bias variable but got values to set as bias!\n");

    cdfilterlayer->filters = weights;
    cdfilterlayer->hasVariables = 1;
}


gboolean dR_cdfilter_generateKernel(dR_Graph* net, dR_Node* layer)
{
    dR_MDFilter_Data* cdfilterlayer = (dR_MDFilter_Data*)(layer->layer);
    int length = 0;

    gchar* source, *temp;
    gchar* kernelname = dR_cdfilter_createKernelName(layer);
    gchar * filePath;

    // Variables that get hardcoded into the shader
    int width =cdfilterlayer->iimgshape.s0;
    int height =cdfilterlayer->iimgshape.s1;
    int inputDepth = cdfilterlayer->iimgshape.s2;
    int widthheight = width*height;
    int filterWidth = cdfilterlayer->shape.s0;
    int filterHeight = cdfilterlayer->shape.s1;
    int numFilters = cdfilterlayer->shape.s2;
    int x = -filterWidth/2;
    int y = -filterHeight/2;
    int filterxy = filterWidth*filterHeight;
    int floatsToReadInThisWG = (cdfilterlayer->localWorkSizexy+filterWidth-1)*(cdfilterlayer->localWorkSizexy+filterHeight-1);
    int numberOfWorkItems = cdfilterlayer->localWorkSizexy*cdfilterlayer->localWorkSizexy;
    int copyWidth = cdfilterlayer->localWorkSizexy+ filterWidth-1;
    int jumpint = cdfilterlayer->localWorkSizexy-1;
    int fid = 0;
    int outputindex,yid,xid;


    //Creating Shader Name and Return type

    source = g_strdup_printf("#ifndef kerneldef_%s \n#define kerneldef_%s \n\n__kernel void %s(\n",kernelname,kernelname,kernelname);


    // Rest of the Header and first part
    temp = g_strdup_printf(
    "    const __global  float * gInputImage,\n"
    "    __global  float * gOutputImage,\n");
    source = concat_and_free_old(source,temp);
    if(cdfilterlayer->useLMEM)
    {
        temp = g_strdup_printf(
        "    __local float* lImage,\n");
        source = concat_and_free_old(source,temp);
    }
    temp = g_strdup_printf(
    "    const __global float * gFilterClass,\n"
    "    const __global float * gFilter,\n"
    "    __local float * lFilter\n"
    "    )\n"
    "{\n"
    "    int biasindex = 0;\n"
    "    float sum = 0.0f;\n");
    source = concat_and_free_old(source,temp);


    if(cdfilterlayer->useLMEM)
    {
        temp = g_strdup_printf(
        "    int index;\n");
        source = concat_and_free_old(source,temp);
    }
    temp = g_strdup_printf(
    "    const int indexstart = get_local_id(1)*%d + get_local_id(0);\n"
    "    const int inImageOffset = %d*get_global_id(1)+get_global_id(0);\n"
    "    const int inFilterOffset = gFilterClass[inImageOffset];", (gint)cdfilterlayer->localWorkSizexy + filterWidth -1,width);
    source = concat_and_free_old(source,temp);

    temp = g_strdup_printf(
    "    const int blx = get_group_id(0)*%d-%d;\n",(gint)cdfilterlayer->localWorkSizexy,filterWidth/2);
    source = concat_and_free_old(source,temp);
    temp = g_strdup_printf(
    "    const int bly = get_group_id(1)*%d-%d;\n",(gint)cdfilterlayer->localWorkSizexy,filterHeight/2);
    source = concat_and_free_old(source,temp);

    // Inputindex >= 0

    temp = g_strdup_printf(
    "        if(get_global_id(0)<%d&&get_global_id(1)<%d)\n"
    "        {\n", width, height);
    source = concat_and_free_old(source,temp);

    // Copy Filter to LocalMem

    temp = g_strdup_printf(
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<%d;i+=%d)\n"
    "            lFilter[i] = gFilter[i];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n", numFilters*filterxy, (gint)cdfilterlayer->localWorkSizexy*(gint)cdfilterlayer->localWorkSizexy);
    source = concat_and_free_old(source,temp);

    // OutputLoop
    for(outputindex=0; outputindex<inputDepth;outputindex++)
    {
        // Copy ImagePart to LocalMem
        if(cdfilterlayer->useLMEM)
        {
            temp = g_strdup_printf(
            "        barrier(CLK_LOCAL_MEM_FENCE);\n"
            "        for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<%d;i+=%d)\n"
            "        {\n"
            "            lImage[i] = sampleZeroPaddedFloat(gInputImage, %d, blx + (i %% %d), bly + (i / %d), %d, %d);\n"
            "        }\n"
            "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                        ,floatsToReadInThisWG, numberOfWorkItems,outputindex*widthheight, copyWidth, copyWidth, width, height);
            source = concat_and_free_old(source,temp);
        }



        if(cdfilterlayer->useLMEM)
        {
            temp = g_strdup_printf(
            "            index = indexstart;\n");
            source = concat_and_free_old(source,temp);
        }
        x = -filterWidth/2;
        y = -filterHeight/2;
        fid = 0;
        //ConvolutionLoop
        for(yid= 0; yid<cdfilterlayer->shape.s1;yid++)
        {
            for(xid = 0; xid<cdfilterlayer->shape.s0;xid++)
            {
                if(cdfilterlayer->useLMEM)
                {
                    temp = g_strdup_printf(
                    "            sum"
                    "+= "
                    "lImage[index]"
                    "* lFilter[%d*inFilterOffset+%d]; index++;\n"
                                ,filterxy,fid);
                    source = concat_and_free_old(source,temp);
                    fid++;
                }
                else
                {
                    temp = g_strdup_printf(
                    "            sum"
                    "+= "
                    "sampleZeroPaddedFloat(gInputImage, %d,get_global_id(0)+%d,get_global_id(1)+%d,%d,%d)"
                                        ,outputindex*widthheight,x,y, width, height);
                    source = concat_and_free_old(source,temp);
                    temp = g_strdup_printf(
                    "* lFilter[%d*inFilterOffset+%d]; index++;\n"
                                ,filterxy,fid);
                    source = concat_and_free_old(source,temp);
                    fid++;x++;
                }
            }
            if(cdfilterlayer->useLMEM)
            {
                temp = g_strdup_printf(
                "            index+=%d;\n",jumpint);
                source = concat_and_free_old(source,temp);
            }
            y++;
        }
        // Output

        temp = g_strdup_printf(
        "            gOutputImage[biasindex*%d+inImageOffset] = sum; \n"
        "            sum = 0.0f; biasindex++;\n"
                    , widthheight);
        source = concat_and_free_old(source,temp);

    }
    temp = g_strdup_printf(
    "        }\n"
    "}\n"
    "#endif \n");
    source = concat_and_free_old(source,temp);

    length = strlen(source);

    filePath = g_build_filename(net->config->modelPath, dR_cdfilter_createKernelName(layer), NULL);
    g_file_set_contents(filePath,source,length,NULL);

    g_free(source);

    return TRUE;
}

gchar* dR_cdfilter_createKernelName(dR_Node* layer)
{
    dR_MDFilter_Data* cdfilterlayer = (dR_MDFilter_Data*)(layer->layer);
    gchar* string, *lmem;
    if(!cdfilterlayer->useLMEM)
        lmem = g_strdup_printf("nolmem");
    else
        lmem = g_strdup_printf("lmem");

    string = g_strdup_printf("cdfilter%dx%dx%d%s", cdfilterlayer->shape.s0, cdfilterlayer->shape.s1,cdfilterlayer->shape.s2,lmem);

    return string;
}


gchar* dR_cdfilter_printLayer(dR_Node* layer)
{
    dR_MDFilter_Data* convlayer = (dR_MDFilter_Data*)(layer->layer);
    gchar* out;

    out = g_strdup_printf("%s%d%s%d%s%d%s%d%s","Context-dependent Filter Layer: ", layer->layerID,"\n Filter dimensions: ", convlayer->shape.s0,"x",convlayer->shape.s1,
            "\n Number of Filters: ",convlayer->shape.s2,"\n");

    return out;
}





// ////////////////////////////////
// Individual Filter Application //
// ////////////////////////////////


// Mandatory

dR_Node* dR_PerPixelFilter(dR_Graph* net, dR_Node* inputLayer, dR_Node* inputFilter, dR_Shape4* sh, dR_Shape4* st){
    dR_PPFilter_Data* conv2d = g_malloc(sizeof(dR_PPFilter_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    conv2d->shape.s0 = sh->s0;
    conv2d->shape.s1 = sh->s1;
    conv2d->shape.s2 = sh->s2;
    conv2d->shape.s3 = sh->s3;
    conv2d->stride.s0 = st->s0;
    conv2d->stride.s1 = st->s1;
    conv2d->stride.s2 = st->s2;
    conv2d->stride.s3 = st->s3;
    l->layer = conv2d;
    l->type = tPPFilter;

    l->compute = dR_conv2duw_compute;
    l->schedule = dR_conv2duw_schedule;
    l->propagateShape = dR_conv2duw_propagateShape;
    l->getRequiredOutputBufferSize = dR_conv2duw_getRequiredOutputBufferSize;
    l->createKernel = dR_conv2duw_createKernel;
    l->allocateBuffers = dR_conv2duw_allocateBuffers;
    l->fillBuffers = dR_conv2duw_fillBuffers;
    l->cleanupBuffers = dR_conv2duw_cleanupBuffers;
    l->cleanupLayer = dR_conv2duw_cleanupLayer;
    l->serializeNode = dR_conv2duw_serializeNode;
    l->parseAppendNode = dR_conv2duw_parseAppendNode;

    l->generateKernel = dR_conv2duw_generateKernel;
    l->setVariables = NULL;
    l->createKernelName = dR_conv2duw_createKernelName;
    l->printLayer = dR_conv2duw_printLayer;

    if(inputLayer&&inputFilter)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        dR_list_append(l->previous_layers,inputFilter);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
        dR_list_append(inputFilter->next_layers,l);
    }
    else
    {
        g_print("Error: Conv2d with unique weights needs 2 appropriate Inputnodes");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_conv2duw_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_PPFilter_Data* ppflayer = (dR_PPFilter_Data*)(layer->layer);
    gchar* desc = "PPFilter";
    gint numNodeParams = 8;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",ppflayer->shape.s0);
    params[1] = g_strdup_printf("%d",ppflayer->shape.s1);
    params[2] = g_strdup_printf("%d",ppflayer->shape.s2);
    params[3] = g_strdup_printf("%d",ppflayer->shape.s3);
    params[4] = g_strdup_printf("%d",ppflayer->stride.s0);
    params[5] = g_strdup_printf("%d",ppflayer->stride.s1);
    params[6] = g_strdup_printf("%d",ppflayer->stride.s2);
    params[7] = g_strdup_printf("%d",ppflayer->stride.s3);
    *numVariables = numNodeVariables;

    return desc;
}

dR_Node* dR_conv2duw_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 2;
    gint numNodeParams = 8;
    gint numNodeVariables = 0;
    dR_Shape4* shape;
    dR_Shape4* stride;
    dR_Node* out;
    shape = g_malloc(sizeof(dR_Shape4));
    stride = g_malloc(sizeof(dR_Shape4));
    if(numINodes!=1)
    {
        g_print("Parsing Error: PPFilter Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: PPFilter Node needs %d Parameters and %d!\n",numNodeParams,numNodeVariables);
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
    out = dR_PerPixelFilter(net, iNodes[0], iNodes[1], shape, stride);
    g_free(shape);
    g_free(stride);
    return out;
}

gboolean dR_conv2duw_schedule(dR_Graph* net, dR_Node* layer){

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

    dR_PPFilter_Data* convlayer = (dR_PPFilter_Data*)(layer->layer);
    gint lws = 16;
    cl_ulong lms;
	size_t mws;
    gint localMemoryConFilter = (convlayer->shape.s0*convlayer->shape.s1*convlayer->shape.s3*4);
    gint localMemoryConInput = ((lws+(convlayer->shape.s0-1))*(lws+(convlayer->shape.s0-1))*4);
    int z=1;

    if(convlayer->shape.s0<=convlayer->stride.s1&&convlayer->shape.s1<=convlayer->stride.s2)
        convlayer->useLMEM = FALSE;
    else
        convlayer->useLMEM = TRUE;



    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lms, NULL);
    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &mws, NULL);

    if(!net->config->silent)
        g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);
    while(localMemoryConInput+localMemoryConFilter<(gint)lms&&lws*lws<=(gint)mws){
        lws*=2;
        localMemoryConInput = ((lws+(convlayer->shape.s0-1))*(lws+(convlayer->shape.s0-1))*4);
    }
    lws/=2;
    convlayer->localWorkSizexy = lws;
    convlayer->numberOfDepthPartitions = z;

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

    if(!net->config->silent&&net->config->debugInfo)
        g_print("LocalWorkSizexy: %d \n", (gint)convlayer->localWorkSizexy);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Scheduled Conv2D with unique weights Layer \n");


    if(!net->config->silent&&net->config->debugInfo)
        g_print("Local Memory Size: %d \nMax Workgroup Size: %d \n", (gint)lms, (gint)mws);

    return TRUE;
 }


gboolean dR_conv2duw_compute(dR_Graph* net, dR_Node* layer){
    dR_PPFilter_Data* convlayer = ((dR_PPFilter_Data*)(layer->layer));
    cl_int filterWindowWidth = convlayer->shape.s0;
    cl_int filterWindowHeight = convlayer->shape.s1;
    //cl_int oDepth = convlayer->shape.s3;
    //cl_int iwidth = convlayer->iimgshape.s0;
    //cl_int iheight = convlayer->iimgshape.s1;
    //cl_int idepth = convlayer->iimgshape.s2;
    cl_int lMemImageSize;
    size_t localWorkSize[3];
    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input1;
    cl_mem* input2;
    dR_list_resetIt(layer->previous_layers);
    input1 = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    input2 = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;


    localWorkSize[0] = convlayer->localWorkSizexy;
    localWorkSize[1] = convlayer->localWorkSizexy;
    localWorkSize[2] = 1;
    lMemImageSize = (localWorkSize[0] + filterWindowWidth - 1) * (localWorkSize[1] + filterWindowHeight - 1);
    globalWorkSize[0] = convlayer->globalWorkSizeX;
    globalWorkSize[1] = convlayer->globalWorkSizeY;
    globalWorkSize[2] = convlayer->numberOfDepthPartitions;

    if ((filterWindowWidth-1)/2 > (cl_int)localWorkSize[0]
    || (filterWindowHeight-1) > (cl_int)localWorkSize[1])
    {
        g_print("Error: Kernel size for convolution is too big (is %ix%i; max %ix%i)\n", filterWindowWidth, filterWindowHeight, (gint)(localWorkSize[0]), (gint)(localWorkSize[1]));
        return FALSE;
    }

    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input1);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);          paramid++;
    if(convlayer->useLMEM)
    {
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemImageSize * sizeof(cl_float), NULL);         paramid++;
    }
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input2);     paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&iwidth);                    paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&iheight);                   paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&idepth);                    paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&filterWindowWidth);         paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&filterWindowHeight);        paramid++;
    //net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&oDepth);                    paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Conv2d unique weights Kernel"))
        return FALSE;
    //execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        localWorkSize, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:conv2duw");

}

gboolean dR_conv2duw_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_PPFilter_Data* convlayer = ((dR_PPFilter_Data*)(layer->layer));
    dR_Node* last_layer;

    dR_list_resetIt(layer->previous_layers);
    last_layer = dR_list_next(layer->previous_layers);

    convlayer->iimgshape.s0 = last_layer->oshape.s0;
    convlayer->iimgshape.s1 = last_layer->oshape.s1;
    convlayer->iimgshape.s2 = last_layer->oshape.s2;

    last_layer = dR_list_next(layer->previous_layers);
    convlayer->ifiltershape.s0 = last_layer->oshape.s0;
    convlayer->ifiltershape.s1 = last_layer->oshape.s1;
    convlayer->ifiltershape.s2 = last_layer->oshape.s2;


    if(convlayer->iimgshape.s0!= convlayer->ifiltershape.s0
            || convlayer->iimgshape.s1!= convlayer->ifiltershape.s1)
    {
        if(!net->config->silent)
            g_print("Error: Conv2d with unique weights: Input shapes do not match: (%d x %d) vs (%d vs %d) \n",
                convlayer->iimgshape.s0,convlayer->iimgshape.s1,convlayer->ifiltershape.s0,convlayer->ifiltershape.s1);
        return FALSE;
    }


    if(convlayer->ifiltershape.s2%(convlayer->iimgshape.s2*convlayer->shape.s0*convlayer->shape.s1)!=0)
    {
        if(!net->config->silent)
            g_print("Error: Conv2d with unique weights: Depth of Filterinput not dividable by depth of Imageinput times filter size: %d vs %d*%d*%d \n",
                convlayer->ifiltershape.s2,convlayer->iimgshape.s2,convlayer->shape.s1,convlayer->shape.s2);
        return FALSE;
    }


    if(convlayer->ifiltershape.s2/(convlayer->shape.s0*convlayer->shape.s1)!=convlayer->shape.s3)
    {
        if(!net->config->silent)
            g_print("Error: Conv2d with unique weights: Depth of Filterinput divided by depth of Imageinput times filter size does not result in shape.s3: %d vs %d \n",
                convlayer->ifiltershape.s2/(convlayer->iimgshape.s2*convlayer->shape.s1*convlayer->shape.s2),convlayer->shape.s3);
        return FALSE;
    }


    layer->oshape.s0 = convlayer->iimgshape.s0/convlayer->stride.s1;
    layer->oshape.s1 = convlayer->iimgshape.s1/convlayer->stride.s2;
    layer->oshape.s2 = convlayer->ifiltershape.s2/(convlayer->shape.s0*convlayer->shape.s1);


    if(layer->oshape.s0<=0 || layer->oshape.s1<=0 || layer->oshape.s2<=0)
    {
        g_print("Conv2d shape propagates to Zero. False shape parameters! \n");
        return FALSE;
    }
    return TRUE;
}

gint32 dR_conv2duw_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_conv2duw_createKernel(dR_Graph* net, dR_Node* layer)
{
    //dR_PPFilter_Data* convlayer = (dR_PPFilter_Data*)(layer->layer);
    return dR_createKernel(net,dR_conv2duw_createKernelName(layer),&(layer->clKernel));
}


gboolean dR_conv2duw_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_conv2duw_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_conv2duw_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    if(net->prepared)
        ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_conv2duw_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
        g_free((dR_PPFilter_Data*)(layer->layer));
    return TRUE;
}


// Optional


gboolean dR_conv2duw_generateKernel(dR_Graph* net, dR_Node* layer)
{
    dR_PPFilter_Data* convlayer = (dR_PPFilter_Data*)(layer->layer);
    int length = 0;

    gchar* kernelname = dR_conv2duw_createKernelName(layer);
    gchar* source, *temp;
    gchar * filePath;

    // Variables that get hardcoded into the shader
    int width =convlayer->iimgshape.s0;
    int height =convlayer->iimgshape.s1;
    int inputDepth =convlayer->iimgshape.s2;
    int widthheight = width*height;
    int filterWidth = convlayer->shape.s0;
    int filterHeight = convlayer->shape.s1;
    int outputDepth = convlayer->shape.s3;
    int outputChannels = outputDepth/inputDepth;
    int x = -filterWidth/2;
    int y = -filterHeight/2;
    int lengthindepth = outputDepth/convlayer->numberOfDepthPartitions;
    int filterxy = filterWidth*filterHeight;
    int floatsToReadInThisWG = (convlayer->localWorkSizexy+filterWidth-1)*(convlayer->localWorkSizexy+filterHeight-1);
    int numberOfWorkItems = convlayer->localWorkSizexy*convlayer->localWorkSizexy;
    int copyWidth = convlayer->localWorkSizexy+ filterWidth-1;
    int jumpint = convlayer->localWorkSizexy-1;
    int fid = 0;
    int outputindex,yid,xid,inputindex;


    //Creating Shader Name and Return type

    source = g_strdup_printf(
    "#ifndef kerneldef_%s \n#define kerneldef_%s \n\n__kernel void %s(\n",kernelname,kernelname,kernelname);


    // Rest of the Header and first part

    temp = g_strdup_printf(
    "    const __global  float * gInputImage,\n"
    "    __global  float * gOutputImage,\n");
    source = concat_and_free_old(source,temp);
    if(convlayer->useLMEM)
    {
        temp = g_strdup_printf(
        "    __local float* lImage,\n");
        source = concat_and_free_old(source,temp);
    }
    temp = g_strdup_printf(
    "    const __global float * gFilter\n"
    "    )\n"
    "{\n"
    "    const int depthstart = %d*get_global_id(2);\n"
    "    int biasindex = 0;\n"
    "    float sum = 0.0f;\n",lengthindepth);
    source = concat_and_free_old(source,temp);


    if(convlayer->useLMEM)
    {
        temp = g_strdup_printf(
        "    int index;\n");
        source = concat_and_free_old(source,temp);
    }

    temp = g_strdup_printf(
    "    const int indexstart = get_local_id(1)*%d + get_local_id(0);\n"
    "    const int inImageOffset = %d*get_global_id(1)+get_global_id(0);\n", (gint)convlayer->localWorkSizexy + filterWidth -1,width);
    source = concat_and_free_old(source,temp);

    temp = g_strdup_printf(
    "    const int blx = get_group_id(0)*%d-%d;\n",(gint)convlayer->localWorkSizexy,filterWidth/2);
    source = concat_and_free_old(source,temp);
    temp = g_strdup_printf(
    "    const int bly = get_group_id(1)*%d-%d;\n",(gint)convlayer->localWorkSizexy,filterHeight/2);
    source = concat_and_free_old(source,temp);

    // Inputindex >= 0
    // Inputloop

    for(inputindex = 0; inputindex<inputDepth; inputindex++)
    {

        temp = g_strdup_printf(
        "        barrier(CLK_LOCAL_MEM_FENCE);\n");
        source = concat_and_free_old(source,temp);

        // Copy ImagePart to LocalMem
        if(convlayer->useLMEM)
        {
            temp = g_strdup_printf(
        "        for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<%d;i+=%d)\n"
        "        {\n"
        "            lImage[i] = sampleZeroPaddedFloat(gInputImage, %d, blx + (i %% %d), bly + (i / %d), %d, %d);\n"
        "        }\n",floatsToReadInThisWG, numberOfWorkItems,inputindex*widthheight, copyWidth, copyWidth, width, height);
            source = concat_and_free_old(source,temp);
        }
        // Copy Filter to LocalMem
        temp = g_strdup_printf(
        "        barrier(CLK_LOCAL_MEM_FENCE);\n"
        "        if(get_global_id(0)<%d&&get_global_id(1)<%d)\n"
        "        {\n", width, height);
        source = concat_and_free_old(source,temp);

        // OutputLoop
        for(outputindex=0; outputindex<outputChannels;outputindex++)
        {
            if(convlayer->useLMEM)
            {
                temp = g_strdup_printf(
                "            index = indexstart;\n");
                source = concat_and_free_old(source,temp);
            }
            x = -filterWidth/2;
            y = -filterHeight/2;
            fid = 0;
            //ConvolutionLoop
            for(yid= 0; yid<convlayer->shape.s1;yid++)
            {
                for(xid = 0; xid<convlayer->shape.s0;xid++)
                {
                    if(convlayer->useLMEM)
                    {
                        temp = g_strdup_printf(
                        "            sum"
                        "+= "
                        "lImage[index]"
                        "* gFilter[%d+inImageOffset]; index++;\n"
                        ,(inputindex*outputChannels*filterxy+outputindex*filterxy+fid)*widthheight);
                        source = concat_and_free_old(source,temp);

                        fid++;
                    }
                    else
                    {
                        temp = g_strdup_printf(
                        "            sum"
                        "+= "
                        "sampleZeroPaddedFloat(gInputImage, (inputindex*%d),get_global_id(0)+%d,get_global_id(1)+%d,%d,%d)"
                                            ,widthheight,x,y, width, height);
                        source = concat_and_free_old(source,temp);
                        temp = g_strdup_printf(
                        "* gFilter[%d+inImageOffset]; index++;\n"
                                    ,(inputindex*outputChannels*filterxy+outputindex*filterxy+fid)*widthheight);
                        source = concat_and_free_old(source,temp);
                        fid++;x++;
                    }
                }
                if(convlayer->useLMEM)
                {
                    temp = g_strdup_printf(
                    "            index+=%d;\n",jumpint);
                    source = concat_and_free_old(source,temp);
                }
                y++;
            }
            // Output

            temp = g_strdup_printf(
            "            gOutputImage[biasindex*%d+inImageOffset] = sum; \n"
            "            sum = 0.0f; biasindex++;\n", widthheight);
            source = concat_and_free_old(source,temp);

        }
        temp = g_strdup_printf(
        "        }\n");
        source = concat_and_free_old(source,temp);
    }

    temp = g_strdup_printf(
    "}\n"
    "#endif \n");
    source = concat_and_free_old(source,temp);

    length = strlen(source);

    filePath = g_build_filename(net->config->modelPath, dR_conv2duw_createKernelName(layer), NULL);
    g_file_set_contents(filePath,source,length,NULL);

    g_free(source);

    return TRUE;
}

gchar* dR_conv2duw_createKernelName(dR_Node* layer)
{
    dR_PPFilter_Data* convlayer = (dR_PPFilter_Data*)(layer->layer);
    gchar* string, *lmem;
    if(!convlayer->useLMEM)
        lmem = g_strdup_printf("nolmem");
    else
        lmem = g_strdup_printf("lmem");
    string = g_strdup_printf("conv2duw%dx%dx%d_%d_%s", convlayer->shape.s0, convlayer->shape.s1,convlayer->shape.s3,layer->layerID,lmem);
    g_free(lmem);
    return string;
}


gchar* dR_conv2duw_printLayer(dR_Node* layer)
{
    dR_PPFilter_Data* convlayer = (dR_PPFilter_Data*)(layer->layer);
    gchar* out;
    out = g_strdup_printf("%s%d%s%d%s%d%s%d%s%d%s%d%s%d%s%d%s%d%s","Conv2d unique weights Layer: ", layer->layerID,"\n Filter dimensions: ", convlayer->shape.s0,"x",convlayer->shape.s1,
            "\n Input channels: ",convlayer->shape.s2,"\n Output channels: ", convlayer->shape.s3,
            "\n Stride: ", convlayer->stride.s0, ", ", convlayer->stride.s1, ", ", convlayer->stride.s2,", ",convlayer->stride.s3,"\n");

    return out;
}
