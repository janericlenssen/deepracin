#include "dR_nodes_transform.h"
#include "dR_core.h"

// //////////////////////
// Extract Slice Layer //
// //////////////////////

dR_Node* dR_Slice(dR_Graph* net, dR_Node* inputLayer, dR_Shape4 origin, dR_Shape4 shape){
    dR_Slice_Data* slicenode = g_malloc(sizeof(dR_Slice_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    slicenode->origin=origin;
    slicenode->oshape=shape;
    l->layer = slicenode;
    l->type = tSlice;

    l->compute = dR_slice_compute;
    l->schedule = dR_slice_schedule;
    l->propagateShape = dR_slice_propagateShape;
    l->getRequiredOutputBufferSize = dR_slice_getRequiredOutputBufferSize;
    l->createKernel = dR_slice_createKernel;
    l->allocateBuffers = dR_slice_allocateBuffers;
    l->fillBuffers = dR_slice_fillBuffers;
    l->cleanupBuffers = dR_slice_cleanupBuffers;
    l->cleanupLayer = dR_slice_cleanupLayer;
    l->serializeNode = dR_slice_serializeNode;
    l->parseAppendNode = dR_slice_parseAppendNode;

    l->generateKernel = NULL;
    l->setVariables = NULL;
    l->createKernelName = NULL;
    l->printLayer = dR_slice_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: Slice Node needs an appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_slice_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_Slice_Data* slicenode = (dR_Slice_Data*)(layer->layer);
    gchar* desc = "Slice";
    gint numNodeParams = 8;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",slicenode->origin.s0);
    params[1] = g_strdup_printf("%d",slicenode->origin.s1);
    params[2] = g_strdup_printf("%d",slicenode->origin.s2);
    params[3] = g_strdup_printf("%d",slicenode->origin.s3);
    params[4] = g_strdup_printf("%d",slicenode->oshape.s0);
    params[5] = g_strdup_printf("%d",slicenode->oshape.s1);
    params[6] = g_strdup_printf("%d",slicenode->oshape.s2);
    params[7] = g_strdup_printf("%d",slicenode->oshape.s3);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_slice_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 8;
    gint numNodeVariables = 0;
    dR_Node* out;
    dR_Shape4 shape;
    dR_Shape4 origin;
    if(numINodes!=1)
    {
        g_print("Parsing Error: Slice Node needs %d InputNodes but got %d!\n",numNodeInputs,numINodes);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: Slice Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    origin.s0 = atoi(params[0]);
    origin.s1 = atoi(params[1]);
    origin.s2 = atoi(params[2]);
    origin.s3 = atoi(params[3]);
    shape.s0 = atoi(params[4]);
    shape.s1 = atoi(params[5]);
    shape.s2 = atoi(params[6]);
    shape.s3 = atoi(params[7]);
    out = dR_Slice(net, iNodes[0], origin, shape);
    return out;
}

gboolean dR_slice_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }


gboolean dR_slice_compute(dR_Graph* net, dR_Node* layer){
    dR_Slice_Data* slicenode = ((dR_Slice_Data*)(layer->layer));
    size_t globalWorkSize[1];
    int paramid = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    globalWorkSize[0] = layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;


    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&slicenode->origin.s0);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&slicenode->origin.s1);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&slicenode->origin.s2);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&slicenode->oshape.s0);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&slicenode->oshape.s1);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&slicenode->oshape.s2);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&slicenode->ishape.s0);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&slicenode->ishape.s1);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&slicenode->ishape.s2);          paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Extract Slice Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 1, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:extractSlice");

}

gboolean dR_slice_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_Slice_Data* slicenode = ((dR_Slice_Data*)(layer->layer));
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("Extract Slice Node with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    slicenode->ishape.s0 = lastlayer->oshape.s0;
    slicenode->ishape.s1 = lastlayer->oshape.s1;
    slicenode->ishape.s2 = lastlayer->oshape.s2;

    layer->oshape.s0 = slicenode->oshape.s0;
    layer->oshape.s1 = slicenode->oshape.s1;
    layer->oshape.s2 = slicenode->oshape.s2;
    return TRUE;
}

gint32 dR_slice_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_slice_createKernel(dR_Graph* net, dR_Node* layer)
{
    return dR_createKernel(net,"extractSlice",&(layer->clKernel));
}


gboolean dR_slice_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_slice_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_slice_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    if(net->prepared)
        ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_slice_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
        g_free((dR_Slice_Data*)(layer->layer));
    return TRUE;
}


gchar* dR_slice_printLayer(dR_Node* layer)
{
    dR_Slice_Data* slicenode = ((dR_Slice_Data*)(layer->layer));
    gchar* out;
    out = g_strdup_printf("%s%d%s%d%s%d%s%d%s%d%s%d%s%d%s%d%s%d%s",
            "Extract Slice Node: ",layer->layerID,
            "\n Origin: [", slicenode->origin.s0,", ",slicenode->origin.s1,", ",slicenode->origin.s2,", ",slicenode->origin.s3,
            "]\n Size: [",slicenode->oshape.s0,", ",slicenode->oshape.s1,", ",slicenode->oshape.s2,", ",slicenode->oshape.s3, "]\n");

    return out;
}


// ///////////////
// Concat Layer //
// ///////////////

dR_Node* dR_Concat(dR_Graph* net, dR_Node** inputNodes, gint n, gint concatDim){
    dR_Concat_Data* concatnode = g_malloc(sizeof(dR_Concat_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    concatnode->n = n;
    concatnode->concatDim = concatDim;
    l->layer = concatnode;
    l->type = tConcat;

    l->compute = dR_concat_compute;
    l->schedule = dR_concat_schedule;
    l->propagateShape = dR_concat_propagateShape;
    l->getRequiredOutputBufferSize = dR_concat_getRequiredOutputBufferSize;
    l->createKernel = dR_concat_createKernel;
    l->allocateBuffers = dR_concat_allocateBuffers;
    l->fillBuffers = dR_concat_fillBuffers;
    l->cleanupBuffers = dR_concat_cleanupBuffers;
    l->cleanupLayer = dR_concat_cleanupLayer;
    l->serializeNode = dR_concat_serializeNode;
    l->parseAppendNode = dR_concat_parseAppendNode;

    l->generateKernel = dR_concat_generateKernel;
    l->setVariables = NULL;
    l->createKernelName = dR_concat_createKernelName;
    l->printLayer = dR_concat_printLayer;

    if(inputNodes!=NULL&&inputNodes[n-1]!=NULL&&n>=0&&n<4)
    {
        gint i;
        l->next_layers = dR_list_createEmptyList();
        l->previous_layers = dR_list_createEmptyList();
        for (i = 0;i<n;i++)
        {
            dR_list_append(l->previous_layers,inputNodes[i]);
            dR_list_append(inputNodes[i]->next_layers,l);
        }
    }
    else
    {
        g_print("Error: Concat node layer needs an appropriate number (n) of Inputnodes and n has to be 0<=n<4 \n");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_concat_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_Concat_Data* concatnode = (dR_Concat_Data*)(layer->layer);
    gchar* desc = "Concat";
    gint numNodeParams = 2;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",concatnode->n);
    params[1] = g_strdup_printf("%d",concatnode->concatDim);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_concat_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeParams = 2;
    gint numNodeVariables = 0;
    dR_Node* out;
    gint n;

    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: Concat Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    n = atoi(params[0]);
    if(numINodes!=n)
    {
        g_print("Parsing Error: Concat Node needs %d InputNodes but got %d!\n",n,numNodeVariables);
        return NULL;
    }
    out = dR_Concat(net, iNodes, n, atoi(params[1]));
    return out;
}

gboolean dR_concat_schedule(dR_Graph* net, dR_Node* layer){
    dR_Concat_Data* concatnode = ((dR_Concat_Data*)(layer->layer));
    dR_Node* lastlayer;
    gint i;
    concatnode->maxInputSizeInConcatDim = 0;
    concatnode->inputSizesInConcatDim = (gint*)g_malloc(concatnode->n*sizeof(gint));
    dR_list_resetIt(layer->previous_layers);
    for(i=0; i<concatnode->n;i++)
    {
        lastlayer = dR_list_next(layer->previous_layers);
        concatnode->inputSizesInConcatDim[i] = ((gint*)&lastlayer->oshape)[concatnode->concatDim];
        if(concatnode->inputSizesInConcatDim[i]>concatnode->maxInputSizeInConcatDim)
        {
            concatnode->maxInputSizeInConcatDim = concatnode->inputSizesInConcatDim[i];
        }
    }
    if(!net->config->silent&&net->config->debugInfo)
        g_print("Concat maxInputSizeInConcatDim: %d \n",concatnode->maxInputSizeInConcatDim);
    return TRUE;
 }


gboolean dR_concat_compute(dR_Graph* net, dR_Node* layer){
    dR_Concat_Data* concatnode = ((dR_Concat_Data*)(layer->layer));
    size_t globalWorkSize[3];
    gint paramid = 0;
    cl_mem* input;
    gint i = 0;
    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;

    globalWorkSize[concatnode->concatDim] = concatnode->maxInputSizeInConcatDim;

    // Give all input node buffers
    dR_list_resetIt(layer->previous_layers);
    for(i = 0; i<concatnode->n;i++)
    {
        input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
        net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                      paramid++;
    }
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);          paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Concat Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:concat");

}

gboolean dR_concat_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_Concat_Data* concatnode = ((dR_Concat_Data*)(layer->layer));
    dR_Node* lastlayer;
    int i = 0;
    gint* oshape;
    if(layer->previous_layers->length!=concatnode->n)
    {
        if(!net->config->silent)
            g_print("Concat Layer Error: Parameter n does not match number of given inputnodes!\n");
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    layer->oshape.s0 = lastlayer->oshape.s0;
    layer->oshape.s1 = lastlayer->oshape.s1;
    layer->oshape.s2 = lastlayer->oshape.s2;
    for(i=1;i<concatnode->n;i++)
    {
        lastlayer = dR_list_next(layer->previous_layers);
        oshape = (gint*)&lastlayer->oshape;

        ((gint*)&layer->oshape)[concatnode->concatDim] += oshape[concatnode->concatDim];
    }
    return TRUE;
}

gint32 dR_concat_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_concat_createKernel(dR_Graph* net, dR_Node* layer)
{
    return dR_createKernel(net,dR_concat_createKernelName(layer),&(layer->clKernel));
}


gboolean dR_concat_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_concat_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_concat_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    if(net->prepared)
    {
        ret &= dR_cleanupKernel((layer->clKernel));
    }
    return ret;
}

gboolean dR_concat_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    dR_Concat_Data* concatnode = ((dR_Concat_Data*)(layer->layer));
    if(net->prepared)
    {
        g_free(concatnode->inputSizesInConcatDim);
        g_free((dR_Concat_Data*)(layer->layer));
    }
    return TRUE;
}


gchar* dR_concat_printLayer(dR_Node* layer)
{
    dR_Concat_Data* concatnode = ((dR_Concat_Data*)(layer->layer));
    gchar* out;
    out = g_strdup_printf("%s%d%s%d%s%d%s",
            "Concat Layer: ",layer->layerID,
            "\n Concat Dimension: ", concatnode->concatDim,
            "\n Number of concatenated slices: ", concatnode->n,"\n");

    return out;
}



gboolean dR_concat_generateKernel(dR_Graph* net, dR_Node* layer)
{
    dR_Concat_Data* concatnode = ((dR_Concat_Data*)(layer->layer));
    gint length = 0;

    gchar* kernelname;
    gchar *temp, *source;
    gchar * filePath;

    // Variables that get hardcoded into the shader
    gint n =concatnode->n;
    gint concatDim =concatnode->concatDim;
    gint owidth =layer->oshape.s0;
    gint oheight =layer->oshape.s1;
    gint i = 0;
    gint acum = 0;
    kernelname = dR_concat_createKernelName(layer);
    source = NULL;
    //Creating Shader Name and Return type
    source = g_strdup_printf(
    "#ifndef kerneldef_%s \n"
    "#define kerneldef_%s \n\n"
    "__kernel void %s(\n"
                ,kernelname,kernelname,kernelname);

    // Rest of the Header and first part
    for(i = 0; i<n; i++)
    {
        temp = g_strdup_printf(
    "    const __global float *gInput%d,\n",i);
        source = concat_and_free_old(source,temp);
    }

    temp = g_strdup_printf(
    "    __global  float * gOutputImage\n"
    ")\n"
    "{\n");
    source = concat_and_free_old(source,temp);
    for(i=0;i<n;i++)
    {
        temp = g_strdup_printf(
        "    if(get_global_id(%d)<%d);\n"
        "    {\n",concatDim,concatnode->inputSizesInConcatDim[i]);
        source = concat_and_free_old(source,temp);
        if(concatnode->concatDim==0)
        {
            temp = g_strdup_printf(
            "       int gid = get_global_id(2)*%d+get_global_id(1)*%d+(get_global_id(0)+%d); \n"
                        ,oheight*owidth,owidth,acum);
        }
        else if(concatnode->concatDim==1)
        {
            temp = g_strdup_printf(
            "       int gid = get_global_id(2)*%d+(get_global_id(1)+%d)*%d+get_global_id(0); \n"
                        ,oheight*owidth,acum,owidth);
        }
        else if(concatnode->concatDim==2)
        {
            temp = g_strdup_printf(
            "       int gid = (get_global_id(2)+%d)*%d+get_global_id(1)*%d+get_global_id(0); \n"
                        ,acum,oheight*owidth,owidth);
        }

        source = concat_and_free_old(source,temp);
        temp = g_strdup_printf(
        "       int igid = get_global_id(2)*%d+get_global_id(1)*%d+get_global_id(0); \n"
        "       gOutputImage[gid]=gInput%d[igid];\n"
        "    }\n"
                    ,oheight*owidth,owidth,i);
        source = concat_and_free_old(source,temp);
        acum +=concatnode->inputSizesInConcatDim[i];
    }

    temp = g_strdup_printf(
    "}\n"
    "#endif \n");
    source = concat_and_free_old(source,temp);

    length = strlen(source);
    filePath = g_build_filename(net->config->modelPath, dR_concat_createKernelName(layer), NULL);
    g_file_set_contents(filePath,source,length,NULL);

    g_free(source);

    return TRUE;
}

gchar* dR_concat_createKernelName(dR_Node* layer)
{
    dR_Concat_Data* concatnode = ((dR_Concat_Data*)(layer->layer));
    gint length = 0;
    gint maxsize = 100;
    gint i;
    gchar* string = g_malloc(sizeof(char)*maxsize);
    length += g_snprintf(string+length,maxsize-length,"concat_%din%d_%d", concatnode->n, concatnode->concatDim,layer->layerID);
    for(i = 0; i<concatnode->n;i++)
    {
        length += g_snprintf(string+length,maxsize-length,"_%d",concatnode->inputSizesInConcatDim[i]);
    }

    return string;
}


// ///////////////////
// Crop or Pad Node //
// ///////////////////

dR_Node* dR_CropOrPad(dR_Graph* net, dR_Node* inputLayer, dR_Shape3 shape){
    dR_CropOrPad_Data* croporpad = g_malloc(sizeof(dR_CropOrPad_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    croporpad->oshape=shape;
    l->layer = croporpad;
    l->type = tCropOrPad;

    l->compute = dR_croporpad_compute;
    l->schedule = dR_croporpad_schedule;
    l->propagateShape = dR_croporpad_propagateShape;
    l->getRequiredOutputBufferSize = dR_croporpad_getRequiredOutputBufferSize;
    l->createKernel = dR_croporpad_createKernel;
    l->allocateBuffers = dR_croporpad_allocateBuffers;
    l->fillBuffers = dR_croporpad_fillBuffers;
    l->cleanupBuffers = dR_croporpad_cleanupBuffers;
    l->cleanupLayer = dR_croporpad_cleanupLayer;
    l->serializeNode = dR_croporpad_serializeNode;
    l->parseAppendNode = dR_croporpad_parseAppendNode;

    l->generateKernel = NULL;
    l->setVariables = NULL;
    l->createKernelName = NULL;
    l->printLayer = dR_croporpad_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: Crop or Pad Node needs an appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_croporpad_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_CropOrPad_Data* croporpad = (dR_CropOrPad_Data*)(layer->layer);
    gchar* desc = "CropOrPad";
    gint numNodeParams = 3;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",croporpad->oshape.s0);
    params[1] = g_strdup_printf("%d",croporpad->oshape.s1);
    params[2] = g_strdup_printf("%d",croporpad->oshape.s2);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_croporpad_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 3;
    gint numNodeVariables = 0;
    dR_Node* out;
    dR_Shape3 shape;
    if(numINodes!=1)
    {
        g_print("Parsing Error: CropOrPad Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: CropOrPad Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    shape.s0 = atoi(params[0]);
    shape.s1 = atoi(params[1]);
    shape.s2 = atoi(params[2]);
    out = dR_CropOrPad(net, iNodes[0], shape);
    return out;
}

gboolean dR_croporpad_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }


gboolean dR_croporpad_compute(dR_Graph* net, dR_Node* layer){
    dR_CropOrPad_Data* croporpad = ((dR_CropOrPad_Data*)(layer->layer));
    size_t globalWorkSize[1];
    int paramid = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    globalWorkSize[0] = layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;


    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&croporpad->oshape.s0);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&croporpad->oshape.s1);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&croporpad->oshape.s2);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&croporpad->ishape.s0);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&croporpad->ishape.s1);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&croporpad->ishape.s2);          paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Extract Slice Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 1, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:CropOrPad");

}

gboolean dR_croporpad_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_CropOrPad_Data* croporpad = ((dR_CropOrPad_Data*)(layer->layer));
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("Extract Slice Node with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    croporpad->ishape.s0 = lastlayer->oshape.s0;
    croporpad->ishape.s1 = lastlayer->oshape.s1;
    croporpad->ishape.s2 = lastlayer->oshape.s2;

    layer->oshape.s0 = croporpad->oshape.s0;
    layer->oshape.s1 = croporpad->oshape.s1;
    layer->oshape.s2 = croporpad->oshape.s2;
    return TRUE;
}

gint32 dR_croporpad_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_croporpad_createKernel(dR_Graph* net, dR_Node* layer)
{
    return dR_createKernel(net,"cropOrPad",&(layer->clKernel));
}


gboolean dR_croporpad_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_croporpad_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_croporpad_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    if(net->prepared)
        ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_croporpad_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
        g_free((dR_CropOrPad_Data*)(layer->layer));
    return TRUE;
}


gchar* dR_croporpad_printLayer(dR_Node* layer)
{
    dR_CropOrPad_Data* croporpad = ((dR_CropOrPad_Data*)(layer->layer));
    gchar* out;
    out = g_strdup_printf("%s%d%s%d%s%d%s%d%s",
            "Crop or Pad Node: ",layer->layerID,
            "\n Size: [",croporpad->oshape.s0,", ",croporpad->oshape.s1,", ",croporpad->oshape.s2, "]\n");

    return out;
}
