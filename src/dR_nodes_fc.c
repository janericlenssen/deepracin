#include "dR_nodes_fc.h"
#include "dR_core.h"

// Mandatory

dR_Node* dR_FullyConnected(dR_Graph* net, dR_Node* inputLayer, dR_Shape2 sh, dR_ActivationType acttype, gboolean useB){
    dR_FC_Data* fc = g_malloc(sizeof(dR_FC_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    fc->shape = sh;
    fc->activation = acttype;
    fc->useBias = useB;
    fc->hasVariables = FALSE;
    l->layer = fc;
    l->type = tFullyConnected;

    l->compute = dR_fc_compute;
    l->schedule = dR_fc_schedule;
    l->propagateShape = dR_fc_propagateShape;
    l->getRequiredOutputBufferSize = dR_fc_getRequiredOutputBufferSize;
    l->createKernel = dR_fc_createKernel;
    l->allocateBuffers = dR_fc_allocateBuffers;
    l->fillBuffers = dR_fc_fillBuffers;
    l->cleanupBuffers = dR_fc_cleanupBuffers;
    l->cleanupLayer = dR_fc_cleanupLayer;
    l->serializeNode = dR_fc_serializeNode;
    l->parseAppendNode = dR_fc_parseAppendNode;

    l->generateKernel = NULL;
    l->setVariables = dR_FullyConnected_setVariables;
    l->createKernelName = NULL;
    l->printLayer = dR_fc_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: Fully Connected needs an appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_fc_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_FC_Data* fclayer = (dR_FC_Data*)(layer->layer);
    gchar* desc = "FullyConnected";
    gint numNodeParams = 4;
    gint numNodeVariables;
    if(fclayer->useBias)
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
    params[0] = g_strdup_printf("%d",fclayer->activation);
    params[1] = g_strdup_printf("%d",fclayer->useBias?1:0);
    params[2] = g_strdup_printf("%d",fclayer->shape.s0);
    params[3] = g_strdup_printf("%d",fclayer->shape.s1);

    *numVariables = numNodeVariables;
    variableSizes[0] = fclayer->shape.s0*fclayer->shape.s1;
    variables[0] = fclayer->weights;
    if(fclayer->useBias)
    {
        variableSizes[1] = fclayer->shape.s1;
        variables[1] = fclayer->biases;
    }
    return desc;
}

dR_Node* dR_fc_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 4;
    gint numNodeVariables = 2;
    dR_Shape2 shape;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: FullyConnected Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables<numNodeVariables-1||numVariables>numNodeVariables)
    {
        g_print("Parsing Error: FullyConnected Node needs %d Parameters and %d or %d Variables!\n",numNodeParams,numNodeVariables-1,numNodeVariables);
        return NULL;
    }
    shape.s0 = atoi(params[2]);
    shape.s1 = atoi(params[3]);

    out = dR_FullyConnected(net, iNodes[0], shape ,atoi(params[0]), atoi(params[1]));
    dR_FullyConnected_setVariables(out,variables[0],atoi(params[1])?variables[1]:NULL);
    return out;
}


gboolean dR_fc_schedule(dR_Graph* net, dR_Node* layer){

    dR_FC_Data* fclayer = (dR_FC_Data*)(layer->layer);
    cl_ulong lms;
	size_t mws;
    gint lws = fclayer->shape.s1;
    gint z=fclayer->shape.s0;
    gint localMemoryConInput = (lws*z*4);
    gint overhead = 0;
    gint iterations = 0;
    gint numberofWG = 0;
    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lms, NULL);
    clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &mws, NULL);

    //WorkGroup size as big as possible
    while(lws>(gint)mws)
    {
        iterations++;
        if(lws%2==0)
        {
            lws/=2;
        }
        else
        {
            lws/=2+1;
            overhead+=iterations;
        }
    }

    //Size of Partition as big as possible
    localMemoryConInput = (z*4);
    iterations=0;
    while(localMemoryConInput>(gint)lms)
    {
        iterations++;
        if(z%2==0)
        {
            z/=2;
        }
        else
        {
            z/=2+1;
            overhead+=iterations;
        }
        localMemoryConInput = (z*4);
    }

    // Trying to have more than 60 Workgroups
    numberofWG = fclayer->shape.s1/lws * fclayer->shape.s0/z;
    while(numberofWG<60&&z!=1)
    {
        iterations++;
        if(z%2==0)
        {
            z/=2;
        }
        else
        {
            z/=2+1;
            overhead+=iterations;
        }
        numberofWG = fclayer->shape.s1/lws * fclayer->shape.s0/z;
    }

    fclayer->localWorkSizeStepOne = lws;
    fclayer->sizeOfPartitions = z;

    // Creating Global Worksize as a multiple of local work size
    fclayer->globalWorkSizeX = fclayer->shape.s1+(fclayer->shape.s1%lws);
    if(fclayer->shape.s0%fclayer->sizeOfPartitions!=0)
    {
        fclayer->globalWorkSizeY = (fclayer->shape.s0/fclayer->sizeOfPartitions)+1;
    }
    else
    {
        fclayer->globalWorkSizeY = fclayer->shape.s0/fclayer->sizeOfPartitions;
    }
    if(!net->config->silent&&net->config->debugInfo)
        g_print("lws: %d, sizeofpartitions: %d \n gwsX: %d, gwsY: %d\n",lws,z,(gint)fclayer->globalWorkSizeX,(gint)fclayer->globalWorkSizeY);

    if(!net->config->silent&&net->config->debugInfo)
        g_print("Scheduled FullyConnected Layer!\n");


    return TRUE;
 }

gboolean dR_fc_compute(dR_Graph* net, dR_Node* layer){
    dR_FC_Data* fclayer = ((dR_FC_Data*)(layer->layer));
    size_t localWorkSize[2];
    size_t globalWorkSize[2];
    cl_int lMemSize = fclayer->sizeOfPartitions;
    gboolean ret = TRUE;
    gint paramid = 0;
	cl_int numberOfPartitions;
    //step one
    cl_int numInputNeurons = fclayer->shape.s0;
    cl_int numOutputNeurons = fclayer->shape.s1;
    cl_int inputSizeX = fclayer->ishape.s0;
    cl_int inputSizeY = fclayer->ishape.s1;
    cl_int inputSizeZ = fclayer->ishape.s2;
    cl_int biasAdd = 0;
    cl_int activationtype = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;

    localWorkSize[0] = fclayer->localWorkSizeStepOne;
    localWorkSize[1] = 1;

    globalWorkSize[0] = fclayer->globalWorkSizeX;
    globalWorkSize[1] = fclayer->globalWorkSizeY;


    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                       paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)&fclayer->weightsBuf);       paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, lMemSize * sizeof(cl_float), NULL);                  paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);   paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&numInputNeurons);           paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&numOutputNeurons);          paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&fclayer->sizeOfPartitions); paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&inputSizeX);                paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&inputSizeY);                paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_int), (void *)&inputSizeZ);                paramid++;


    if (dR_openCLError(net, "Setting kernel args failed.", "FullyConnected Kernel1"))
        return FALSE;
    //execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 2, NULL, globalWorkSize,
        localWorkSize, 0, NULL, net->clConfig->clEvent);
    ret &= dR_finishCLKernel(net, "deepRACIN:FullyConnectedStep1");


    // Step two
    numberOfPartitions = fclayer->shape.s0/fclayer->sizeOfPartitions;
    globalWorkSize[0] = fclayer->shape.s1;
    if(fclayer->useBias)
        biasAdd = 1;

    if(fclayer->activation == tReLU)
        activationtype = 1;
    else if(fclayer->activation == tSigmoid)
        activationtype = 2;
    else if(fclayer->activation == tTan)
        activationtype = 3;

    net->clConfig->clError |= clSetKernelArg(fclayer->clKernelStepTwo, 0, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);
    net->clConfig->clError |= clSetKernelArg(fclayer->clKernelStepTwo, 1, sizeof(cl_mem), (void *)&fclayer->biasBuf);
    net->clConfig->clError |= clSetKernelArg(fclayer->clKernelStepTwo, 2, sizeof(cl_int), (void *)&biasAdd);
    net->clConfig->clError |= clSetKernelArg(fclayer->clKernelStepTwo, 3, sizeof(cl_int), (void *)&activationtype);
    net->clConfig->clError |= clSetKernelArg(fclayer->clKernelStepTwo, 4, sizeof(cl_int), (void *)&numOutputNeurons);
    net->clConfig->clError |= clSetKernelArg(fclayer->clKernelStepTwo, 5, sizeof(cl_int), (void *)&numberOfPartitions);


    if (dR_openCLError(net, "Setting kernel args failed.", "FullyConnected Kernel2"))
        return FALSE;
    //execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, fclayer->clKernelStepTwo, 1, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    ret &= dR_finishCLKernel(net, "deepRACIN:FullyConnectedStep2");

    return ret;
}

gboolean dR_fc_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_FC_Data* fclayer = ((dR_FC_Data*)(layer->layer));
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("FullyConnected Layer with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    fclayer->ishape.s0 = lastlayer->oshape.s0;
    fclayer->ishape.s1 = lastlayer->oshape.s1;
    fclayer->ishape.s2 = lastlayer->oshape.s2;

    // Check InputSizes
    if(fclayer->ishape.s0*fclayer->ishape.s1*fclayer->ishape.s2 != fclayer->shape.s0)
    {
        if(!net->config->silent)
            g_print("FullyConnected Layer does not match Input Size! \n");
        return FALSE;
    }
    layer->oshape.s0 = fclayer->shape.s1;
    layer->oshape.s1 = 1;
    layer->oshape.s2 = 1;
    return TRUE;
}

gint32 dR_fc_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0;
}

gboolean dR_fc_createKernel(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    ret &= dR_createKernel(net,"matrixVectorMultFirst",&(layer->clKernel));
    ret &= dR_createKernel(net,"matrixVectorMultSecond",&(((dR_FC_Data*)(layer->layer))->clKernelStepTwo));
    return ret;
}

gboolean dR_fc_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_Shape2 shape;
    dR_FC_Data* fclayer = ((dR_FC_Data*)(layer->layer));
    shape = fclayer->shape;
    ret &= dR_createFloatBuffer(net, &(((dR_FC_Data*)(layer->layer))->weightsBuf),shape.s0*shape.s1, CL_MEM_READ_ONLY);
    if(fclayer->useBias)
        ret &= dR_createFloatBuffer(net, &(((dR_FC_Data*)(layer->layer))->biasBuf),shape.s1, CL_MEM_READ_ONLY);
    return ret;
}

gboolean dR_fc_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    dR_FC_Data* fclayer = ((dR_FC_Data*)(layer->layer));
    ret &= dR_uploadArray(net,"",fclayer->weights,0,fclayer->shape.s0*fclayer->shape.s1*sizeof(float),fclayer->weightsBuf);
    if(fclayer->useBias)
        ret &= dR_uploadArray(net,"",fclayer->biases,0,fclayer->shape.s1*sizeof(float),fclayer->biasBuf);
    return ret;
}

gboolean dR_fc_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    ret &= dR_clMemoryBufferCleanup(net, ((dR_FC_Data*)(layer->layer))->weightsBuf);
    ret &= dR_clMemoryBufferCleanup(net, ((dR_FC_Data*)(layer->layer))->biasBuf);
    ret &= dR_cleanupKernel((layer->clKernel));
    ret &= dR_cleanupKernel((((dR_FC_Data*)(layer->layer))->clKernelStepTwo));
    return ret;
}

gboolean dR_fc_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
    {
        g_free(((dR_FC_Data*)(layer->layer))->weights);
        g_free(((dR_FC_Data*)(layer->layer))->biases);
        g_free((dR_FC_Data*)(layer->layer));
    }
    return TRUE;
}


// Optional

void dR_FullyConnected_setVariables(dR_Node* layer, gfloat* weights, gfloat* biases)
{
    dR_FC_Data* fclayer = ((dR_FC_Data*)(layer->layer));
    int i;
    fclayer->weights = g_malloc(fclayer->shape.s0*fclayer->shape.s1*sizeof(gfloat));
    for(i=0; i<fclayer->shape.s0*fclayer->shape.s1;i++)
    {
        fclayer->weights[i] = weights[i];
    }
    if(fclayer->useBias)
    {
        fclayer->biases = g_malloc(fclayer->shape.s1*sizeof(gfloat));
        for(i=0; i<fclayer->shape.s1;i++)
        {
            fclayer->biases[i] = biases[i];
        }
    }
    fclayer->hasVariables = 1;
}


gchar* dR_fc_printLayer(dR_Node* layer)
{
    dR_FC_Data* fclayer = (dR_FC_Data*)(layer->layer);
    gchar *out;
    gchar* actstr;
    gchar* biasstr;
    if(fclayer->activation == tReLU)
        actstr = "ReLU";
    else if(fclayer->activation == tLinear)
        actstr = "Linear";
    else if(fclayer->activation == tSigmoid)
        actstr = "Sigmoid";
    else if(fclayer->activation == tTan)
        actstr = "Tan";
    if(fclayer->useBias)
        biasstr = "Yes";
    else
        biasstr = "No";
    out = g_strdup_printf("%s%d%s%d%s%d%s%s%s%s%s",
            "FullyConnected Layer: ",layer->layerID,"\n Input Neurons: ", fclayer->shape.s0,
            "\n Output Neurons: ", fclayer->shape.s1,
            "\n Activation: ",actstr,
            "\n Use bias: ", biasstr,"\n");
    /*if(printVariables&&fclayer->hasVariables)
    {
        char buffer[5];
        char weights[1000000];
        int i;
        for(i = 0; i<fclayer->shape.s0*fclayer->shape.s1;i++)
        {
            sprintf(buffer,"%.3f, ", fclayer->weights[i]);
            strcat(weights,buffer);
        }
        strcat(out,"Weights:\n");
        strcat(out,weights);
        strcat(out,"\n\n");
        if(fclayer->useBias)
        {
            for(i = 0; i<fclayer->shape.s1;i++)
            {
                sprintf(buffer,"%.3f, ", fclayer->biases[i]);
                strcat(weights,buffer);
            }
            strcat(out,"Biases:\n");
            strcat(out,weights);
            strcat(out,"\n\n");
        }
    }*/
    return out;
}

