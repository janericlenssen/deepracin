#include "dR_nodes_math.h"
#include "dR_core.h"


// ///////////////////////////
// Element-wise 2-operation //
// ///////////////////////////

dR_Node* dR_ElemWise2Operation(dR_Graph* net, dR_Node* inputLayer1, dR_Node* inputLayer2, dR_ElemWise2OperationType op){
    dR_ElemWise2Op_Data* elemwise2op = g_malloc(sizeof(dR_ElemWise2Op_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    elemwise2op->op = op;
    l->layer = elemwise2op;
    l->type = tElemWise2Op;

    l->compute = dR_elemwise2op_compute;
    l->schedule = dR_elemwise2op_schedule;
    l->propagateShape = dR_elemwise2op_propagateShape;
    l->getRequiredOutputBufferSize = dR_elemwise2op_getRequiredOutputBufferSize;
    l->createKernel = dR_elemwise2op_createKernel;
    l->allocateBuffers = dR_elemwise2op_allocateBuffers;
    l->fillBuffers = dR_elemwise2op_fillBuffers;
    l->cleanupBuffers = dR_elemwise2op_cleanupBuffers;
    l->cleanupLayer = dR_elemwise2op_cleanupLayer;
    l->serializeNode = dR_elemwise2op_serializeNode;
    l->parseAppendNode = dR_elemwise2op_parseAppendNode;

    l->generateKernel = NULL;
    l->setVariables = NULL;
    l->createKernelName = NULL;
    l->printLayer = dR_elemwise2op_printLayer;

    if(inputLayer1&&inputLayer2)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer1);
        dR_list_append(l->previous_layers,inputLayer2);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer1->next_layers,l);
        dR_list_append(inputLayer2->next_layers,l);
    }
    else
    {
        g_print("Error: ElemWise2Operation node needs 2 appropriate Inputnodes");
    }


    dR_appendLayer(net, l);
    return l;
}

gchar* dR_elemwise2op_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_ElemWise2Op_Data* elemwise2op = (dR_ElemWise2Op_Data*)(layer->layer);
    gchar* desc = "ElemWise2Op";
    gint numNodeParams = 1;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",elemwise2op->op);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_elemwise2op_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 2;
    gint numNodeParams = 1;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: ElemWise2Op Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: ElemWise2Op Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    out = dR_ElemWise2Operation(net, iNodes[0], iNodes[1], atoi(params[0]));
    return out;
}

gboolean dR_elemwise2op_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }


gboolean dR_elemwise2op_compute(dR_Graph* net, dR_Node* layer){
    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input1, *input2;
    dR_list_resetIt(layer->previous_layers);
    input1 = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    input2 = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;


    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input1);                      paramid++;
    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input2);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);          paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Element-wise 2-operation Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:elemWise2Op");

}

gboolean dR_elemwise2op_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_ElemWise2Op_Data* elemwise2op = (dR_ElemWise2Op_Data*)(layer->layer);
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=2)
    {
        if(!net->config->silent)
            g_print("Elem-wise 2-operation Node with id %d has %d inputs but needs 2!\n",layer->layerID,layer->previous_layers->length);
        return FALSE;
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    elemwise2op->ishape.s0 = lastlayer->oshape.s0;
    elemwise2op->ishape.s1 = lastlayer->oshape.s1;
    elemwise2op->ishape.s2 = lastlayer->oshape.s2;

    lastlayer = dR_list_next(layer->previous_layers);
    if(elemwise2op->ishape.s0!=lastlayer->oshape.s0||elemwise2op->ishape.s1!=lastlayer->oshape.s1||elemwise2op->ishape.s2!=lastlayer->oshape.s2)
    {
        if(!net->config->silent)
        {
            g_print("Elem-wise 2-operation Node needs 2 input nodes with the same shape!\n");
            g_print("[%d, %d, %d] and [%d, %d, %d] not matching!\n",
                    elemwise2op->ishape.s0,elemwise2op->ishape.s1,elemwise2op->ishape.s2,lastlayer->oshape.s0,lastlayer->oshape.s1,lastlayer->oshape.s2);
        }
        return FALSE;
    }
    layer->oshape.s0 = elemwise2op->ishape.s0;
    layer->oshape.s1 = elemwise2op->ishape.s1;
    layer->oshape.s2 = elemwise2op->ishape.s2;
    return TRUE;
}

gint32 dR_elemwise2op_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_elemwise2op_createKernel(dR_Graph* net, dR_Node* layer)
{
    dR_ElemWise2Op_Data* elemwise2op = (dR_ElemWise2Op_Data*)(layer->layer);
    gboolean ret=FALSE;
    switch(elemwise2op->op){
    case tAdd:
        ret = dR_createKernel(net,"elemWiseAdd",&(layer->clKernel));
        break;
    case tSub:
        ret = dR_createKernel(net,"elemWiseSub",&(layer->clKernel));
        break;
    case tMul:
        ret = dR_createKernel(net,"elemWiseMul",&(layer->clKernel));
        break;
    case tDiv:
        ret = dR_createKernel(net,"elemWiseDiv",&(layer->clKernel));
        break;
    case tPow:
        ret = dR_createKernel(net,"elemWisePow",&(layer->clKernel));
        break;
    }
    return ret;
}


gboolean dR_elemwise2op_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_elemwise2op_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_elemwise2op_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    if(net->prepared)
        ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_elemwise2op_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
        g_free((dR_ElemWise2Op_Data*)(layer->layer));
    return TRUE;
}


gchar* dR_elemwise2op_printLayer(dR_Node* layer)
{
    dR_ElemWise2Op_Data* elemwise2op = (dR_ElemWise2Op_Data*)(layer->layer);
    gchar* out;
    gchar* op;
    switch(elemwise2op->op){
    case tAdd:
        op = "Add";
        break;
    case tSub:
        op = "Sub";
        break;
    case tMul:
        op = "Mul";
        break;
    case tDiv:
        op = "Div";
        break;
    case tPow:
        op = "Pow";
        break;
    default:
        op = "Error";
    }
    out = g_strdup_printf("%s%d%s%s%s",
            "Element-wise 2 input operation node: ",layer->layerID,
            "\n Operation: ", op,"\n");

    return out;
}



// ///////////////////////////
// Element-wise 1-operation //
// ///////////////////////////

dR_Node* dR_ElemWise1Operation(dR_Graph* net, dR_Node* inputLayer, dR_ElemWise1OperationType op, gfloat scalar){
    dR_ElemWise1Op_Data* elemwise1op = g_malloc(sizeof(dR_ElemWise1Op_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    elemwise1op->op = op;
    elemwise1op->scalar = scalar;
    l->layer = elemwise1op;
    l->type = tElemWise1Op;

    l->compute = dR_elemwise1op_compute;
    l->schedule = dR_elemwise1op_schedule;
    l->propagateShape = dR_elemwise1op_propagateShape;
    l->getRequiredOutputBufferSize = dR_elemwise1op_getRequiredOutputBufferSize;
    l->createKernel = dR_elemwise1op_createKernel;
    l->allocateBuffers = dR_elemwise1op_allocateBuffers;
    l->fillBuffers = dR_elemwise1op_fillBuffers;
    l->cleanupBuffers = dR_elemwise1op_cleanupBuffers;
    l->cleanupLayer = dR_elemwise1op_cleanupLayer;
    l->serializeNode = dR_elemwise1op_serializeNode;
    l->parseAppendNode = dR_elemwise1op_parseAppendNode;

    l->generateKernel = NULL;
    l->setVariables = NULL;
    l->createKernelName = NULL;
    l->printLayer = dR_elemwise1op_printLayer;

    if(inputLayer)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: ElemWise2Operation node needs an appropriate Inputnode");
    }


    dR_appendLayer(net, l);
    return l;
}

gchar* dR_elemwise1op_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_ElemWise1Op_Data* elemwise1op = (dR_ElemWise1Op_Data*)(layer->layer);
    gchar* desc = "ElemWise1Op";
    gint numNodeParams = 2;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("SerializeNode needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;
    params[0] = g_strdup_printf("%d",elemwise1op->op);
    params[1] = g_strdup_printf("%f",elemwise1op->scalar);

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_elemwise1op_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 2;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: ElemWise1Op Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: ElemWise1Op Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
	out = dR_ElemWise1Operation(net, iNodes[0], atoi(params[0]), (gfloat)atof(params[1]));
    return out;
}

gboolean dR_elemwise1op_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }


gboolean dR_elemwise1op_compute(dR_Graph* net, dR_Node* layer){
    dR_ElemWise1Op_Data* elemwise1op = (dR_ElemWise1Op_Data*)(layer->layer);
    size_t globalWorkSize[3];
    int paramid = 0;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;


    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)input);                      paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);            paramid++;
    if(elemwise1op->op==tAddS||elemwise1op->op==tSubS||elemwise1op->op==tMulS||elemwise1op->op==tDivS||elemwise1op->op==tFill||elemwise1op->op==tPowS)
    {
        net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_float), (void *)&elemwise1op->scalar);          paramid++;
    }

    if (dR_openCLError(net, "Setting kernel args failed.", "Element-wise 1-operation Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:elemWise1Op");

}

gboolean dR_elemwise1op_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_ElemWise1Op_Data* elemwise1op = (dR_ElemWise1Op_Data*)(layer->layer);
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("Elem-wise 1-operation node with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
        return FALSE;
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    elemwise1op->ishape.s0 = lastlayer->oshape.s0;
    elemwise1op->ishape.s1 = lastlayer->oshape.s1;
    elemwise1op->ishape.s2 = lastlayer->oshape.s2;

    layer->oshape.s0 = elemwise1op->ishape.s0;
    layer->oshape.s1 = elemwise1op->ishape.s1;
    layer->oshape.s2 = elemwise1op->ishape.s2;
    return TRUE;
}

gint32 dR_elemwise1op_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_elemwise1op_createKernel(dR_Graph* net, dR_Node* layer)
{
    dR_ElemWise1Op_Data* elemwise1op = (dR_ElemWise1Op_Data*)(layer->layer);
    gboolean ret=FALSE;
    switch(elemwise1op->op){
    case tAddS:
        ret = dR_createKernel(net,"addScalar",&(layer->clKernel));
        break;
    case tSubS:
        ret = dR_createKernel(net,"subScalar",&(layer->clKernel));
        break;
    case tMulS:
        ret = dR_createKernel(net,"mulScalar",&(layer->clKernel));
        break;
    case tDivS:
        ret = dR_createKernel(net,"divScalar",&(layer->clKernel));
        break;
    case tLog:
        ret = dR_createKernel(net,"computeLog",&(layer->clKernel));
        break;
    case tExp:
        ret = dR_createKernel(net,"computeExp",&(layer->clKernel));
        break;
    case tSqrt:
        ret = dR_createKernel(net,"computeSqrt",&(layer->clKernel));
        break;
    case tFill:
        ret = dR_createKernel(net,"fill",&(layer->clKernel));
        break;
    case tPowS:
        ret = dR_createKernel(net,"powScalar",&(layer->clKernel));
        break;
    }
    return ret;
}


gboolean dR_elemwise1op_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_elemwise1op_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_elemwise1op_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    if(net->prepared)
        ret &= dR_cleanupKernel((layer->clKernel));
    return ret;
}

gboolean dR_elemwise1op_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    if(net->prepared)
        g_free((dR_ElemWise1Op_Data*)(layer->layer));
    return TRUE;
}


gchar* dR_elemwise1op_printLayer(dR_Node* layer)
{
    dR_ElemWise1Op_Data* elemwise1op = (dR_ElemWise1Op_Data*)(layer->layer);
    gchar* out;
    gchar* op;
    switch(elemwise1op->op){
    case tAddS:
        op = "Add Scalar";
        break;
    case tSubS:
        op = "Sub Scalar";
        break;
    case tMulS:
        op = "Mul Scalar";
        break;
    case tDivS:
        op = "Div Scalar";
        break;
    case tLog:
        op = "Log";
        break;
    case tExp:
        op = "Exp";
        break;
    case tSqrt:
        op = "Sqrt";
        break;
    case tFill:
        op = "Fill";
        break;
    case tPowS:
        op = "Pow Scalar";
        break;
    default:
        op = "Error";
    }
    if(elemwise1op->op==tAddS||elemwise1op->op==tSubS||elemwise1op->op==tMulS||elemwise1op->op==tDivS||elemwise1op->op==tFill||elemwise1op->op==tPowS)
    {
        out = g_strdup_printf("%s%d%s%s%s%f%s",
            "Element-wise 1 input operation node: ",layer->layerID,
            "\n Operation: ", op,"\n Scalar: ",elemwise1op->scalar,"\n");
    }
    else
    {
        out = g_strdup_printf("%s%d%s%s%s",
            "Element-wise 1 input operation node: ",layer->layerID,
            "\n Operation: ", op,"\n");
    }
    return out;
}



// ///////////
// Softmax  //
// ///////////


dR_Node* dR_Softmax(dR_Graph* net, dR_Node* inputLayer){
    dR_Softmax_Data* softmax = g_malloc(sizeof(dR_Softmax_Data));
    dR_Node* l = g_malloc(sizeof(dR_Node));
    l->layer = softmax;
    l->type = tSoftmax;

    // Mandatory
    l->compute = dR_softmax_compute;
    l->schedule = dR_softmax_schedule;
    l->propagateShape = dR_softmax_propagateShape;
    l->getRequiredOutputBufferSize = dR_softmax_getRequiredOutputBufferSize;
    l->createKernel = dR_softmax_createKernel;
    l->allocateBuffers = dR_softmax_allocateBuffers;
    l->fillBuffers = dR_softmax_fillBuffers;
    l->cleanupBuffers = dR_softmax_cleanupBuffers;
    l->cleanupLayer = dR_softmax_cleanupLayer;
    l->serializeNode = dR_softmax_serializeNode;
    l->parseAppendNode = dR_softmax_parseAppendNode;

    // Optional
    l->generateKernel = NULL;
    l->setVariables = NULL;
    l->createKernelName = NULL;
    l->printLayer = dR_softmax_printLayer;

    if(inputLayer!=NULL)
    {
        l->previous_layers = dR_list_createEmptyList();
        dR_list_append(l->previous_layers,inputLayer);
        l->next_layers = dR_list_createEmptyList();
        dR_list_append(inputLayer->next_layers,l);
    }
    else
    {
        g_print("Error: Softmax Layer needs an appropriate Inputnode\n");
    }

    dR_appendLayer(net, l);
    return l;
}

gchar* dR_softmax_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    gchar* desc = "Softmax";
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

dR_Node* dR_softmax_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 0;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: Softmax Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: Softmax Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    out = dR_Softmax(net, iNodes[0]);
    return out;
}

gboolean dR_softmax_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }


gboolean dR_softmax_compute(dR_Graph* net, dR_Node* layer){
    dR_Softmax_Data* softmaxlayer = (dR_Softmax_Data*)(layer->layer);
    cl_float sumExp = 0.0f;
    size_t globalWorkSize[3];
    gint paramid = 0;
    gint i;
    gint inputsize = softmaxlayer->ishape.s0*softmaxlayer->ishape.s1*softmaxlayer->ishape.s2;
    cl_mem* input;
    dR_list_resetIt(layer->previous_layers);
    input = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;


    //Compute Exp(x)

    globalWorkSize[0] = softmaxlayer->ishape.s0;
    globalWorkSize[1] = softmaxlayer->ishape.s1;
    globalWorkSize[2] = softmaxlayer->ishape.s2;

    paramid=0;
    net->clConfig->clError = clSetKernelArg(softmaxlayer->clKernelComputeExp, paramid, sizeof(cl_mem), (void *)input);                           paramid++;
    net->clConfig->clError |= clSetKernelArg(softmaxlayer->clKernelComputeExp, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);       paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Compute Exp(x) Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, softmaxlayer->clKernelComputeExp, 3, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    dR_finishCLKernel(net, "deepRACIN:softmaxExp");

    // CPU implementation of the expsum-reduction (inefficient for large input matrices)
    dR_downloadArray(net, "reduceExps", layer->outputBuf->bufptr, 0 /*offset*/, inputsize * sizeof(cl_float), softmaxlayer->expsHost);

    for(i = 0; i<inputsize;i++)
    {
        sumExp += softmaxlayer->expsHost[i];
    }

    // Multiply with 1/sum with mulScalar kernel

    sumExp = 1/sumExp;
    globalWorkSize[0] = layer->oshape.s0;
    globalWorkSize[1] = layer->oshape.s1;
    globalWorkSize[2] = layer->oshape.s2;

    paramid = 0;
    net->clConfig->clError = clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);                           paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_mem), (void *)layer->outputBuf->bufptr);       paramid++;
    net->clConfig->clError |= clSetKernelArg(layer->clKernel, paramid, sizeof(cl_float), (void *)&sumExp);                      paramid++;

    if (dR_openCLError(net, "Setting kernel args failed.", "Mul Scalar for softmax Kernel"))
        return FALSE;
    // execute kernel
     net->clConfig->clError = clEnqueueNDRangeKernel(net->clConfig->clCommandQueue, layer->clKernel, 3, NULL, globalWorkSize,
        NULL, 0, NULL, net->clConfig->clEvent);

    return dR_finishCLKernel(net, "deepRACIN:softmaxNorm");

}

gboolean dR_softmax_propagateShape(dR_Graph* net, dR_Node* layer)
{
    dR_Softmax_Data* softmaxlayer = ((dR_Softmax_Data*)(layer->layer));
    dR_Node* lastlayer;
    if(layer->previous_layers->length!=1)
    {
        if(!net->config->silent)
            g_print("Softmax Layer with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
    }
    dR_list_resetIt(layer->previous_layers);
    lastlayer = dR_list_next(layer->previous_layers);
    softmaxlayer->ishape.s0 = lastlayer->oshape.s0;
    softmaxlayer->ishape.s1 = lastlayer->oshape.s1;
    softmaxlayer->ishape.s2 = lastlayer->oshape.s2;

    layer->oshape.s0 = softmaxlayer->ishape.s0;
    layer->oshape.s1 = softmaxlayer->ishape.s1;
    layer->oshape.s2 = softmaxlayer->ishape.s2;
    return TRUE;
}

gint32 dR_softmax_getRequiredOutputBufferSize(dR_Node* layer)
{
    return layer->oshape.s0*layer->oshape.s1*layer->oshape.s2;
}

gboolean dR_softmax_createKernel(dR_Graph* net, dR_Node* layer)
{
    dR_Softmax_Data* softmax = (dR_Softmax_Data*)(layer->layer);
    gchar* kernelname;
    kernelname = "mulScalar";
    dR_createKernel(net,"computeExp",&(softmax->clKernelComputeExp));
    return dR_createKernel(net,kernelname,&(layer->clKernel));
}


gboolean dR_softmax_allocateBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    if(!net->prepared)
    {
        dR_Softmax_Data* softmax = ((dR_Softmax_Data*)(layer->layer));
        softmax->expsHost = g_malloc(softmax->ishape.s0*softmax->ishape.s1*softmax->ishape.s2*sizeof(gfloat));
    }
    return ret;
}

gboolean dR_softmax_fillBuffers(dR_Graph* net, dR_Node* layer)
{
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
}

gboolean dR_softmax_cleanupBuffers(dR_Graph* net, dR_Node* layer)
{
    gboolean ret = TRUE;
    if(net->prepared)
    {
        dR_Softmax_Data* softmax = ((dR_Softmax_Data*)(layer->layer));
        ret &= dR_cleanupKernel((softmax->clKernelComputeExp));
        ret &= dR_cleanupKernel((layer->clKernel));
    }
    return ret;
}

gboolean dR_softmax_cleanupLayer(dR_Graph* net, dR_Node* layer)
{
    dR_Softmax_Data* softmax = ((dR_Softmax_Data*)(layer->layer));
    if(net->prepared)
    {
        g_free(softmax->expsHost);
        g_free((dR_Softmax_Data*)(layer->layer));
    }
    return TRUE;
}

gchar* dR_softmax_printLayer(dR_Node* layer)
{
    //dR_Softmax_Data* softmax = (dR_Softmax_Data*)(layer->layer);
    gchar* out;
    out = g_strdup_printf("%s%d%s",
            "Softmax Layer: ",layer->layerID,"\n");
    return out;
}

