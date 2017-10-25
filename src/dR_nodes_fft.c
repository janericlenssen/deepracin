#include "dR_nodes_fft.h"
#include "dR_core.h"

// ///////////////////////////
// Fast Fourier Transform //
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
    l->printLayer = dR_elemwise2op_printLayer;

    l->generateKernel = NULL;
    l->createKernelName = NULL;
    l->setVariables = NULL;

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
