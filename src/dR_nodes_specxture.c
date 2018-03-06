#include "dR_nodes_specxture.h"
#include "dR_core.h"

dR_Node* dR_Specxture(dR_Graph* net, dR_Node* inputNode1)
{
    // allocate memory for dR_Shape3 (3 dimensional vector)
    dR_Specxture_Data* specxture = g_malloc(sizeof(dR_Specxture_Data));
    // allocate memory for a node
    dR_Node* l = g_malloc(sizeof(dR_Node));

    // set all attributes of fft node
    // dR_Shape3
    l->layer = specxture;
    // node type
    l->type = tSpecxture;
    // set functions (implemented in this file) for this node

    l->compute = dR_specxture_compute;

    l->schedule = dR_specxture_schedule;
    l->propagateShape = dR_specxture_propagateShape;
    l->getRequiredOutputBufferSize = dR_specxture_getRequiredOutputBufferSize;
    l->createKernel = dR_specxture_createKernel;
    l->allocateBuffers = dR_specxture_allocateBuffers;
    l->fillBuffers = dR_specxture_fillBuffers;
    l->cleanupBuffers = dR_specxture_cleanupBuffers;
    l->cleanupLayer = dR_specxture_cleanupLayer;
    l->serializeNode = dR_specxture_serializeNode;
    l->parseAppendNode = dR_specxture_parseAppendNode;

    l->generateKernel = NULL;
    l->createKernelName = NULL;
    l->setVariables = NULL;
    l->printLayer = dR_specxture_printLayer;

    if (inputNode1)
    {
      // create empty list for previous nodes
      l->previous_layers = dR_list_createEmptyList();
      // append the input of this node to the list
      dR_list_append(l->previous_layers,inputNode1);
      // create empty list for following nodes
      l->next_layers = dR_list_createEmptyList();
      // append the current (fft) node as the following node of the previous node
      dR_list_append(inputNode1->next_layers,l);
    }
    else
    {
        g_print("Error: Specxture node needs an appropriate Inputnode");
    }
    // append node to graph
    dR_appendLayer(net, l);
    // return pointer to node
    return l;
}

gchar* dR_specxture_serializeNode(dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables)
{
    dR_Specxture_Data* specxture = (dR_Specxture_Data*)(layer->layer);
    gchar* desc = "specxture";
    gint numNodeParams = 0;
    gint numNodeVariables = 0;
    if(*numParams<numNodeParams||*numVariables<numNodeVariables)
    {
        g_print("Specxture Node needs space for %d parameters and %d variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }
    *numParams = numNodeParams;

    *numVariables = numNodeVariables;
    return desc;
}

dR_Node* dR_specxture_parseAppendNode(dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables)
{
    gint numNodeInputs = 1;
    gint numNodeParams = 0;
    gint numNodeVariables = 0;
    dR_Node* out;
    if(numINodes!=1)
    {
        g_print("Parsing Error: specxture Node needs %d InputNodes but got %d!\n",numNodeInputs,numNodeVariables);
        return NULL;
    }
    if(numParams!=numNodeParams||numVariables!=numNodeVariables)
    {
        g_print("Parsing Error: specxture Node needs %d Parameters and %d Variables!\n",numNodeParams,numNodeVariables);
        return NULL;
    }

    out = dR_Specxture( net, iNodes[0] );
    return out;
}

gboolean dR_specxture_compute(dR_Graph* net, dR_Node* layer){

    dR_Specxture_Data* specxture = ((dR_Specxture_Data*)(layer->layer));
    int paramid = 0;
    dR_list_resetIt(layer->previous_layers);

    //calculate specxture
    printf("\nI WAS HERE.\n");

    return TRUE;

}

gboolean dR_specxture_schedule(dR_Graph* net, dR_Node* layer){
    // Nothing to do
    // Warnings shut up, please
    net = net;
    layer = layer;
    return TRUE;
 }

 gboolean dR_specxture_propagateShape(dR_Graph* net, dR_Node* layer)
 {
     // compute output shape of node
     dR_Specxture_Data* specxture = (dR_Specxture_Data*)(layer->layer);
     dR_Node* lastlayer;

     if(layer->previous_layers->length!=1)
     {
         if(!net->config->silent)
             g_print("Specxture Node with id %d has %d inputs but needs 1!\n",layer->layerID,layer->previous_layers->length);
         return FALSE;
     }
     dR_list_resetIt(layer->previous_layers); // to NULL
     // store last node
     lastlayer = dR_list_next(layer->previous_layers);

     // input shape of specxture node is output of FFTAbs (shifted frequency domain magnitudes)
     specxture->ishape.s0 = lastlayer->oshape.s0;
     specxture->ishape.s1 = lastlayer->oshape.s1;
     specxture->ishape.s2 = lastlayer->oshape.s2;

     gint32 x = specxture->ishape.s0;
     gint32 y = specxture->ishape.s1;

     // store min(x,y)
     specxture->rmax = x > y ? y/2 - 1 : x/2 - 1;

     // store middle of frequency domain
     specxture->x0 = x/2 + 1;
     specxture->y0 = y/2 + 1;

     // as an outputshape, only an array of rmax is needed
     layer->oshape.s0 = specxture->rmax;
     layer->oshape.s1 = 1;
     layer->oshape.s2 = 1;

     return TRUE;
 }

 gint32 dR_specxture_getRequiredOutputBufferSize(dR_Node* layer)
 {
     dR_Specxture_Data* specxture = (dR_Specxture_Data*)(layer->layer);
     gint32 ret;
     ret = specxture->rmax;
     return ret;
 }

 gboolean dR_specxture_createKernel(dR_Graph* net, dR_Node* layer)
 {
     //call all Opencl kernel creation routines required
     dR_Specxture_Data* specxture = (dR_Specxture_Data*)(layer->layer);
     gboolean ret = FALSE;

     ret = TRUE;
     return ret;
 }

 gboolean dR_specxture_allocateBuffers(dR_Graph* net, dR_Node* layer)
 {
     /* create buffer for intermediate steps */
     gboolean ret = TRUE;
     return ret;
 }

 gboolean dR_specxture_fillBuffers(dR_Graph* net, dR_Node* layer)
 {
     net = net;
     layer = layer;
     return TRUE;
 }

 gboolean dR_specxture_cleanupBuffers(dR_Graph* net, dR_Node* layer)
 {
     gboolean ret = TRUE;
     if(net->prepared)
     {
         dR_Specxture_Data* specxture = ((dR_Specxture_Data*)(layer->layer));
         ret &= dR_cleanupKernel((layer->clKernel));
     }
     return ret;
 }

 gboolean dR_specxture_cleanupLayer(dR_Graph* net, dR_Node* layer)
 {
     dR_Specxture_Data* specxture = ((dR_Specxture_Data*)(layer->layer));
     // free all memory that was reserved for node
     if(net->prepared)
         g_free(specxture);
     return TRUE;
 }

 gchar* dR_specxture_printLayer(dR_Node* layer)
 {
     // print node
     dR_Specxture_Data* specxture = (dR_Specxture_Data*)(layer->layer);
     gchar* out;
     out =  g_strdup_printf("%s%d%s", "Specxture operation node: ",layer->layerID, "\n");
     return out;
 }
