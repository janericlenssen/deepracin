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
    gfloat* out = (gfloat*)layer->outputBuf->bufptr;

    int paramid = 0;
    dR_list_resetIt(layer->previous_layers);

    //calculate specxture
    printf("\n**SPECXTURE BEGIN**\n");

    gint32 rmax = specxture->rmax;
    printf("\nrmax: %d\n", rmax);
    // array for storing srad values
    gfloat srad[rmax];

    gint32 xc[180];
    gint32 yc[180];

    srad[0] = 0; // we do not use this value
    srad[1] = 1; // store DC component here

    for( gint32 curr_radius = 2; curr_radius <= rmax; curr_radius++ )
    {
      halfcircle(curr_radius, xc, yc, specxture->x0, specxture->y0, out);
    }

    printf("\n**SPECXTURE END**\n");
    return TRUE;
}

void halfcircle(gint32 r, gint32* xc, gint32* yc, gint32 x0, gint32 y0, gfloat* out)
{
    gfloat theta[180];
    gint32 array_index;

    // for testing
    int n = x0*2;
    for(int i = 0; i<n; i++)
    {
      for (int j = 0; j < n; j++)
      {
          out[i*n + j] = 0.0;
      }
    }

    for( gint32 angle = 91; angle <= 270; angle++ )
    {
      array_index = angle-91;
      theta[array_index] = angle*(M_PI_F/180); // in radiants
       //printf("%d ,", theta[angle-91] );
       //in polar coordinates
       xc[array_index] = round(r*cos(theta[array_index])) + x0;
       yc[array_index] = round(r*sin(theta[array_index])) + y0;
       //printf("\narrayIndex: %d, cartX: %d, cartY: %d\n", array_index, xc[array_index], yc[array_index]);
       //for testing
       out[yc[array_index]*n + xc[array_index]] = 1.0;
    }
    printf("\nx0: %d, y0: %d\n", x0, y0);

    // for testing
    for(int i = 0; i<n; i++)
    {
      for (int j = 0; j < n; j++)
      {
          printf("%.0f, ", out[i*n + j]);
      }
      printf("\n");
    }

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
     // TODO: not sure if x/2 + 1. but that would not be the middle
     specxture->x0 = x/2;
     specxture->y0 = y/2;

     // as an outputshape, only an array of rmax is needed
     layer->oshape.s0 = lastlayer->oshape.s0; //specxture->rmax;
     layer->oshape.s1 = lastlayer->oshape.s1;//1;
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
