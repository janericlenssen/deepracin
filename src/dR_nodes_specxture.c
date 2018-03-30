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

void halfcircle(gint32 r, gint32* xc, gint32* yc, gint32 x0, gint32 y0, gfloat* out);
void intline(gint32 x1, gint32 y1, gint32 x2, gint32 y2, gfloat* out);

gboolean dR_specxture_compute(dR_Graph* net, dR_Node* layer){

    dR_Specxture_Data* specxture = ((dR_Specxture_Data*)(layer->layer));
    gfloat* out = (gfloat*)layer->outputBuf->bufptr;

    // Download the shifted FFT magnitude to the host into out.
    size_t numBytes = specxture->x0*specxture->y0*4*sizeof(cl_float);
    dR_list_resetIt(layer->previous_layers);
    cl_mem* fftMagArray;
    fftMagArray = ((dR_Node*)dR_list_next(layer->previous_layers))->outputBuf->bufptr;
    dR_downloadArray(net, "ffMagDownload", fftMagArray, 0, numBytes, out);

    int paramid = 0;
    dR_list_resetIt(layer->previous_layers);

    printf("\n**SPECXTURE BEGIN**\n");

    int width = specxture->x0*2;
    gint32 rmax = specxture->rmax;
    //printf("\nrmax: %d\n", rmax);
    // array for storing srad values
    gfloat srad[rmax];
    gfloat sang[180];

    // array for integer lines
    //gint32 intline[2*width];

    gint32 xc[180];
    gint32 yc[180];

    srad[0] = 0; // we do not use this array element
    srad[1] = 1; // store DC component here

    gint32 prev_xc = 0;
    gint32 prev_yc = 0;
    gint32 index_cart;

    gint32 curr_radius;
    // compure srad. This loop has N/2 iterations (for N*N image)
    for(curr_radius = 2; curr_radius <= rmax; curr_radius++)
    {
        srad[curr_radius] = 0;
        // create halfcircle. Loop in halfcircle has 180 iterations
        halfcircle(curr_radius, xc, yc, specxture->x0, specxture->y0, out);
        // sum on circle. 180 iterations
        for(gint32 i = 0; i < 180; i++)
        {
            // only sum when there is a new coordinate
            if(i == 0 || prev_xc != xc[i] || prev_yc != yc[i])
            {
                //printf("\nI: %d\n", i);
                //printf("\nxc: %d, yc: %d\n", xc[i], yc[i]);
                index_cart = yc[i]*width + xc[i];
                //printf("\nout[%d] = %.0f\n", index_cart, out[index_cart]);
                srad[curr_radius] += out[index_cart];
                prev_xc = xc[i];
                prev_yc = yc[i];
            }
        }
        printf("\nsrad[%d] = %0.f\n", curr_radius, srad[curr_radius]);
    }

    prev_xc = 0;
    prev_yc = 0;

    // calculate sang
    // make room for coordinates

    #if 0
    for(gint32 i = 0; i < 180; i++)
    {
        // only sum when there is a new coordinate
        if(i == 0 || prev_xc != xc[i] || prev_yc != yc[i])
        {
            printf("\nx1,y1:  %d, %d\n", specxture->x0, specxture->y0);
            /******
            gint32 x2 = 2;
            gint32 y2 = 3;
            */
            printf("\nx2,y2: %d, %d\n", xc[i], yc[i]);
            intline(specxture->x0, specxture->y0, xc[i], yc[i], out);
            //printf("\nI: %d\n", i);
            //printf("\nxc: %d, yc: %d\n", xc[i], yc[i]);
            prev_xc = xc[i];
            prev_yc = yc[i];
        }
    }
    #endif
    printf("\n**SPECXTURE END**\n");
    return TRUE;
}

// compute coordinates of a halfcircle around x0 y0 with radius r.
void halfcircle(gint32 r, gint32* xc, gint32* yc, gint32 x0, gint32 y0, gfloat* out)
{
    gfloat theta[180];
    gint32 array_index;
    int n = x0*2;
    /*
    // for testing;
    for(int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
          out[i*n + j] = 0.0;
      }
    }
    */
    for( gint32 angle = 91; angle <= 270; angle++ )
    {
      array_index = angle-91;
      theta[array_index] = angle*(M_PI_F/180); // in radiants
       //printf("%d ,", theta[angle-91] );
       xc[array_index] = round(r*cos(theta[array_index])) + x0;
       yc[array_index] = round(r*sin(theta[array_index])) + y0;
       //printf("\narrayIndex: %d, cartX: %d, cartY: %d\n", array_index, xc[array_index], yc[array_index]);
       //out[yc[array_index]*n + xc[array_index]] = 1.0;
    }
    //printf("\nx0: %d, y0: %d\n", x0, y0);
     // for testing
     /*
    for(int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
          printf("%.0f, ", out[i*n + j]);
      }
      printf("\n");
    }
    */
}

// Computes the coordinates of a straight line segment extending from center (x1, y1), to (x2, y2).
void intline(gint32 x1, gint32 y1, gint32 x2, gint32 y2, gfloat* out)
{
    gint32 n = x1*2;
    gint32 flip = 0;
    gfloat m = 0.0;

    gint32 dx = abs(x2 - x1);
    gint32 dy = abs(y2 - y1);

    // no line to draw
    if((dx == 0) && (dy == 0))
    {
        x2 = x1;
        y2 = y1;
        return;
    }

    for(int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
          out[i*n + j] = 0.0;
      }
    }

    // always take the longer variable, otherwise approximation of the line would be worse
    if(dx >= dy)
    {
        printf("\nfirst\n");
        if(x1 > x2)
        {
            // swap x1 with x2, and y1 with y2 to always draw from (x1,y1) to (x2,y2), left to right
            gint32 temp;
            temp = x1; x1 = x2; x2 = temp;
            temp = y1; y1 = y2; y2 = temp;
            flip = 1;
        }
        printf("\nx1,y1: %d, %d | x2,y2: %d, %d\n", x1, y1, x2, y2);
        // calculate gradient
        m = (gfloat)(y2 - y1) / (x2 - x1);
        printf("\nm = %.2f\n", m);
        // create an array X of length dx which has all values between x1 and x2
        gint32 *x_coord = (gint32 *)malloc(dx * sizeof(gint32));
        gint32 *y_coord = (gint32 *)malloc(dx * sizeof(gint32));
        printf("\n Output of sang:\n");

        printf("\nx_coord:");
        for(gint32 i = 0; i < dx; i++)
        {
            x_coord[i] = x1 + i;
            printf("%d,", x_coord[i]);
        }
        printf("\n");

        printf("\ny_coord:");
        for(gint32 i = 0; i < dx; i++)
        {
            //printf("%d,%d | ", x_coord[i], y_coord[i]);
            y_coord[i] = round(y1 + m*(x_coord[i]-x1));
            printf("%d,", y_coord[i]);
            out[y_coord[i]*n + x_coord[i]] = 1.0;
        }
        printf("\n");

        for(int i = 0; i < n; i++)
        {
           for (int j = 0; j < n; j++)
           {
               printf("%.0f, ", out[i*n + j]);
           }
           printf("\n");
        }

        free(x_coord);
        free(y_coord);
        // create another array Y of length dx which calculates y = round(y1 + m*(x - x1)) for all values in X
    }
    else
    {
        printf("\nsecond\n");
        if(y1 > y2)
        {
            // swap x1 with x2, and y1 with y2 to always draw from (x1,y1) to (x2,y2)
            gint32 temp;
            temp = x1; x1 = x2; x2 = temp;
            temp = y1; y1 = y2; y2 = temp;
            flip = 1;
        }
        printf("\nx1,y1: %d, %d | x2,y2: %d, %d\n", x1, y1, x2, y2);
        // calculate slope
        m = (gfloat)(x2 - x1) / (y2 - y1);
        // x = round(x1 + m*(y - y1));
        gint32 *y_coord = (gint32 *)malloc(dy * sizeof(gint32));
        gint32 *x_coord = (gint32 *)malloc(dy * sizeof(gint32));

        printf("\nx_coord:");
        for(gint32 i = 0; i < dy; i++)
        {
            y_coord[i] = y1 + i;
            printf("%d,", y_coord[i]);
        }

        printf("\n Output of sang:\n");
        for(gint32 i = 0; i < dy; i++)
        {
            x_coord[i] = round(x1 + m*(y_coord[i]-y1));
            out[y_coord[i]*n + x_coord[i]] = 1.0;
        }

        for(int i = 0; i < n; i++)
        {
          for (int j = 0; j < n; j++)
          {
              printf("%.0f, ", out[i*n + j]);
          }
          printf("\n");
        }
        free(x_coord);
        free(y_coord);
      }

    if (flip)
    {

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
