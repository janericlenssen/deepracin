#include "dR_parser.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "deepRACIN.h"

char* own_strtok_r(
    char *str,
    const char *delim,
    char **nextp)
{
    char *ret;

    if (str == NULL)
    {
        str = *nextp;
    }

    str += strspn(str, delim);

    if (*str == '\0')
    {
        return NULL;
    }

    ret = str;

    str += strcspn(str, delim);

    if (*str)
    {
        *str++ = '\0';
    }

    *nextp = str;

    return ret;
}

float* loadVariables(char* path, int length)
{
 gfloat* ret = g_malloc(length*sizeof(gfloat));
 FILE *fp;
 fp = fopen(path,"rb");
 if( fp == NULL )
 {
    g_print("Error while opening the model file.\n");
    return 0;
 }
 fread(ret,sizeof(gfloat),length,fp);
 return ret;
}

gboolean dR_serializeGraph(dR_Graph* net, gchar* path)
{
    gchar* source, *temp;
    gint length = 0;
    gint i;
    dR_Node* current_node;
    gchar* desc;
    gchar* params[10];
    gint num_params = 10;
    gfloat* variables[2];
    gint num_variables = 2;
    gint variable_sizes[2];
    gint global_variable_counter = 0;
    dR_Node* it;
    FILE *fp;
    gchar* folderPath;
    gchar* filePath;
    gchar* folderName;
    folderName = g_strdup_printf("dr_graph_export");
    folderPath = g_build_filename(path,folderName, NULL);
    g_mkdir(folderPath,0777);
    source = g_strdup_printf(
    "deepRACIN graph: %d nodes, %d feed nodes",net->allNodes->length,net->feed_layers->length);

    dR_list_resetIt(net->allNodes);
    current_node = (dR_Node*)dR_list_next(net->allNodes);
    while(current_node)
    {
        desc = current_node->serializeNode(current_node, params, &num_params, variables, variable_sizes, &num_variables);

        temp = g_strdup_printf(
        "\nNode %d: %s\nparams:",current_node->layerID,desc);
        source = concat_and_free_old(source,temp);
        for(i=0;i<num_params;i++)
        {
            temp = g_strdup_printf(
            " %s",params[i]);
            source = concat_and_free_old(source,temp);
        }
        temp = g_strdup_printf(
        "\nprevious_nodes:");
        source = concat_and_free_old(source,temp);
        dR_list_resetIt(current_node->previous_layers);
        it = (dR_Node*)dR_list_next(current_node->previous_layers);
        while(it)
        {
            temp = g_strdup_printf(
            " %d",it->layerID);
            source = concat_and_free_old(source,temp);
            it = (dR_Node*)dR_list_next(current_node->previous_layers);
        }
        temp = g_strdup_printf(
        "\nnext_nodes:");
        source = concat_and_free_old(source,temp);
        dR_list_resetIt(current_node->next_layers);
        it = (dR_Node*)dR_list_next(current_node->next_layers);
        while(it)
        {
            temp = g_strdup_printf(
            " %d",it->layerID);
            source = concat_and_free_old(source,temp);
            it = (dR_Node*)dR_list_next(current_node->next_layers);
        }

        temp = g_strdup_printf(
        "\nvariable_files:");
        source = concat_and_free_old(source,temp);
        for(i=0;i<num_variables;i++)
        {
            gchar* filename = g_strdup_printf("%d.var",global_variable_counter);
            temp = g_strdup_printf(
            " %s",filename);
            source = concat_and_free_old(source,temp);
            // Save variable file
            filePath = g_build_filename(folderPath,filename, NULL);
            fp = fopen(filePath,"wb");
            if( fp == NULL )
            {
               g_print("Error while creating the variable file %s.\n",filename);
               return 0;
            }
            fwrite(variables[i],sizeof(gfloat),variable_sizes[i],fp);
            fclose(fp);
            global_variable_counter++;
            g_free(filePath);
        }

        num_variables = 2;
        num_params = 10;
        current_node = (dR_Node*)dR_list_next(net->allNodes);
    }
    temp = g_strdup_printf(
    "\n");
    source = concat_and_free_old(source,temp);


    length = strlen(source);
    filePath = g_build_filename(folderPath,"graph.dr", NULL);
    g_file_set_contents(filePath,source,length,NULL);
    g_free(source);
    g_free(filePath);
    g_free(folderPath);
}

dR_Node* dR_parseGraph(dR_Graph* net, gchar* path, dR_Node*** outnodes, gint* numnodes, dR_Node*** outfeednodes, gint* numfeednodes)
{
    gsize length;
    gchar* filecontent;
    gchar* filepath;
    gint i;
    gint feednode_it=0;
    dR_Node** graphnodes;
    dR_Node** feednodes;
    gboolean success;
    gchar* lineptr;
    gchar* inlineptr;
    gchar* line;
    gchar* token;

    gint number_of_nodes;
    gint number_of_feednodes;
    GError *err = NULL;

    if(net->number_of_layers!=0)
    {
         g_print("dR_parseGraph can only fill an empty graph! Given graph not empty!\n");
         return 0;
    }
    net->config->modelPath = path;

    filepath = g_build_filename(path, "graph.dr", NULL);

    if(!net->config->silent)
        g_print("Loading Model File %s ...\n",path);

    success = g_file_get_contents(filepath,&filecontent,&length,&err);

    if(!success)
    {
        g_print("Error while opening the graph file.\n");
        return 0;
    }

    line = own_strtok_r(filecontent,"\n",&lineptr);

    // Get number of nodes and feed nodes;
    own_strtok_r(line," ",&inlineptr);
    own_strtok_r(NULL," ",&inlineptr);
    number_of_nodes = atoi(own_strtok_r(NULL," ",&inlineptr));
    own_strtok_r(NULL," ",&inlineptr);
    number_of_feednodes = atoi(own_strtok_r(NULL," ",&inlineptr));

    *outnodes = g_malloc(number_of_nodes*sizeof(dR_Node*));
    *outfeednodes = g_malloc(number_of_feednodes*sizeof(dR_Node*));
    *numnodes = number_of_nodes;
    *numfeednodes = number_of_feednodes;
    graphnodes = *outnodes;
    feednodes = *outfeednodes;

    if(!net->config->silent)
        g_print("Found graph with %d nodes and %d feed nodes.\n",number_of_nodes,number_of_feednodes);

    // Build graph
    for(i=0;i<number_of_nodes;i++)
    {
        gchar* desc;
        // Currently no node type has more than 10 parameters and 2 variables
        gchar* params[10];
        gfloat* variables[2];
        gchar* variableFilenames[2];
        gint num_variables;
        gint num_params;
        gint id;
        gint it = 0;
        // Hard cap 256 of previous nodes...
        gint prev_nodes[256];
        gint num_prev_nodes;
        dR_Node** input;
        dR_Node* node;

        line = own_strtok_r(NULL,"\r\n",&lineptr);

        own_strtok_r(line, " ",&inlineptr);
        id = atoi(own_strtok_r(NULL, " ",&inlineptr));
        desc = own_strtok_r(NULL, " ",&inlineptr);

        if(!net->config->silent&&net->config->debugInfo)
            g_print("Loading Node %d\n",id);

        // Get params
        line = own_strtok_r(NULL,"\r\n",&lineptr);

        token = own_strtok_r(line, " ",&inlineptr);
        token = own_strtok_r(NULL, " ",&inlineptr);
        while(token != NULL) {
            params[it] = token;
            token = own_strtok_r(NULL, " ",&inlineptr);
            it++;
        }
        num_params = it;

        // Get prev_nodes
        line = own_strtok_r(NULL,"\r\n",&lineptr);

        token = own_strtok_r(line, " ",&inlineptr);
        token = own_strtok_r(NULL, " ",&inlineptr);
        it = 0;
        while(token != NULL) {
            prev_nodes[it] = atoi(token);
            token = own_strtok_r(NULL, " ",&inlineptr);
            it++;
        }
        num_prev_nodes = it;

        // Ignore next_node line
        line = own_strtok_r(NULL,"\r\n",&lineptr);

        // Get variables
        line = own_strtok_r(NULL,"\r\n",&lineptr);

        token = own_strtok_r(line, " ",&inlineptr);
        token = own_strtok_r(NULL, " ",&inlineptr);
        it = 0;
        while(token != NULL) {
            variableFilenames[it] = token;
            token = own_strtok_r(NULL, " ",&inlineptr);
            it++;
        }
        num_variables = it;

        if(!net->config->silent&&net->config->debugInfo)
            g_print("Found %s with id %d, %d params and %d variables\n",desc,id,num_params,num_variables);
        // Load variables
        for(it=0;it<num_variables;it++)
        {
            gsize length;
            gchar* filecontent;
            gchar* filepath;
            filepath = g_build_filename(path,variableFilenames[it], NULL); 
            success = g_file_get_contents(filepath,&filecontent,&length,&err);
            variables[it] = (gfloat*)filecontent;
            if(!success)
            {
                g_print("Error while opening variable file: %s, Length: %d, Errcode: %d\n", filepath, length, *err);
                return 0;
            }
            g_free(filepath);
        }

        if(num_prev_nodes>0)
        {
            input = g_malloc(num_prev_nodes*sizeof(dR_Node*));
            for(it=0;it<num_prev_nodes;it++)
            {
                input[it] = graphnodes[prev_nodes[it]];
            }
        }

        if(strncmp(desc, "Conv2D", strlen("Conv2D"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse Conv2d node!\n");
                return 0;
            }
            node = dR_conv2d_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "Conv2DTranspose", strlen("Conv2DTranspose"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse Conv2DTranspose node!\n");
                return 0;
            }
            node = dR_conv2dtranspose_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "Pooling", strlen("Pooling"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse Pooling node!\n");
                return 0;
            }
            node = dR_pooling_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "FullyConnected", strlen("FullyConnected"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse FullyConnected node!\n");
                return 0;
            }
            node = dR_fc_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "PPFilter", strlen("PPFilter"))==0)
        {
            if(num_prev_nodes!=2)
            {
                g_print("Failed to parse PPFilter node!\n");
                return 0;
            }
            node = dR_conv2duw_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "MaskDependentFilter", strlen("MaskDependentFilter"))==0)
        {
            if(num_prev_nodes!=2)
            {
                g_print("Failed to parse MaskDependentFilter node!\n");
                return 0;
            }
            node = dR_cdfilter_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "LabelCreation", strlen("LabelCreation"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse LabelCreation node!\n");
                return 0;
            }
            node = dR_lc_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "Normalization", strlen("Normalization"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse Normalization node!\n");
                return 0;
            }
            node = dR_norm_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "Upscaling", strlen("Upscaling"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse Upscaling node!\n");
                return 0;
            }
            node = dR_us_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "RGB2Gray", strlen("RGB2Gray"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse RGB2Gray node!\n");
                return 0;
            }
            node = dR_rgb2gray_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "ResolveRoI", strlen("ResolveRoI"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse ResolveRoI node!\n");
                return 0;
            }
            node = dR_resolveRoI_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "Softmax", strlen("Softmax"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse Softmax node!\n");
                return 0;
            }
            node = dR_softmax_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "Slice", strlen("Slice"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse Slice node!\n");
                return 0;
            }
            node = dR_slice_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "Concat", strlen("Concat"))==0)
        {
            if(num_prev_nodes<=1)
            {
                g_print("Failed to parse Concat node!\n");
                return 0;
            }
            node = dR_concat_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "CropOrPad", strlen("CropOrPad"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse Conv2d node!\n");
                return 0;
            }
            node = dR_croporpad_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "ElemWise2Op", strlen("ElemWise2Op"))==0)
        {
            if(num_prev_nodes!=2)
            {
                g_print("Failed to parse ElemWise2Op node!\n");
                return 0;
            }
            node = dR_elemwise2op_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "ElemWise1Op", strlen("ElemWise1Op"))==0)
        {
            if(num_prev_nodes!=1)
            {
                g_print("Failed to parse Conv2d node!\n");
                return 0;
            }
            node = dR_elemwise1op_parseAppendNode(net,input,num_prev_nodes,params,num_params,variables,num_variables);
        }
        else if(strncmp(desc, "DataFeedNode", strlen("DataFeedNode"))==0)
        {
            if(num_prev_nodes!=0)
            {
                g_print("Failed to parse DataFeedNode node!\n");
                return 0;
            }
            node = dR_datafeednode_parseAppendNode(net,NULL,num_prev_nodes,params,num_params,variables,num_variables);
            feednodes[feednode_it] = node;
            feednode_it++;
        }
        else
        {
            g_print("Parsing Error: Node could not be identified: %s!\n", desc);
            return NULL;
        }
        if(node==NULL)
        {
            g_print("Parsing Error: Node could not be appended!\n");
            return NULL;
        }
        graphnodes[id] = node;
        for(it=0;it<num_variables;it++)
        {
            g_free(variables[it]);
        }
        if(num_prev_nodes>0)
        {
            g_free(input);
        }
    }
    g_free(filecontent);
    g_free(filepath);
    return graphnodes[number_of_nodes-1];
}


// Deprecated
dR_Node* dR_parseModel(dR_Graph* net, dR_Node* input, gchar* path, dR_Node** nodelist, gint* numnodes)
{
    char netfilepath[256];
    FILE *fp;
    char line1[256];
    char line2[256];
    gint i = 0;
    dR_Node* last_layer = input;
    gint maxnodes = 0;
    g_print("Loading Model File %s ...\n",path);
    if(numnodes != NULL)
    {
        maxnodes = *numnodes;
    }
    g_snprintf(netfilepath,256,"%s%s",path,"net");
    fp = fopen(netfilepath,"r");
    if( fp == NULL )
    {
       g_print("Error while opening the model file.\n");
       return 0;
    }
    net->config->modelPath=g_strdup(path);
    fgets(line1, sizeof(line1), fp);
    fgets(line2, sizeof(line2), fp);
    while(!feof(fp)){
        char layertype[20];
        char *p;
        float params[20];
        int variableFiles[5];
        int numberoffiles = 0;
        int numberofparams = 0;

        p = strtok(line1, " ");
        strcpy(layertype,p);
        while(p != NULL) {
            p = strtok(NULL, " ");
            if(p != NULL)
            {
                variableFiles[numberoffiles] = atoi(p);
                numberoffiles++;
            }
        }

        p = strtok(line2, " ");
        params[0] = atof(p);
        while(p != NULL) {
            numberofparams++;
            p = strtok(NULL, " ");
            if(p != NULL)
            {
                params[numberofparams]= atof(p);
            }
        }
        numberofparams--;
        if(strncmp(layertype, "Conv2D", 6)==0)
        {
            dR_Shape4 shape;
            dR_Shape4 stride;
			gfloat* biases;
			gfloat* weights;
            char weightpath[256];
            dR_ActivationType type;

            switch((int)params[0]){
            case 0:
                type=tLinear;
                break;
            case 1:
                type=tReLU;
                break;
            case 2:
                type=tSigmoid;
                break;
            case 3:
                type=tTan;
                break;
            }
            shape.s0 = (gint32)params[2];
            shape.s1 = (gint32)params[3];
            shape.s2 = (gint32)params[4];
            shape.s3 = (gint32)params[5];
            stride.s0 = (gint32)params[6];
            stride.s1 = (gint32)params[7];
            stride.s2 = (gint32)params[8];
            stride.s3 = (gint32)params[9];
            last_layer = dR_Conv2d(net,last_layer,shape,stride,type,(int)params[1]);
            sprintf(weightpath,"%s%d",path,variableFiles[0]);
            weights = loadVariables(weightpath,shape.s0*shape.s1*shape.s2*shape.s3);
            if((int)params[1])
            {
                char biaspath[256];
                sprintf(biaspath,"%s%d",path,variableFiles[1]);
                biases = loadVariables(biaspath,shape.s3);
            }
            dR_Conv2d_setVariables(last_layer, weights, biases);
            g_free(weights);
            if((int)params[1])
            {
                g_free(biases);
            }
        }
        else if(strncmp(layertype, "Pooling", 7)==0)
        {
            dR_PoolingType type;
            dR_Shape4 shape;
            dR_Shape4 stride;
            switch((int)params[0]){
            case 0:
                type=tMax;
                break;
            case 1:
                type=tAverage;
                break;
            case 2:
                type=tl2norm;
                break;
            }
            shape.s0 = (gint32)params[1];
            shape.s1 = (gint32)params[2];
            shape.s2 = (gint32)params[3];
            shape.s3 = (gint32)params[4];
            stride.s0 = (gint32)params[5];
            stride.s1 = (gint32)params[6];
            stride.s2 = (gint32)params[7];
            stride.s3 = (gint32)params[8];
            last_layer = dR_Pooling(net, last_layer, shape, stride, type);
        }
        else if(strncmp(layertype, "FullyConnected", 18)==0)
        {
            dR_Shape2 shape;
            gfloat* biases;
			gfloat* weights;
            dR_ActivationType type;
            char weightpath[256];

            switch((int)params[0]){
            case 0:
                type=tLinear;
                break;
            case 1:
                type=tReLU;
                break;
            case 2:
                type=tSigmoid;
                break;
            case 3:
                type=tTan;
                break;
            }
            shape.s0 = (gint32)params[2];
            shape.s1 = (gint32)params[3];
            last_layer = dR_FullyConnected(net,last_layer,shape,type,(int)params[1]);
            sprintf(weightpath,"%s%d",path,variableFiles[0]);
            weights = loadVariables(weightpath,shape.s0*shape.s1);
            if((int)params[1])
            {
                char biaspath[256];
                sprintf(biaspath,"%s%d",path,variableFiles[1]);
                biases = loadVariables(biaspath,shape.s1);
            }
            dR_FullyConnected_setVariables(last_layer, weights, biases);
            g_free(weights);
            if((int)params[1])
            {
                g_free(biases);
            }
        }
        else
        {
            g_print("Can not parse Model File: Not Supported Layer in Model File?\n");
            return 0;
        }
        if(nodelist!=NULL&&i<maxnodes)
            nodelist[i]=last_layer;
        i++;
        fgets(line1, sizeof(line1), fp);
        fgets(line2, sizeof(line2), fp);
    }
    if(numnodes!=NULL)
    {
        *numnodes = i;
    }
    fclose(fp);
    return last_layer;
}

void dR_printSchedule(dR_Graph* net)
{
    gint i = 0;
    dR_Node* current_layer;
    dR_list_resetIt(net->scheduledLayers);
    current_layer = dR_list_next(net->scheduledLayers);
    g_print("CNN Model: %d Scheduled Layers\n\n",net->scheduledLayers->length);

    while(current_layer){
        i++;
        g_print("%d. Layer %d\n",i,current_layer->layerID);
        current_layer = dR_list_next(net->scheduledLayers);
    }
    g_print("\n\n");
}

void dR_printLayer(dR_Node* layer, gchar* path, FILE* fp)
{
    gchar *out, *tempstr;
    dR_Node* temp;
    if(layer->printLayer)
    {
        out = layer->printLayer(layer);
        tempstr = g_strdup_printf(" Previous Layers: ");
        out = concat_and_free_old(out,tempstr);
        dR_list_resetIt(layer->previous_layers);
        temp = dR_list_next(layer->previous_layers);
        if(layer->previous_layers->length==0)
        {
            tempstr = g_strdup_printf("none");
            out = concat_and_free_old(out,tempstr);
        }
        while(temp)
        {
            tempstr = g_strdup_printf("%d, ", temp->layerID);
            out = concat_and_free_old(out,tempstr);
            temp = dR_list_next(layer->previous_layers);
        }

        tempstr = g_strdup_printf("\n Next Layers: ");
        out = concat_and_free_old(out,tempstr);
        dR_list_resetIt(layer->next_layers);
        temp = dR_list_next(layer->next_layers);
        if(layer->next_layers->length==0)
        {
            tempstr = g_strdup_printf("none");
            out = concat_and_free_old(out,tempstr);
        }
        while(temp)
        {
            tempstr = g_strdup_printf("%d, ", temp->layerID);
            out = concat_and_free_old(out,tempstr);
            temp = dR_list_next(layer->next_layers);
        }
        tempstr = g_strdup_printf("\n\n");
        out = concat_and_free_old(out,tempstr);
    }
    else
    {
        out = g_strdup_printf("Layer with ID %d, no print function available! \n\n",layer->layerID);
    }
    if(path)
    {
        fputs(out,fp);
    }
    else
    {
        g_print("%s",out);
    }
    g_free(out);
}

void dR_printNet(dR_Graph* net, char* path)
{
    dR_Node* current_layer;
    FILE *fp;
    char init[100];
    if(net->prepared)
    {
        g_print("Net needs to be prepared before printing!\n");
        return;

    }
    dR_list_resetIt(net->allNodes);
    current_layer = dR_list_next(net->allNodes);
    if(path)
    {
        fp = fopen(path,"w");
        if(fp == NULL)
        {
           g_print("Error while opening the output file.\n");
           return;
        }
    }
    sprintf(init,"\n---------------------------------\ndeepRACIN graph model: %d Layers\n\n",net->allNodes->length);
    if(path)
    {
        fputs(init,fp);
    }
    else
    {
        g_print("%s",init);
    }
    while(current_layer){
        dR_printLayer(current_layer,path, fp);
        current_layer = dR_list_next(net->allNodes);
    }
    if(path)
        fclose(fp);
    g_print("---------------------------------\n\n");
}


