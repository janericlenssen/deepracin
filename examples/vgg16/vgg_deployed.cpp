#include <iostream>
#include <dirent.h>
#include <deepRACIN.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(void)
{
    //Get Image dimensions
    int width = 224;
    int height = 224;

    // Required Pointers and Variables
    dR_Graph* net;
    dR_Node** feedlist;
    dR_Node** nodeslist;
    gint numnodes;
    gint numfeeds;
    dR_Node* lastnode;

    // New deepracin graph
    net = dR_NewGraph();

    // Configure deepracin
    // Do not display debug information
    net->config->debugInfo = FALSE;
    // Profile GPU runtime
    net->config->profilingGPU = TRUE;
    // Profile CPU runtime
    net->config->profilingCPU = TRUE;
    // Give command line output
    net->config->silent = FALSE;
    // No Platform Hint - Choose first GPU available
    net->clConfig->clPlatformName = NULL;
    // Initialize OpenCL
    dR_initCL(net);

    // Load dR Model and get list of nodes and data feed nodes
    dR_loadGraph(net,"../deepRacinModels/vgg16_whole/",&nodeslist,&numnodes,&feedlist,&numfeeds);
    lastnode = nodeslist[numnodes-1];

    // Mark the output nodes whose output buffers can be fetched with dR_getOutputBuffers
    // Nodes marked as output are guaranteed to have consistent output buffers after using dR_apply()
    dR_setAsOutput(net,lastnode);

    // Print network to console
    dR_printNetObject(net, NULL);

    // Prepare network for execution - scheduling, buffer creation, kernel creation is done here
    dR_prepare(net);


    cl_float* odata = (gfloat*)g_malloc(1000*sizeof(gfloat));
    cl_float* debug = (gfloat*)g_malloc(width*height*3*sizeof(gfloat));
    cl_mem* outbuffers[1];

    // Fetch pointers to output buffers
    dR_getOutputBuffers(net,outbuffers);

    for(int i=0;i<10;i++)
    {
        // Read and prepare image using OpenCV (does not have to be OpenCV)
        Mat image;
        if(i%2==0)
        {
            image = imread("tiger.png", CV_LOAD_IMAGE_COLOR);
        }
        else
        {
            image = imread("puzzle.png", CV_LOAD_IMAGE_COLOR);
        }
        if(!image.data)
        {
            g_print("Example test image not found! Please move tiger.png and puzzle.png to examples folder!\n");
            break;
        }
        image.convertTo(image,CV_32FC3);
        Mat channelarray[3];
        cv::split(image,channelarray);

        gint buffersize = image.total()*image.elemSize()/sizeof(cl_float);

        // Feed data to graph
        // We expect [c,h,w] order in deepracin but OpenCV Mat.data has [h,w,c], therefore the split and swizzle
        dR_feedData(net,feedlist[0],(cl_float*)channelarray[0].data,0,buffersize/3);
        dR_feedData(net,feedlist[0],(cl_float*)channelarray[1].data,buffersize/3,buffersize/3);
        dR_feedData(net,feedlist[0],(cl_float*)channelarray[2].data,buffersize*2/3,buffersize/3);

        // Apply the graph
        dR_apply(net);

        // Download content from previously declared and fetched output buffer (Softmax output)
        dR_downloadArray(net,"", outbuffers[0],0,1000*sizeof(cl_float),odata);

        // Parse Classfile and print class
        float max = odata[0];
        int argmax = 0;
        gchar classfilename[] = "synset.txt";
        FILE *classfile = fopen(classfilename, "r");
        gint count = 0;
        for(int j=0;j<1000;j++)
        {
            if(odata[j]>max)
            {
                max = odata[j];
                argmax = j;
            }
        }

        g_print("\n");
        if ( classfile != NULL )
        {
            gchar line[256];
            while (fgets(line, sizeof line, classfile) != NULL)
            {
                if (count == argmax)
                {
                    break;
                }
                else
                {
                    count++;
                }
            }
            fclose(classfile);
            g_print("Classification Result: Class %d, %s\n",argmax, line);
        }
        else
        {
            g_print("Failed to read class name file! \n", line);
        }

        // Show image
        image.convertTo(image,CV_32FC3,1/255.0);
        imshow("Display Image", image);
        g_print("Press key to continue...\n");
        waitKey(0);
    }

    dR_cleanup(net, TRUE);

    return 0;
}



