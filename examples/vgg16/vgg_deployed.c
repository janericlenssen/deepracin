#include <deepRACIN.h>
#include <stdio.h>
#include <png.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

int x, y;

int width = 224;
int height = 224;
png_byte color_type;
png_byte bit_depth;

png_structp png_ptr;
png_infop info_ptr;
int number_of_passes;
png_bytep * row_pointers;


// Init libpng and image read buffers
void initialize_image_buffer()
{
    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y<height; y++)
            row_pointers[y] = (png_byte*) malloc(sizeof(png_bytep) * width * 4);
}

// Read PNG file with libpng
void read_image(char* path)
{
    char png_header[8];
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
    {
        printf("[initialize_image_reading] read struct failed");
        abort();
    }
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        printf("[initialize_image_reading] png_create_info_struct failed");
        abort();
    }
    png_set_sig_bytes(png_ptr, 8);
    FILE *fp = fopen(path, "rb");
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("[initialize_image_reading] init_io failed");
        abort();
    }
    png_init_io(png_ptr, fp);  
    if (!fp)
    {
        printf("[read_image] File %s could not be opened.", path);
        abort();
    }
    fread(png_header, 1, 8, fp);
    if (png_sig_cmp(png_header, 0, 8))
    {
        printf("[read_image] File %s is not a PNG file", path);
        abort();
    }


    png_read_info(png_ptr, info_ptr);

    color_type = png_get_color_type(png_ptr, info_ptr);
    bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    number_of_passes = png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("[read_png_file] Error during read_image");
        abort();
    }
    
    png_read_image(png_ptr, row_pointers);
    // strip the alpha channel if existing
    if (color_type & PNG_COLOR_MASK_ALPHA)
    {
        png_set_strip_alpha(png_ptr);
    }
    fclose(fp);
}

int main(void)
{
    int i,j;
    int buffersize = width*height*3*sizeof(float);
    float* feed_buffer = malloc(buffersize);

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
    initialize_image_buffer();
    for(i=0;i<10;i++)
    {
        // Read and prepare image using OpenCV (does not have to be OpenCV)
        if(i%2==0)
        {
            read_image("tiger.png");
        }
        else
        {
            read_image("puzzle.png");
        }

        // We expect [c,h,w] order in deepracin but the loaded data has order [h,w,c], therefore we swap axes and cast to float
        for(j = 0; j < width*height*3; j++)
        {
            feed_buffer[j] = (float)row_pointers[(j/width)%height][(j%width)*4+j/(width*height)];
        }

        // Feed data to graph
        dR_feedData(net,feedlist[0],(cl_float*)feed_buffer,0,buffersize/sizeof(float));

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
        for(j=0;j<1000;j++)
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
	    if(i%2==0)
            {
                g_print("Should be: Class 292, tiger, Panthera tigris.\n\n");
            }
            else
            {
                g_print("Should be: Class 611, jigsaw puzzle.\n\n");
            }
        }
        else
        {
            g_print("Failed to read class name file! \n\n");
        }

        // Show image
        g_print("Press Enter to continue...\n");
        getchar();
    }

    // Cleanup image read buffers
    for (y=0; y<height; y++)
        free(row_pointers[y]);
    free(row_pointers);
    free(feed_buffer);

    dR_cleanup(net, TRUE);

    return 0;
}






