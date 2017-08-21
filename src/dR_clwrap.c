#include "dR_clwrap.h"
#include "dR_core.h"
#include <stdio.h>
gboolean dR_clInit(dR_Graph* net)
{
    gboolean ret = FALSE;
    //gint previousLenght = 0;

    //for command queue:
    size_t nContextDescriptorSize = 0;
    size_t size = 0;
    //for device info:
    gchar temp[1024];
    cl_device_type deviceType = 0;
    cl_uint computeUnits = 0;
    cl_ulong memSize = 0;
    cl_bool imageSupport = FALSE;
    cl_bool endianLittle = FALSE;
    cl_uint frequency = 0;
    size_t workitemSize[3];
    cl_command_queue_properties queueProps = net->config->profilingGPU ? CL_QUEUE_PROFILING_ENABLE : 0;
    guint i;
    gchar* currentDir = NULL;
    const gchar * tempKernelPath;

    net->clConfig->clError = 0;

    if (!net->config->silent)
        g_print("Initializing OpenCL\n");

    if (!net->config->silent&&net->config->debugInfo)
        g_print("%s\n",net->clConfig->clKernelPath);

    //Make kernel path absolute
    if (g_path_is_absolute(net->clConfig->clKernelPath) == FALSE)
    {
        currentDir = g_get_current_dir();
        tempKernelPath = g_strdup(net->clConfig->clKernelPath);
        net->clConfig->clKernelPath = g_build_path(G_DIR_SEPARATOR_S, currentDir, tempKernelPath, NULL);
        if (tempKernelPath)
            g_free((gpointer)tempKernelPath);
        if (currentDir)
            g_free((gpointer)currentDir);
    }
    if (!net->config->silent&&net->config->debugInfo)
        g_print("%s\n",net->clConfig->clKernelPath);


    if (!net->config->silent)
        g_print ("Creating OpenCL context\n");

    //context
    ret = dR_clCreateCLContext(net);
    if (ret == FALSE)
    {
        g_print("Could not create the OpenCL context.");
        return FALSE;
    }

    if (!net->config->silent)
        g_print("Creating OpenCL command queue... \n");

    //query all devices available to the context
    net->clConfig->clError = clGetContextInfo(net->clConfig->clContext, CL_CONTEXT_DEVICES, 0, 0, &nContextDescriptorSize);
    if(dR_openCLError(net, "Could get the context info (nContextDescriptorSize).", "clInit"))
        return FALSE;

    net->clConfig->clError = clGetContextInfo(net->clConfig->clContext, CL_CONTEXT_DEVICES, nContextDescriptorSize, net->clConfig->clDeviceIds, 0);
    if(dR_openCLError(net, "Could get the context info (clDeviceIds).", "clInit"))
        return FALSE;

    net->clConfig->clNumDevices = nContextDescriptorSize / sizeof(cl_device_id);

    if (net->clConfig->clDeviceNumber >= net->clConfig->clNumDevices)
    {
        g_print("Choosen device id too big: %i instead of %i, using device id 0 instead.", net->clConfig->clDeviceNumber, net->clConfig->clNumDevices);
        net->clConfig->clDeviceNumber = 0;
    }

    //Save the selected device id
    net->clConfig->clDeviceId = net->clConfig->clDeviceIds[net->clConfig->clDeviceNumber];

    //Now that there is a valid context and device id check if OpenGL Sharing is available and update context if neccessary
    // clUpdateContextAndCreateBuffersWithGLInteroperation();

    /*if (net->clConfig->clCommandQueue != NULL)
    clReleaseCommandQueue(net->clConfig->clCommandQueue);*/

    //create a command queue for choosen device the context reported
    net->clConfig->clCommandQueue = clCreateCommandQueue(net->clConfig->clContext, net->clConfig->clDeviceId, queueProps, &net->clConfig->clError);

    if (dR_openCLError(net, "Could not create the command queue.", "clInit"))
        return FALSE;

    if (!net->config->silent)
        g_print("Done.\n");

    if (!net->config->silent)
        g_print("Found %i OpenCL devices(s) for %s:\n", net->clConfig->clNumDevices, (net->clConfig->clDeviceType == DEVICE_TYPE_DEFAULT) ? "DEVICE_TYPE_DEFAULT" : (net->clConfig->clDeviceType == DEVICE_TYPE_CPU) ? "DEVICE_TYPE_CPU" : (net->clConfig->clDeviceType == DEVICE_TYPE_GPU) ? "DEVICE_TYPE_GPU" : (net->clConfig->clDeviceType == DEVICE_TYPE_ACCELERATOR) ? "DEVICE_TYPE_ACCELERATOR" : "UNKNOWN!!!");

    for (i=0; i < net->clConfig->clNumDevices; i++)
    {
        clGetDeviceInfo(net->clConfig->clDeviceIds[i], CL_DEVICE_NAME, sizeof(temp), &temp, NULL);

        if (!net->config->silent)
            g_print("Device %i: %s", i, temp);
        if (i == net->clConfig->clDeviceNumber)
        {
            if (net->clConfig->clDeviceName)
                g_free((gpointer)net->clConfig->clDeviceName);
            net->clConfig->clDeviceName = g_strdup(temp);
            if (!net->config->silent)
            {
                g_print(" <-- selected \n");
            }
            break;
        }
        if (net->clConfig->clInfo)
            g_print("\n");
    }


    if (!net->config->silent)
        g_print ("Selected OpenCL Device: %s\n", net->clConfig->clDeviceName);

    //print device info:
    if (net->config->debugInfo && !net->config->silent)
    {
        g_print ("OpenCL device info:\n");

        g_print ("CL_DEVICE_NAME: \t\t\t %s\n", net->clConfig->clDeviceName);

        // CL_DEVICE_VENDOR
        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_VENDOR, sizeof(temp), &temp, NULL);
        g_print("CL_DEVICE_VENDOR: \t\t\t %s\n", temp);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL);
        g_print ("CL_DEVICE_TYPE: \t\t\t %s\n", deviceType==CL_DEVICE_TYPE_CPU ? "CPU":deviceType==CL_DEVICE_TYPE_GPU?"GPU" : deviceType==CL_DEVICE_TYPE_ACCELERATOR ? "Accelerator" : deviceType==CL_DEVICE_TYPE_DEFAULT ? "Default" : "");

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DRIVER_VERSION, sizeof(temp), &temp, NULL);
        g_print("CL_DRIVER_VERSION: \t\t\t %s\n", temp);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_VERSION, sizeof(temp), &temp, NULL);
        g_print("CL_DEVICE_VERSION: \t\t\t %s\n", temp);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_EXTENSIONS, sizeof(temp), &temp, NULL);
        g_print("CL_DEVICE_EXTENSIONS: \t\t\t %s\n", temp);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_ADDRESS_BITS, sizeof(size_t), &size, NULL);
        g_print ("CL_DEVICE_ADDRESS_BITS: \t\t %i\n", (gint) size);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &frequency, NULL);
        g_print ("CL_DEVICE_MAX_CLOCK_FREQUENCY: \t\t %u MHz \n", frequency);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &computeUnits, NULL);
        g_print ("CL_DEVICE_MAX_COMPUTE_UNITS: \t\t %u\n", computeUnits);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &size, NULL);
        g_print ("CL_DEVICE_MAX_PARAMETER_SIZE: \t\t %i\n", (gint) size);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(size_t), &size, NULL);
        g_print ("CL_DEVICE_MAX_CONSTANT_ARGS: \t\t %i\n", (gint) size);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(size_t), &size, NULL);
        g_print ("CL_DEVICE_MAX_READ_IMAGE_ARGS: \t\t %i\n", (gint) size);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_SAMPLERS, sizeof(size_t), &size, NULL);
        g_print ("CL_DEVICE_MAX_SAMPLERS: \t\t %i\n", (gint) size);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memSize, NULL);
        g_print ("CL_DEVICE_GLOBAL_MEM_SIZE: \t\t %lu MB \n", (long unsigned int) memSize/1024/1024);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &memSize, NULL);
        g_print ("CL_DEVICE_LOCAL_MEM_SIZE: \t\t %.lu KByte \n", (long unsigned int) memSize/1024);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &memSize, NULL);
        g_print ("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: \t %li KByte \n", (long int) memSize/1024);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memSize, NULL);
        g_print ("CL_DEVICE_MAX_MEM_ALLOC_SIZE: \t\t %li KByte \n", (long int) memSize/1024);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &memSize, NULL);
        g_print ("CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: \t %li KByte \n", (long int) memSize/1024);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitemSize), &workitemSize, NULL);
        g_print ("CL_DEVICE_MAX_WORK_ITEM_SIZES: \t\t %i | %i | %i\n", (gint) workitemSize[0], (gint) workitemSize[1], (gint) workitemSize[2]);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_uint), &size, NULL);
        g_print ("CL_DEVICE_MAX_WORK_GROUP_SIZE: \t\t %lu\n", (long unsigned int) size);

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(cl_bool), &imageSupport, NULL);
        g_print ("CL_DEVICE_ERROR_CORRECTION_SUPPORT: \t %s \n", (imageSupport==1?"YES":"NO"));

        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, NULL);
        g_print ("CL_DEVICE_IMAGE_SUPPORT: \t\t %s \n", (imageSupport==1?"YES":"NO"));

        if (imageSupport == 1)
        {
            clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_IMAGE2D_MAX_WIDTH , sizeof(size_t), &size, NULL);
            g_print ("CL_DEVICE_IMAGE2D_MAX_WIDTH: \t\t %i \n", (gint) size);
            clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &size, NULL);
            g_print ("CL_DEVICE_IMAGE2D_MAX_HEIGHT: \t\t %i \n", (gint) size);
        }

#ifndef GPGPU_SIM_MODE
        clGetDeviceInfo(net->clConfig->clDeviceId, CL_DEVICE_ENDIAN_LITTLE, sizeof(cl_bool), &endianLittle, NULL);
        g_print ("ENDIANNESS: \t\t\t\t %s \n", (endianLittle==1?"LITTLE-ENDIAN":"BIG_ENDIAN"));
#endif

        clGetDeviceInfo(net->clConfig->clDeviceId,CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(size_t), &size, NULL);
        g_print ("CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: \t %i\n", (gint) size);

        clGetDeviceInfo(net->clConfig->clDeviceId,CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(size_t), &size, NULL);
        g_print ("CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: %i\n", (gint) size);

        clGetDeviceInfo(net->clConfig->clDeviceId,CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(size_t), &size, NULL);
        g_print ("CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: \t %i\n", (gint) size);

        clGetDeviceInfo(net->clConfig->clDeviceId,CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(size_t), &size, NULL);
        g_print ("CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: \t %i\n", (gint) size);

        clGetDeviceInfo(net->clConfig->clDeviceId,CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(size_t), &size, NULL);
        g_print ("CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: %i\n", (gint) size);

        clGetDeviceInfo(net->clConfig->clDeviceId,CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(size_t), &size, NULL);
        g_print ("CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: %i\n", (gint) size);

        g_print ("\n");
    }

    if (!net->config->silent)
        g_print("Done.\n\n");

    return ret;
}


gboolean dR_clBuildWithSource(dR_Graph* net, const gchar * const filenames[], gint numberoffiles)
{
    gchar * sourceCodes[25] = { NULL };
    size_t sourceCodeLengths[25] = { 0 };
    cl_uint numberOfCodes = numberoffiles;
    cl_uint numberOfFiles = numberoffiles;
    gboolean kernelLoadSuccess;
    guint i;
    gchar * filePath = NULL;
    GError *err = NULL;
    gsize length;
#ifndef GPGPU_SIM_MODE
    size_t len;
#endif
    gint totalLength = 0;
    gboolean error = FALSE;
#ifdef GPGPU_SIM_MODE
    gchar * gpgpuSimCode;
    gsize destPos;
#endif
	
    dR_Node* current_layer;
	int numGeneratedCodes;
	gchar** genKernelFilenames;
    int sourceindex= 0;
    gboolean loadedKernel = FALSE;

    if (!net->config->silent)
        g_print("Loading kernels...\n");

    if (!net->config->silent&&net->config->debugInfo)
        g_print("from %s...\n", net->clConfig->clKernelPath);

    dR_list_resetIt(net->scheduledLayers);
    current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    while(current_layer){
        if(current_layer->createKernelName)
            numberOfCodes++;
        current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    }
    if (!net->config->silent&&net->config->debugInfo)
        g_print("Number of Kernelfiles: %d \n", numberOfCodes);
    numGeneratedCodes = numberOfCodes-numberOfFiles;
    genKernelFilenames = g_malloc(sizeof(gchar*)*numGeneratedCodes);
    dR_list_resetIt(net->scheduledLayers);
    current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
    //loading kernels form file
    //numberOfCodes=26;
    for(i=0; i < numberOfCodes; ++i)
    {
        loadedKernel = FALSE;
        if (i == 0 && net->clConfig->configH != NULL && net->clConfig->configH[0] != '\0')
        {
            //Use config.h from given path
            if (!net->config->silent)
            {
                g_print("Loading manually set config.h from %s...\n", net->clConfig->configH);
            }
            kernelLoadSuccess = g_file_get_contents(net->clConfig->configH, &(sourceCodes[sourceindex]), &length, &err);
            loadedKernel = TRUE;
            sourceindex++;
        }
        else
        {
            if(i<numberOfFiles)
            {
                //Load kernel file i
                filePath = g_build_filename(net->clConfig->clKernelPath, filenames[i], NULL);
                if (!net->config->silent&&net->config->debugInfo)
                {
                    g_print("Loading kernel file %s...\n", filePath);
                }
                kernelLoadSuccess = g_file_get_contents(filePath, &(sourceCodes[sourceindex]), &length, &err);
                g_free((gpointer)filePath);
                loadedKernel = TRUE;
                sourceindex++;
            }
            else
            {
                while(current_layer&&!current_layer->createKernelName)
                {
                    current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
                }
                if(current_layer&&current_layer->createKernelName)
                {
                    genKernelFilenames[sourceindex-numberOfFiles] = current_layer->createKernelName(current_layer);
                    filePath = g_build_filename(net->config->modelPath, genKernelFilenames[sourceindex-numberOfFiles], NULL);
                    if (!net->config->silent&&net->config->debugInfo)
                    {
                        g_print("Loading kernel file %s...\n", filePath);
                    }
                    kernelLoadSuccess = g_file_get_contents(filePath, &(sourceCodes[sourceindex]), &length, &err);
                    g_free((gpointer)filePath);
                    loadedKernel = TRUE;
                    if (!net->config->silent&&net->config->debugInfo)
                        g_print("Filename: %s\n", genKernelFilenames[sourceindex-numberOfFiles]);
                    sourceindex++;
                }
                current_layer = (dR_Node*)dR_list_next(net->scheduledLayers);
                //if (!net->config->silent&&net->config->debugInfo)
                //    g_print("Got generated kernel string with length: %d \n",(gint)length);
            }
        }
        if(loadedKernel)
        {
            if (!kernelLoadSuccess)
            {
                dR_openCLError(net, err->message, "clBuildWithSource");
                g_error_free(err);
                for(i=0; i < numberOfFiles; ++i)
                    if ( sourceCodes[i] ) g_free((gpointer)sourceCodes[i]);
                    else break;
                    return FALSE;
            }
            sourceCodeLengths[sourceindex-1] = length;
            totalLength += length;
            if (!net->config->silent&&net->config->debugInfo)
                g_print("Got generated kernel string with length: %d, total length: %d \n",(gint)length, (gint)totalLength);
        }
    }
    numberOfCodes = sourceindex;
#ifdef GPGPU_SIM_MODE
    //Copy all codes to one string, as GPGPU-Sim does not like more than one string
    gpgpuSimCode = g_malloc(totalLength + numberOfCodes + 1);

    destPos = 0;
    //copy all source codes to gpgpuSimCode and free them afterwards
    for(i=0; i < numberOfCodes; ++i)
    {
        //Copy code to gpgpuSimCode
        memcpy((void*)&(gpgpuSimCode[destPos]), (const void*)sourceCodes[i], sourceCodeLengths[i]);
        destPos += sourceCodeLengths[i];
        gpgpuSimCode[destPos] = '\n';
        destPos += 1;

        //Free old code
        if (sourceCodes[i])
        {
            g_free((gpointer)sourceCodes[i]);
        }
    }
    gpgpuSimCode[destPos] = '\0';
    sourceCodes[0] = gpgpuSimCode;
    sourceCodeLengths[0] = totalLength + numberOfCodes + 1;
    numberOfCodes = 1;
#endif



    if (net->config->debugInfo)
    {
        g_print("\n");
        for(i=0; i < numberOfCodes; ++i)
        {
            if(i<numberOfFiles)
                g_print("Kernel file %i:\n%s\n", i, filenames[i]);
            else
                g_print("Kernel file %i:\n%s\n", i, genKernelFilenames[i-numberOfFiles]);
        }
    }

    if (!net->config->silent)
    {
        g_print("Done.\n");
        g_print("Creating OpenCL program... \n");
        g_print("Source code length: %d \n",(gint)totalLength);
    }


    // create program
    net->clConfig->clProgram = clCreateProgramWithSource(net->clConfig->clContext,
        numberOfCodes,
        (const char **)sourceCodes,
        (const size_t *)sourceCodeLengths,
        &net->clConfig->clError);

    if (!net->config->silent)
        g_print("Done.\n");

    if (!net->config->silent)
        g_print("Cleanup loaded kernels...\n");
    g_free(genKernelFilenames);
    for(i=0; i < numberOfCodes; ++i)
    {
        if (sourceCodes[i])
        {
            g_free((gpointer)sourceCodes[i]);
        }
        else
        {
            break;
        }
    }

    if (dR_openCLError(net, "Could not create OpenCL program!", "clInit"))
    {
        return FALSE;
    }

    if (!net->config->silent)
        g_print("Done.\n");

    if (!net->config->silent)
        g_print("Building OpenCL program... ");


    //compile program
    if (g_strcmp0("NVIDIA Corporation", net->clConfig->clPlatformName) == 0)
    {
        //NVIDIA platform
        if (net->clConfig->fastRelaxedMath)
        {
#ifdef _DEBUG
            if (!net->config->silent&&net->config->debugInfo)
                g_print("clBuildProgram: NVIDIA Platform in debug mode with fast-relaxed-math\n");
            net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, "-w -cl-nv-verbose -cl-fast-relaxed-math -cl-mad-enable", NULL, NULL);
#else
            if (!net->config->silent&&net->config->debugInfo)
                g_print("clBuildProgram: NVIDIA Platform in release mode with fast-relaxed-math\n");
            net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL);
#endif
        }
        else
        {
#ifdef _DEBUG
            if (!net->config->silent&&net->config->debugInfo)
                g_print("clBuildProgram: NVIDIA Platform in debug mode without fast-relaxed-math\n");
            net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, "-w -cl-nv-verbose -cl-mad-enable", NULL, NULL);
#else
            if (!net->config->silent&&net->config->debugInfo)
                g_print("clBuildProgram: NVIDIA Platform in release mode without fast-relaxed-math\n");
            net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
#endif
        }
    }
    else
    {
        //Non-NVIDIA platform
        if (net->config->debugInfo && net->clConfig->clDeviceType == CL_DEVICE_TYPE_CPU)
        {
            //CPU
            if (net->clConfig->fastRelaxedMath)
            {
#ifdef _DEBUG
                if (!net->config->silent&&net->config->debugInfo)
                    g_print("clBuildProgram: CL_DEVICE_TYPE_CPU in debug mode with fast-relaxed-math\n");
                net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, "-cl-opt-disable -w -cl-fast-relaxed-math", NULL, NULL);
#else
                if (!net->config->silent&&net->config->debugInfo)
                    g_print("clBuildProgram: CL_DEVICE_TYPE_CPU in release mode with fast-relaxed-math\n");
                net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
#endif
            }
            else
            {
#ifdef _DEBUG
                if (!net->config->silent&&net->config->debugInfo)
                    g_print("clBuildProgram: CL_DEVICE_TYPE_CPU in debug mode without fast-relaxed-math\n");
                net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, "-cl-opt-disable -w", NULL, NULL);
#else
                if (!net->config->silent&&net->config->debugInfo)
                    g_print("clBuildProgram: CL_DEVICE_TYPE_CPU in release mode without fast-relaxed-math\n");
                net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, NULL, NULL, NULL);
#endif
            }
        }
        else
        {
            //Non-CPU
            if (net->clConfig->fastRelaxedMath)
            {
#ifdef _DEBUG
                if (!net->config->silent&&net->config->debugInfo)
                    g_print("clBuildProgram: Non-CPU in debug mode with fast-relaxed-math\n");
                net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, "-cl-opt-disable -w -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
#else
                if (!net->config->silent&&net->config->debugInfo)
                    g_print("clBuildProgram: Non-CPU in release mode with fast-relaxed-math\n");
                net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
#endif
            }
            else
            {
#ifdef _DEBUG
                if (!net->config->silent&&net->config->debugInfo)
                    g_print("clBuildProgram: Non-CPU in debug mode without fast-relaxed-math\n");
                net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, "-cl-opt-disable -w -cl-mad-enable", NULL, NULL);
#else
                if (!net->config->silent&&net->config->debugInfo)
                    g_print("clBuildProgram: Non-CPU in release mode without fast-relaxed-math\n");
                net->clConfig->clError = clBuildProgram(net->clConfig->clProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
#endif
            }
        }
    }

    error = dR_openCLError(net, "Could not build OpenCL program!", "clBuildWithSource");

    if (error == FALSE)
    {
        if (!net->config->silent)
            g_print("Done.\n");
    }
    else
    {
        g_print("Error while compiling the OpenCL program (%i).\n", error);
    }

    if (error == TRUE || net->config->debugInfo == TRUE)
    {
        //compile not successfully or just as debug info
#ifndef GPGPU_SIM_MODE
        net->clConfig->clError = clGetProgramBuildInfo(net->clConfig->clProgram, net->clConfig->clDeviceId, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &net->clConfig->build_status, NULL);
        if (! dR_openCLErrorWithoutCleanup(net,"Could not get the OpenCL build status.", "clBuildWithSource"))
        {
            if (net->clConfig->build_status != CL_BUILD_SUCCESS || net->config->debugInfo == TRUE)
            {
                net->clConfig->clError = clGetProgramBuildInfo(net->clConfig->clProgram, net->clConfig->clDeviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

                if (! dR_openCLErrorWithoutCleanup(net,"Could not get the OpenCL build log size.\n", "clBuildWithSource"))
                {
                    gchar * build_log_buffer = g_malloc(len+1);
                    net->clConfig->clError = clGetProgramBuildInfo(net->clConfig->clProgram, net->clConfig->clDeviceId, CL_PROGRAM_BUILD_LOG, len, build_log_buffer, NULL);
                    if (! dR_openCLErrorWithoutCleanup(net,"Could not get the OpenCL build log.\n", "clBuildWithSource"))
                    {
                        // to be carefully, terminate with \0
                        build_log_buffer[len] = '\0';
                        g_print("Build log: %s\n", build_log_buffer);
                        //GST_ELEMENT_WARNING(GST_ELEMENT(filter), STREAM, FAILED, ("Build log:\n%s", build_log_buffer), (NULL));
                    }
                    g_free((gpointer)build_log_buffer);
                }
            }
        }
#endif
        if (error == TRUE)
        {
            dR_cleanupCL(net);
            return FALSE;
        }
    } //compile successfully
    return TRUE;
}

void CL_CALLBACK dR_openCLErrorInfo(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    //Unused variables:
    (void)private_info;
    (void)cb;
    (void)user_data;

    g_print("OpenCL Error in detail: %s\n", errinfo);
}

gboolean dR_clCreateCLContext(dR_Graph* net)
{
    cl_uint numPlatforms = 0;
    cl_platform_id firstPlatformId = 0;
    cl_context_properties cps[3];
    cl_platform_id platforms[20];
    unsigned i = 0;
    char pbuf[100];
    net->clConfig->clPlatformId = NULL;
    net->clConfig->clError = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);

    if (dR_openCLError(net, "Failed to get the platform id count for the clContext. It failed on the first openCL call in clCreateCLContext. Make sure you have an OpenCL enabled driver installed for your hardware and that the driver is available for the program.", "clCreateCLContext"))
    {
        return FALSE;
    }

    if (numPlatforms <= 0)
    {
        g_print("OpenCL error: No OpenCL platform found. Make sure you have an OpenCL enabled driver installed for your hardware and that the driver is available for the program.");
        return FALSE;
    }

    if (numPlatforms > 20)
    {
        g_print("OpenCL error: Too many OpenCL platforms found.");
        return FALSE;
    }

    //platforms = new cl_platform_id[numPlatforms];
    net->clConfig->clError = clGetPlatformIDs(20, platforms, NULL);
    if (dR_openCLError(net, "Failed to get the platform id for the clContext.", "clCreateCLContext"))
    {
        return FALSE;
    }

    if (!net->config->silent)
        g_print("Found %i OpenCL platform(s):\n", numPlatforms);

    for (i = 0; i < numPlatforms; ++i)
    {
        net->clConfig->clError = clGetPlatformInfo(platforms[i],
            CL_PLATFORM_VENDOR,
            sizeof(pbuf),
            pbuf,
            NULL);
        if (dR_openCLError(net, "Failed to get the platform info for the clContext.", "clCreateCLContext"))
        {
            return FALSE;
        }
        if (!net->config->silent)
            g_print("Platform %i: %s", i, pbuf);

        if (!g_strcmp0(pbuf, net->clConfig->clPlatformName))
        {
            net->clConfig->clPlatformId = platforms[i];
            if (!net->config->silent)
            {
                g_print(" <-- selected");
            }
        }

        if (!net->config->silent)
            g_print("\n");
    } // for

    //If no matching platform was found try to match a substring
    if (numPlatforms > 0 && net->clConfig->clPlatformId == NULL)
    {
        GString * platformNameTemp = NULL;
        GString * substringPlatformNameTemp = NULL;
        GString * platformName = g_string_new(net->clConfig->clPlatformName);
        GString* substringPlatformName = NULL;

        if (platformName->len >= 5)
        {
            substringPlatformName = g_string_new_len (platformName->str, 5);
        }

        if (substringPlatformName != NULL)
        {
            if(!net->config->silent)
                g_print("Platform name %s not found, try to match substring %s.\n", platformName->str, substringPlatformName->str);

            for (i = 0; i < numPlatforms; ++i)
            {
                net->clConfig->clError = clGetPlatformInfo(platforms[i],
                    CL_PLATFORM_VENDOR,
                    sizeof(pbuf),
                    pbuf,
                    NULL);
                if (dR_openCLError(net, "Failed to get the platform info for the clContext.", "clCreateCLContext"))
                {
                    return FALSE;
                }

                if (platformNameTemp != NULL)
                {
                    g_string_free(platformNameTemp, TRUE);
                    platformNameTemp = NULL;
                }
                if (substringPlatformNameTemp != NULL)
                {
                    g_string_free(substringPlatformNameTemp, TRUE);
                    substringPlatformNameTemp = NULL;
                }

                platformNameTemp = g_string_new(pbuf);

                if (platformNameTemp->len >= 5)
                {
                    substringPlatformNameTemp = g_string_new_len (platformNameTemp->str, 5);
                }
                else
                {
                    substringPlatformNameTemp = g_string_new(pbuf);
                }

                if (!g_strcmp0(substringPlatformNameTemp->str, substringPlatformName->str))
                {
                    net->clConfig->clPlatformId = platforms[i];
                    if (!net->config->silent)
                    {
                        if(!net->config->silent)
                            g_print("Fallback to platform %i: %s\n", i, pbuf);
                        g_free(net->clConfig->clPlatformName);
                        net->clConfig->clPlatformName = g_strdup(pbuf); ;
                    }
                }
            } // for
        }

        //Cleanup strings
        if (platformName != NULL)
            g_string_free(platformName, TRUE);
        if (substringPlatformName != NULL)
            g_string_free(substringPlatformName, TRUE);
        if (platformNameTemp != NULL)
            g_string_free(platformNameTemp, TRUE);
        if (substringPlatformNameTemp != NULL)
            g_string_free(substringPlatformNameTemp, TRUE);
    }

    if (numPlatforms > 0 && net->clConfig->clPlatformId == NULL)
    {
        net->clConfig->clPlatformId = platforms[0];
        net->clConfig->clError = clGetPlatformInfo(platforms[0],
            CL_PLATFORM_VENDOR,
            sizeof(pbuf),
            pbuf,
            NULL);
        if (dR_openCLError(net, "Failed to get the platform info for the clContext.", "clCreateCLContext"))
        {
            g_print("Failed to get the platform info for the clContext.");
            return FALSE;
        }
		if(!net->config->silent)
		{
			g_print("Selected platform name not found: %s != %s\n", pbuf, net->clConfig->clPlatformName);
		    if (! net->clConfig->clPlatformName)
		        g_print("CL Platform name is empty!\n");
		    g_print("Could not find CL Platform: %s\n", net->clConfig->clPlatformName);
		    g_print("Fallback to CL Platform 0: %s\n", pbuf);
		}
        net->clConfig->clPlatformName = g_strdup(pbuf);

    }
    else if (numPlatforms <= 0)
    {
        g_print("No OpenCL platforms found for the selected OpenCL device type. Please make shure that there is an OpenCL Platform present in your computer and that the drivers are installed.");
        return FALSE;
    }
    if(net->clConfig->clPlatformId == NULL)
    {
        g_print("OpenCL Platform is NULL.");
        return FALSE;
    }

    if (net->clConfig->clContext != NULL)
    {
        clReleaseContext(net->clConfig->clContext);
        net->clConfig->clContext = NULL;
    }

    cps[0] = CL_CONTEXT_PLATFORM;
    cps[1] = (cl_context_properties)net->clConfig->clPlatformId;
    cps[2] = 0;


#ifndef GPGPU_SIM_MODE
    //Create context for the given device type
    if (net->config->debugInfo)
    {
        g_print("Creating context for device type: %s\n", net->clConfig->clDeviceType==CL_DEVICE_TYPE_CPU ? "CPU" : net->clConfig->clDeviceType==CL_DEVICE_TYPE_GPU?"GPU" : net->clConfig->clDeviceType==CL_DEVICE_TYPE_ACCELERATOR ? "Accelerator" : net->clConfig->clDeviceType==CL_DEVICE_TYPE_DEFAULT ? "Default" : "Unknown");
    }
    net->clConfig->clContext = clCreateContextFromType(cps, net->clConfig->clDeviceType, &dR_openCLErrorInfo, 0, &net->clConfig->clError);
#else
    g_print("clCreateContext...");
    net->clConfig->clError = clGetDeviceIDs(net->clConfig->clPlatformId, CL_DEVICE_TYPE_ALL, 20, net->clConfig->clDeviceIds, &net->clConfig->clNumDevices);
    net->clConfig->clContext = clCreateContext(cps, net->clConfig->clNumDevices, net->clConfig->clDeviceIds, 0, 0, &net->clConfig->clError);
    g_print("Done.\n");
#endif
    if (net->clConfig->clError != CL_SUCCESS)
    {
        g_print("Creation of clContext failed: %s\n",
            net->clConfig->clError == CL_DEVICE_NOT_AVAILABLE ? "CL_DEVICE_NOT_AVAILABLE (No devices that match device_type are currently available)" :
            net->clConfig->clError == CL_INVALID_VALUE ? "CL_INVALID_VALUE (Context property name in properties is not a supported property name, or if pfn_notify is NULL but user_data is not NULL. Or pfn_notify is NULL but user_data is not NULL)" :
            net->clConfig->clError == CL_INVALID_PLATFORM ? "CL_INVALID_PLATFORM (Properties is NULL and no platform could be selected or platform value specified in properties is not a valid platform)" :
            net->clConfig->clError == CL_DEVICE_NOT_FOUND ? "CL_DEVICE_NOT_FOUND (No devices that match device_type were found)" :
            net->clConfig->clError == CL_OUT_OF_HOST_MEMORY ? "CL_OUT_OF_HOST_MEMORY (A failure to allocate resources required by the OpenCL implementation on the host)" :
            net->clConfig->clError == CL_INVALID_DEVICE_TYPE ? "CL_INVALID_DEVICE_TYPE (device_type is not a valid value)" :
            "Unknown error."
            );
    }

    //Fallback: Try to get context by clCreateContext instead of clCreateContextFromType
    if (net->clConfig->clError != CL_SUCCESS || net->clConfig->clContext == NULL)
    {
        g_print("CL device could not be created!\n");
        g_print("Fallback to clCreateContext instead of clCreateContextFromType\n");

        net->clConfig->clError = clGetDeviceIDs(net->clConfig->clPlatformId, net->clConfig->clDeviceType, 20, net->clConfig->clDeviceIds, &net->clConfig->clNumDevices);

        if (net->clConfig->clError != CL_SUCCESS)
        {
            g_print("clGetDeviceIDs failed: %s\n",
                net->clConfig->clError == CL_INVALID_PLATFORM ? "CL_INVALID_PLATFORM (Platform is not a valid platform.)" :
                net->clConfig->clError == CL_INVALID_DEVICE_TYPE ? "CL_INVALID_DEVICE_TYPE (device_type is not a valid value. )" :
                net->clConfig->clError == CL_INVALID_VALUE ? "CL_INVALID_VALUE (num_entries is equal to zero and device_type is not NULL or if both num_devices and device_type are NULL.)" :
                net->clConfig->clError == CL_DEVICE_NOT_FOUND ? "CL_OUT_OF_HOST_MEMORY (No OpenCL devices that matched device_type were found.)" :
                "Unknown error."
                );
        }
        else
        {
            cps[0] = CL_CONTEXT_PLATFORM;
            cps[1] = (cl_context_properties)net->clConfig->clPlatformId;
            cps[2] = 0;
            net->clConfig->clContext = clCreateContext(cps, net->clConfig->clNumDevices, net->clConfig->clDeviceIds, &dR_openCLErrorInfo, NULL, &net->clConfig->clError);
            if (net->clConfig->clError != CL_SUCCESS)
            {
                g_print("Creation of clContext failed: %s\n",
                    net->clConfig->clError == CL_INVALID_PLATFORM ? "CL_INVALID_PLATFORM (Properties is NULL and no platform could be selected or if platform value specified in properties is not a valid platform. )" :
                    net->clConfig->clError == CL_DEVICE_NOT_AVAILABLE ? "CL_DEVICE_NOT_AVAILABLE (No devices that match device_type are currently available)" :
                    net->clConfig->clError == CL_INVALID_VALUE ? "CL_INVALID_VALUE (Context property name in properties is not a supported property name; if devices is NULL; if num_devices is equal to zero; or if pfn_notify is NULL but user_data is not NULL.)" :
                    net->clConfig->clError == CL_OUT_OF_HOST_MEMORY ? "CL_OUT_OF_HOST_MEMORY (A failure to allocate resources required by the OpenCL implementation on the host)" :
                    net->clConfig->clError == CL_INVALID_DEVICE ? "CL_INVALID_DEVICE (Devices contains an invalid device or are not associated with the specified platform. )" :
                    "Unknown error."
                    );
            }
        }
    }

    //Fallback: Try to get context for CL_DEVICE_TYPE_DEFAULT
    if (net->clConfig->clError != CL_SUCCESS || net->clConfig->clContext == NULL)
    {
        g_print("CL device could not be created!\n");
        g_print("Fallback to CL device type CL_DEVICE_TYPE_DEFAULT\n");
        net->clConfig->clDeviceType = CL_DEVICE_TYPE_DEFAULT;
#ifndef GPGPU_SIM_MODE
        net->clConfig->clContext = clCreateContextFromType(cps, net->clConfig->clDeviceType, &dR_openCLErrorInfo, 0, &net->clConfig->clError);
#else
        g_print("clCreateContext...");
        net->clConfig->clContext = clCreateContext(cps,1, 0, 0, 0, &net->clConfig->clError);
        g_print("Done.\n");
#endif
        if (dR_openCLError(net, "Failed to create a clContext with OpenCL error! Perhaps OpenCL device drivers not installed? Giving up.", "clCreateCLContext"))
            return FALSE;
        if (net->clConfig->clContext == NULL)
        {
            g_print("Failed to create a clContext, clContext is NULL! Perhaps OpenCL device drivers not installed? Giving up.\n");
            return FALSE;
        }
    }

    if (net->config->debugInfo)
    {
        g_print("Context created...\n");
    }

    return TRUE;
}

gboolean dR_openCLError(dR_Graph * net, char * message, char * callerName)
{
    if(G_LIKELY(net->clConfig->clError == CL_SUCCESS))
    {
        return FALSE;
    }
    dR_openCLErrorWithoutCleanup(net, message, callerName);
    dR_cleanupCL(net);
    return TRUE;
}


gboolean dR_createFloatBuffer(dR_Graph* net, cl_mem* mem, gint size, int buffertype)
{
    if(net->config->debugInfo&&!net->config->silent)
        g_print("Buffer Created: %d \n", (gint)(size*sizeof(gfloat)));
    *mem = clCreateBuffer(net->clConfig->clContext,
            buffertype/* | CL_MEM_ALLOC_HOST_PTR*/,
            size*sizeof(gfloat),
            NULL,
            &net->clConfig->clError);
    if (dR_openCLError(net, "Creation of float mem buffer failed.", "initMemory"))
            return FALSE;
    net->clConfig->memConsumptionGPU += size*sizeof(gfloat);
    return TRUE;
}


gboolean dR_clMemoryBufferCleanup(dR_Graph* net, cl_mem mem)
{
    gboolean ret = TRUE;

    if(mem) {
        net->clConfig->clError = clReleaseMemObject(mem);
        if(dR_openCLError(net, "Could not release Memory ", "clMemoryCleanup")){
            ret = FALSE;
        }
        mem = NULL;
    }
    return ret;
}

gboolean dR_createKernel(dR_Graph* net, char * kernelName, cl_kernel* kernel)
{
    gchar * message = NULL;
    gchar* currentDir = NULL;
    *kernel = clCreateKernel(net->clConfig->clProgram, kernelName, &net->clConfig->clError);

    if (net->clConfig->clError == CL_SUCCESS)
    {
        dR_kernelInfo(net, *kernel, kernelName);
        return TRUE;
    }
    currentDir = g_get_current_dir();
    message = g_strconcat("Could not create OpenCL kernel ", kernelName, " Make sure the kernel files in ", net->clConfig->clKernelPath, " are up to date. Perhaps deleting the .ptx file in ", currentDir, " helps.", NULL);
    dR_openCLError(net, message, "createKernel" );
    g_free((gpointer)currentDir);
    g_free((gpointer)message);
    return FALSE;
}

gboolean dR_cleanupKernel(cl_kernel kernel)
{
    if (kernel) {
        if (clReleaseKernel(kernel) != CL_SUCCESS)
            g_print("Could not release Kernel");
        kernel = NULL;
    }
    return TRUE;
}

gboolean dR_cleanupProgram(cl_program program)
{
    if (program) {
        if (clReleaseProgram(program) != CL_SUCCESS)
            g_print("Could not release Program");
        program = NULL;
    }
    return TRUE;
}

gboolean dR_finishCLKernel(dR_Graph* net, char * kernelName)
{
    long long queued = 0;
    long long start = 0;
    long long end = 0;
    double queueTime = 0;
    double total = 0;
    double numTexels = 0;
    double megaTexelsPerSecond = 0;
    dR_DataFeedNode_Data* feednode;
    dR_Node* start_layer;

    if (dR_openCLError(net, "Error yet before finishing kernel.", kernelName))
        return FALSE;

    if (net->config->profilingGPU)
    {
        //Wait for events
        net->clConfig->clError = clWaitForEvents(1, net->clConfig->clEvent);
        if(dR_openCLError(net, "clWaitForEvents failed", kernelName))
        {
            g_print("Error while waiting for event for kernel %s\n", kernelName);
            return FALSE;
        }
        // Wait for the commands to get serviced before reading back results
        net->clConfig->clError = clFinish(net->clConfig->clCommandQueue);
        if (dR_openCLError(net, "clFinish failed.", kernelName))
        {
            return FALSE;
        }

        net->clConfig->clError = clGetEventProfilingInfo(*net->clConfig->clEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, NULL);
        if (dR_openCLError(net, "clGetEventProfilingInfo(COMMAND_QUEUED) failed.", kernelName))
            start = 0;

        if (net->clConfig->clError != CL_SUCCESS)
        {
            g_print("Could not get queued time for kernel %s\n", kernelName);
            return FALSE;
        }

        net->clConfig->clError = clGetEventProfilingInfo(*net->clConfig->clEvent, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        if (dR_openCLError(net, "clGetEventProfilingInfo(COMMAND_START) failed.", kernelName))
            start = 0;

        if (net->clConfig->clError != CL_SUCCESS)
        {
            g_print("Could not get start time for kernel %s\n", kernelName);
            return FALSE;
        }

        net->clConfig->clError = clGetEventProfilingInfo(*net->clConfig->clEvent, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        if (dR_openCLError(net, "clGetEventProfilingInfo(COMMAND_END) failed.", kernelName))
            end = 0;

        if (net->clConfig->clError != CL_SUCCESS)
        {
            g_print("Could not get end time for kernel %s\n", kernelName);
            return FALSE;
        }

        queueTime = (double)(start - queued) / 1.0e6; //time between queued and actual start
        total = (double)(end - start) / 1.0e6; // Convert nanoseconds to msecs

        dR_list_resetIt(net->feed_layers);
        start_layer = dR_list_next(net->feed_layers);
        feednode = ((dR_DataFeedNode_Data*)(start_layer->layer));
        numTexels = feednode->shape.s0 * (double)feednode->shape.s1;
        megaTexelsPerSecond = 1.0e-6 * numTexels/(total/1000.0);

        if (queued == 0.0 || start == 0.0 || end == 0.0 || total < 0.0)
        {
            g_print("Error: Unexpected runtime. %s took: %2.3fms (+%2.2f)\nPerhaps OpenCL is out of resources or some kernel args are set wrong/missing!?\n", kernelName, total, queueTime);
        }

        if (total != 0)
            g_print("GPU Profiling: %s took: %2.3fms (+%2.2f, %2.0fMT/s)\n", kernelName, total, queueTime, megaTexelsPerSecond);
        net->clConfig->totalKernelGPUTime += total;
        net->clConfig->queueKernelGPUTime += queueTime;

        /*Effective Bandwidth = (Br + Bw)/T
        where:
        Br = total number of bytes read from global memory.
        Bw = total number of bytes written to global memory.
        T = time required to run kernel, specified in nanoseconds.
        If Br and Bw are specified in bytes, and T in ns, the resulting effective bandwidth
        is measured in GB/s, which is appropriate for current CPUs and GPUs for which
        the peak bandwidth range is 20-200 GB/s. Computing Br and Bw requires a
        thorough understanding of the kernel algorithm;*/
        if (clReleaseEvent(*net->clConfig->clEvent) != CL_SUCCESS)
        {
            g_print("Could not release clEvent");
            return FALSE;
        }
    }
    else
    {
        // Wait for the commands to get serviced before reading back results
        net->clConfig->clError = clFlush(net->clConfig->clCommandQueue);
        if (dR_openCLError(net, "clFlush failed", kernelName))
            return FALSE;
    }
    return TRUE;
}


void dR_kernelInfo(dR_Graph* net, cl_kernel kernel, char* kernelName)
{
    size_t size = 0;
    size_t compileWorkGroupSize[3] = {0,0,0};
    cl_ulong localMemSize = (cl_ulong)0.0;

    if (net->config->debugInfo && !net->config->silent)
    {
        g_print("\nInfo for kernel %s:\n", kernelName);
        if (CL_SUCCESS == clGetKernelWorkGroupInfo(kernel, net->clConfig->clDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size), (void *) &size, NULL))
            g_print("CL_KERNEL_WORK_GROUP_SIZE: %i\n",(gint) size);
        if (CL_SUCCESS == clGetKernelWorkGroupInfo(kernel, net->clConfig->clDeviceId, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(compileWorkGroupSize), (void *) &compileWorkGroupSize, NULL))
            g_print("CL_KERNEL_COMPILE_WORK_GROUP_SIZE: %i | %i | %i \n", (gint) compileWorkGroupSize[0], (gint) compileWorkGroupSize[1], (gint) compileWorkGroupSize[2]);
        if (CL_SUCCESS == clGetKernelWorkGroupInfo(kernel, net->clConfig->clDeviceId, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(localMemSize), (void *) &localMemSize, NULL))
            g_print("CL_KERNEL_LOCAL_MEM_SIZE: %i bytes\n", (gint)localMemSize);
    }
}

gboolean dR_uploadArray(dR_Graph* net, gchar* name, void * hostMem, size_t offset, size_t numBytes, cl_mem deviceMem)
{
    long long startGPU = 0;
    long long endGPU = 0;
    double totalGPU = 0;

    /*
    //Using pinned memory as shown on page 16 in http://www.nvidia.com/content/cudazone/CUDABrowser/downloads/papers/NVIDIA_OpenCL_BestPracticesGuide.pdf#page=16
    //In a testcase it took twice as long as not using pinned memory
    Make shure to use CL_MEM_ALLOC_HOST_PTR with net->clConfig->deviceMemGStreamer
    mappedMem = clEnqueueMapBuffer(net->clConfig->clCommandQueue, net->clConfig->deviceMemGStreamer, CL_TRUE, CL_MAP_WRITE, 0, net->clConfig->bufferDimensionsDeviceMemGStreamerPadded, 0, NULL, filter->clEvent, &filter->clError);
    if (dR_openCLError(net, "clEnqueueMapBuffer failed.", "gst_virus_detection_cl_chain"))
    return;
    memcpy(mappedMem, buf->data, buf->size);
    net->clConfig->clError = clEnqueueUnmapMemObject(net->clConfig->clCommandQueue, net->clConfig->deviceMemGStreamer, mappedMem, 0, NULL, net->clConfig->clEvent);
    if (dR_openCLError(net, "clEnqueueUnmapMemObject failed.", "gst_virus_detection_cl_chain"))
    return;
    dR_submitClKernel(net, "clEnqueueMemObject");
    */

    //Not using pinned memory
    net->clConfig->clError = clEnqueueWriteBuffer(net->clConfig->clCommandQueue, deviceMem, CL_TRUE, offset, numBytes, hostMem, 0, NULL, net->clConfig->clEvent);
    if (dR_openCLError(net,"Could not upload buffer to the OpenCL device.", "Initializing Buffer Content"))
    {
        g_print("Error: Upload of array %s failed\n", name);
        return FALSE;
    }

    if (net->config->profilingGPU)
    {
        net->clConfig->clError = clWaitForEvents(1, net->clConfig->clEvent);
        dR_openCLErrorWithoutCleanup(net, "clWaitForEvents failed.", "dR_uploadArray");
        g_print("Event here. \n");
        net->clConfig->clError = clGetEventProfilingInfo(*(net->clConfig->clEvent), CL_PROFILING_COMMAND_QUEUED,
            sizeof(startGPU), &startGPU, NULL);
        g_print("Got Prof Info 1. \n");
        if (dR_openCLErrorWithoutCleanup(net, "clGetEvent ProfilingInfo(COMMAND_START) failed.", "dR_uploadArray"))
        {
            startGPU = 0;
            return FALSE;
        }
        net->clConfig->clError = clGetEventProfilingInfo(*net->clConfig->clEvent, CL_PROFILING_COMMAND_END,
            sizeof(endGPU), &endGPU, NULL);

        g_print("Got Prof Info 2. \n");
        if (dR_openCLErrorWithoutCleanup(net, "clGetEventProfilingInfo(COMMAND_END) failed.", "dR_uploadArray"))
        {
            endGPU = 0;
            return FALSE;
        }
        totalGPU = (double)(endGPU - startGPU) / 1e6; // Convert nanoseconds to msecs
        if (totalGPU >= 0.000001)
        {
            g_print("\nProfiling: Upload time GPU for array %s: %5.2fms (%3.2fGB/s)\n", name, totalGPU, (((float)numBytes) / 1024 / 1024 / 1024) / (totalGPU/1000) );
        }
        if (clReleaseEvent(*net->clConfig->clEvent) != CL_SUCCESS)
        {
            g_print("Could not release clEvent");
            return FALSE;
        }

        g_print("Finished one Prof. \n");
    }
    return TRUE;
}

void dR_downloadArray(dR_Graph* net, gchar* name, cl_mem* deviceMem, size_t byteOffset, size_t numBytes, void * hostMem)
{
    long long startGPU = 0;
    long long endGPU = 0;
    double totalGPU = 0;


    if (hostMem == NULL)
    {
        g_print("Error: Memory %s is uninitialized.\n", name);
        return;
    }

    //Download array
    net->clConfig->clError = clEnqueueReadBuffer(net->clConfig->clCommandQueue, *deviceMem, CL_TRUE, byteOffset, numBytes, hostMem, 0, NULL /*event_wait_list*/, net->clConfig->clEvent);

    if (dR_openCLError(net, "Could not download host mem", name))
    {
        g_print("Debug info: downloadArray * net=%p, gchar* name=%s, cl_mem deviceMem=%p, size_t offset=%i, size_t numBytes=%i, void * hostMem=%p)\n", net, name, deviceMem, (gint)byteOffset, (gint)numBytes, hostMem);
        g_print("Error: Download of array %s failed\n", name);
        return;
    }

    if (net->config->profilingGPU && ! net->config->silent)
    {
        clWaitForEvents(1, net->clConfig->clEvent);
        net->clConfig->clError = clGetEventProfilingInfo(*net->clConfig->clEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(startGPU), &startGPU, NULL);

        if (dR_openCLErrorWithoutCleanup(net, "clGetEventProfilingInfo(COMMAND_START) failed.", "gst_virus_detection_cl_chain"))
            startGPU = 0;

        net->clConfig->clError = clGetEventProfilingInfo(*net->clConfig->clEvent, CL_PROFILING_COMMAND_END, sizeof(endGPU), &endGPU, NULL);
        if (dR_openCLErrorWithoutCleanup(net, "clGetEventProfilingInfo(COMMAND_END) failed.", "gst_virus_detection_cl_chain"))
            endGPU = 0;

        totalGPU = (double)(endGPU - startGPU) / 1e6; // Convert nanoseconds to msecs
        if (totalGPU >= 0.000001)
        {
            g_print("Transfer Profiling: Download time GPU for array %s: %5.2fms (%3.2fGB/s)\n", name, totalGPU, (((float)numBytes) / 1024 / 1024 / 1024) / (totalGPU/1000) );
        }
        if (clReleaseEvent(*net->clConfig->clEvent) != CL_SUCCESS)
        {
            g_print("Could not release clEvent");
        }
    }
}


gboolean dR_cleanupCL(dR_Graph* net)
{
    gboolean ret = TRUE;

    if (net->clConfig->cleanupDone)
        return ret;

    net->clConfig->cleanupDone = TRUE;

    if (!net->config->silent&&net->config->debugInfo)
    {
        g_print("CL Cleanup\n");
    }

    if (net->config->debugInfo)
    {
        g_print("Release: Events\n");
    }

    //cleanup clEvents
    if (net->clConfig->clEvent)
    {
        net->clConfig->clEvent = NULL;
    }

    if (net->config->debugInfo)
    {
        g_print("Release: Queue\n");
    }

    //cleanup clCommandQueue
    if (net->clConfig->clCommandQueue != (cl_command_queue) NULL)
    {
        if (clReleaseCommandQueue(net->clConfig->clCommandQueue) != CL_SUCCESS)
        {
            g_print("Could not release clCommandQueue");
            ret = FALSE;
        }
        net->clConfig->clCommandQueue = NULL;
    }

    if (net->config->debugInfo)
        g_print("Release: Context\n");


    //cleanup clContext
    if (net->clConfig->clContext != NULL)
    {
        if (clReleaseContext(net->clConfig->clContext) != CL_SUCCESS)
        {
            g_print("Could not release clContext");
            ret = FALSE;
        }
        net->clConfig->clContext = NULL;
    }

    net->clConfig->clPlatformId = 0;

    return ret;
}

gboolean dR_openCLErrorWithoutCleanup(dR_Graph* net, char * message, char * callerName)
{
    gchar * clError;
    if (net->clConfig->clError != CL_SUCCESS)
    {
        //Descriptions taken from http://streamcomputing.eu/blog/2013-04-28/opencl-1-2-error-codes/
        switch(net->clConfig->clError)
        {
        case CL_DEVICE_NOT_FOUND: clError = "CL_DEVICE_NOT_FOUND"; break;
        case CL_DEVICE_NOT_AVAILABLE: clError = "CL_DEVICE_NOT_AVAILABLE"; break;
        case CL_COMPILER_NOT_AVAILABLE: clError = "CL_COMPILER_NOT_AVAILABLE: Is raised if program is created with clCreateProgramWithSource and a compiler is not available i.e. CL_DEVICE_COMPILER_AVAILABLE specified in the table of OpenCL Device Queries for clGetDeviceInfo is set to CL_FALSE."; break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: clError = "CL_MEM_OBJECT_ALLOCATION_FAILURE: Is raised if there is a failure to allocate memory for buffer object."; break;
        case CL_OUT_OF_RESOURCES: clError = "CL_OUT_OF_RESOURCES: Is raised if there is a failure to allocate resources required by the OpenCL implementation on the device. It is also raised if a null pointer buffer is accessed, so check if all used buffers are initialized and args are passed to the kernel in the right order."; break;
        case CL_OUT_OF_HOST_MEMORY: clError = "CL_OUT_OF_HOST_MEMORY: Is raised if there is a failure to allocate resources required by the OpenCL implementation on the host."; break;
        case CL_PROFILING_INFO_NOT_AVAILABLE: clError = "CL_PROFILING_INFO_NOT_AVAILABLE: Is raised if the CL_QUEUE_PROFILING_ENABLE flag is not set for the command-queue and if the profiling information is currently not available (because the command identified by event has not completed)"; break;
        case CL_MEM_COPY_OVERLAP: clError = "CL_MEM_COPY_OVERLAP: Is raised if src_buffer and dst_buffer are the same buffer or subbuffer object and the source and destination regions overlap or if src_buffer and dst_buffer are different sub-buffers of the same associated buffer object and they overlap. The regions overlap if src_offset <= to dst_offset <= to src_offset + size – 1, or if dst_offset <= to src_offset <= to dst_offset + size – 1."; break;
        case CL_IMAGE_FORMAT_MISMATCH: clError = "CL_IMAGE_FORMAT_MISMATCH: Is raised if src_image and dst_image do not use the same image format."; break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: clError = "CL_IMAGE_FORMAT_NOT_SUPPORTED: Is raised if the image_format is not supported."; break;
        case CL_BUILD_PROGRAM_FAILURE: clError = "CL_BUILD_PROGRAM_FAILURE: Is raised if there is a failure to build the program executable. This error will be returned if clBuildProgram does not return until the build has completed."; break;
        case CL_MAP_FAILURE: clError = "CL_MAP_FAILURE: Is raised if there is a failure to map the requested region into the host address space. This error cannot occur for image objects created with CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR."; break;
        case CL_MISALIGNED_SUB_BUFFER_OFFSET: clError = "CL_MISALIGNED_SUB_BUFFER_OFFSET: Is raised if a sub-buffer object is specified as the value for an argument that is a buffer object and the offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue."; break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: clError = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: Is raised if the execution status of any of the events in event_list is a negative integer value."; break;
        case -15: clError = "CL_COMPILE_PROGRAM_FAILURE: Is raised if there is a failure to compile the program source. This error will be returned if clCompileProgram does not return until the compile has completed."; break;
        case -16: clError = "CL_LINKER_NOT_AVAILABLE: Is raised if a linker is not available i.e. CL_DEVICE_LINKER_AVAILABLE specified in the table of allowed values for param_name for clGetDeviceInfo is set to CL_FALSE."; break;
        case -17: clError = "CL_LINK_PROGRAM_FAILURE: Is raised if there is a failure to link the compiled binaries and/or libraries."; break;
        case -18: clError = "CL_DEVICE_PARTITION_FAILED: Is raised if the partition name is supported by the implementation but in_device could not be further partitioned."; break;
        case -19: clError = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE: Is raised if the argument information is not available for kernel."; break;
        case CL_INVALID_VALUE: clError = "CL_INVALID_VALUE: This depends on the function: two or more coupled parameters had errors."; break;
        case CL_INVALID_DEVICE_TYPE: clError = "CL_INVALID_DEVICE_TYPE: Is raised if an invalid device_type is given"; break;
        case CL_INVALID_PLATFORM: clError = "CL_INVALID_PLATFORM: Is raised if an invalid platform was given"; break;
        case CL_INVALID_DEVICE: clError = "CL_INVALID_DEVICE: Is raised if devices contains an invalid device or are not associated with the specified platform."; break;
        case CL_INVALID_CONTEXT: clError = "CL_INVALID_CONTEXT: Is raised if context is not a valid context."; break;
        case CL_INVALID_QUEUE_PROPERTIES: clError = "CL_INVALID_QUEUE_PROPERTIES: Is raised if specified command-queue-properties are valid but are not supported by the device."; break;
        case CL_INVALID_COMMAND_QUEUE: clError = "CL_INVALID_COMMAND_QUEUE: Is raised if command_queue is not a valid command-queue."; break;
        case CL_INVALID_HOST_PTR: clError = "CL_INVALID_HOST_PTR: This flag is valid only if host_ptr is not NULL. If specified, it indicates that the application wants the OpenCL implementation to allocate memory for the memory object and copy the data from memory referenced by host_ptr.CL_MEM_COPY_HOST_PTR and CL_MEM_USE_HOST_PTR are mutually exclusive.CL_MEM_COPY_HOST_PTR can be used with CL_MEM_ALLOC_HOST_PTR to initialize the contents of the cl_mem object allocated using host-accessible (e.g. PCIe) memory."; break;
        case CL_INVALID_MEM_OBJECT: clError = "CL_INVALID_MEM_OBJECT: Is raised if memobj is not a valid OpenCL memory object."; break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: clError = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: Is raised if the OpenGL/DirectX texture internal format does not map to a supported OpenCL image format."; break;
        case CL_INVALID_IMAGE_SIZE: clError = "CL_INVALID_IMAGE_SIZE: Is raised if an image object is specified as an argument value and the image dimensions (image width, height, specified or compute row and/or slice pitch) are not supported by device associated with queue."; break;
        case CL_INVALID_SAMPLER: clError = "CL_INVALID_SAMPLER: Is raised if sampler is not a valid sampler object."; break;
        case CL_INVALID_BINARY: clError = "CL_INVALID_BINARY: The provided binary is unfit for the selected device. If program is created with clCreateProgramWithBinary and devices listed in device_list do not have a valid program binary loaded."; break;
        case CL_INVALID_BUILD_OPTIONS: clError = "CL_INVALID_BUILD_OPTIONS: Is raised if the build options specified by options are invalid."; break;
        case CL_INVALID_PROGRAM: clError = "CL_INVALID_PROGRAM: Is raised if program is a not a valid program object."; break;
        case CL_INVALID_PROGRAM_EXECUTABLE: clError = "CL_INVALID_PROGRAM_EXECUTABLE: Is raised if there is no successfully built program executable available for device associated with command_queue."; break;
        case CL_INVALID_KERNEL_NAME: clError = "CL_INVALID_KERNEL_NAME: Is raised if kernel_name is not found in program."; break;
        case CL_INVALID_KERNEL_DEFINITION: clError = "CL_INVALID_KERNEL_DEFINITION: Is raised if the function definition for __kernel function given by kernel_name such as the number of arguments, the argument types are not the same for all devices for which the program executable has been built."; break;
        case CL_INVALID_KERNEL: clError = "CL_INVALID_KERNEL: Is raised if kernel is not a valid kernel object."; break;
        case CL_INVALID_ARG_INDEX: clError = "CL_INVALID_ARG_INDEX: Is raised if arg_index is not a valid argument index."; break;
        case CL_INVALID_ARG_VALUE: clError = "CL_INVALID_ARG_VALUE: Is raised if arg_value specified is not a valid value."; break;
        case CL_INVALID_ARG_SIZE: clError = "CL_INVALID_ARG_SIZE: Is raised if arg_size does not match the size of the data type for an argument that is not a memory object or if the argument is a memory object and arg_size != sizeof(cl_mem) or if arg_size is zero and the argument is declared with the __local qualifier or if the argument is a sampler and arg_size != sizeof(cl_sampler)."; break;
        case CL_INVALID_KERNEL_ARGS: clError = "CL_INVALID_KERNEL_ARGS: Is raised if the kernel argument values have not been specified."; break;
        case CL_INVALID_WORK_DIMENSION: clError = "CL_INVALID_WORK_DIMENSION: Is raised if work_dim is not a valid value (i.e. a value between 1 and 3)."; break;
        case CL_INVALID_WORK_GROUP_SIZE: clError = "CL_INVALID_WORK_GROUP_SIZE: Is raised if local_work_size is specified and number of work-items specified by global_work_size is not evenly divisable by size of work-group given by local_work_size or does not match the work-group size specified for kernel using the __attribute__ ((reqd_work_group_size(X, Y, Z))) qualifier in program source.if local_work_size is specified and the total number of work-items in the work-group computed as local_work_size[0] *… local_work_size[work_dim - 1] is greater than the value specified by CL_DEVICE_MAX_WORK_GROUP_SIZE in the table of OpenCL Device Queries for clGetDeviceInfo.if local_work_size is NULL and the __attribute__ ((reqd_work_group_size(X, Y, Z))) qualifier is used to declare the work-group size for kernel in the program source."; break;
        case CL_INVALID_WORK_ITEM_SIZE: clError = "CL_INVALID_WORK_ITEM_SIZE: Is raised if the number of work-items specified in any of local_work_size[0], … local_work_size[work_dim - 1] is greater than the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0], …. CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1]."; break;
        case CL_INVALID_GLOBAL_OFFSET: clError = "CL_INVALID_GLOBAL_OFFSET: Is raised if the value specified in global_work_size + the corresponding values in global_work_offset for any dimensions is greater than the sizeof(size_t) for the device on which the kernel execution will be enqueued."; break;
        case CL_INVALID_EVENT_WAIT_LIST: clError = "CL_INVALID_EVENT_WAIT_LIST: Is raised if event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0, or if event objects in event_wait_list are not valid events."; break;
        case CL_INVALID_EVENT: clError = "CL_INVALID_EVENT: Is raised if event objects specified in event_list are not valid event objects."; break;
        case CL_INVALID_OPERATION: clError = "CL_INVALID_OPERATION: Is raised if interoperability is specified by setting CL_CONTEXT_ADAPTER_D3D9_KHR, CL_CONTEXT_ADAPTER_D3D9EX_KHR or CL_CONTEXT_ADAPTER_DXVA_KHR to a non-NULL value, and interoperability with another graphics API is also specified. (only if the cl_khr_dx9_media_sharing extension is supported)."; break;
        case CL_INVALID_GL_OBJECT: clError = "CL_INVALID_GL_OBJECT: Is raised if texture is not a GL texture object whose type matches texture_target, if the specified miplevel of texture is not defined, or if the width or height of the specified miplevel is zero."; break;
        case CL_INVALID_BUFFER_SIZE: clError = "CL_INVALID_BUFFER_SIZE: Is raised if size is 0.Implementations may return CL_INVALID_BUFFER_SIZE if size is greater than the CL_DEVICE_MAX_MEM_ALLOC_SIZE value specified in the table of allowed values for param_name for clGetDeviceInfo for all devices in context."; break;
        case CL_INVALID_MIP_LEVEL: clError = "CL_INVALID_MIP_LEVEL: Is raised if miplevel is greater than zero and the OpenGL implementation does not support creating from non-zero mipmap levels."; break;
        case CL_INVALID_GLOBAL_WORK_SIZE: clError = "CL_INVALID_GLOBAL_WORK_SIZE: Is raised if global_work_size is NULL, or if any of the values specified in global_work_size[0], …global_work_size [work_dim - 1] are 0 or exceed the range given by the sizeof(size_t) for the device on which the kernel execution will be enqueued."; break;
        case CL_INVALID_PROPERTY: clError = "CL_INVALID_PROPERTY: Vague error, depends on the function"; break;
        case -65: clError = "CL_INVALID_IMAGE_DESCRIPTOR: Is raised if values specified in image_desc are not valid or if image_desc is NULL."; break;
        case -66: clError = "CL_INVALID_COMPILER_OPTIONS: Is raised if the compiler options specified by options are invalid."; break;
        case -67: clError = "CL_INVALID_LINKER_OPTIONS: Is raised if the linker options specified by options are invalid."; break;
        case -68: clError = "CL_INVALID_DEVICE_PARTITION_COUNT: Is raised if the partition name specified in properties is CL_DEVICE_PARTITION_BY_COUNTS and the number of sub-devices requested exceeds CL_DEVICE_PARTITION_MAX_SUB_DEVICES or the total number of compute units requested exceeds CL_DEVICE_PARTITION_MAX_COMPUTE_UNITS for in_device, or the number of compute units requested for one or more sub-devices is less than zero or the number of sub-devices requested exceeds CL_DEVICE_PARTITION_MAX_COMPUTE_UNITS for in_device."; break;
        case -1000: clError = "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR: CL and GL not on the same device (only when using a GPU)."; break;
        case -1001: clError = "CL_PLATFORM_NOT_FOUND_KHR: Is raised if in the clIcdGetPlatformIDsKHR() call num_entries is equal to zero and platforms is not NULL or if both num_platforms and platforms are NULL. "; break;
        default: clError = "Unknown OpenCL error";
        }
        g_print("%s: %s\nOpenCL Error: %s (%i)\n\n", message, callerName, clError, net->clConfig->clError);
        return TRUE;
    }
    return FALSE;
}


gboolean dR_exportFloatArrayToCSV(dR_Graph* net, gchar * filepath, gchar * filename, cl_mem deviceMem, cl_float * hostMem)
{
    FILE * f = NULL;
    gint x = 0, y = 0;
    guint i = 0;
    //gchar frameNumberStr[10];
    gchar * filepathAndFilename = NULL;
    dR_Node* start_layer;
    dR_DataFeedNode_Data* feednode;

    const gchar * sep = ", ";

    if (filepath != NULL)
    {
        filepathAndFilename = g_build_filename(filepath, filename, NULL);
    }
    else
    {
        gchar* currentDir = g_get_current_dir();
        filepathAndFilename = g_build_filename(currentDir, filename, NULL);
        if (currentDir)
            g_free((gpointer)currentDir);
    }

    if (net->config->debugInfo)
    {
        g_print("Writing float array to: %s\n", filepathAndFilename);
    }


#if defined(WIN32) || defined(WIN64)
    fopen_s(&f, filepathAndFilename, "w");
#else
    f = fopen(filename, "w");
#endif

    if ( f == NULL )
    {
        g_print("Could not open file to put Image as csv to disk.");
        return FALSE;
    }
    //file f ready to write
    dR_list_resetIt(net->feed_layers);
    start_layer = dR_list_next(net->feed_layers);
    feednode = ((dR_DataFeedNode_Data*)(start_layer->layer));
    g_print("%d x %d x %d", feednode->shape.s0,feednode->shape.s1,feednode->shape.s2);
    dR_downloadArray(net, filename, &deviceMem, 0, feednode->shape.s0*feednode->shape.s1 * sizeof(cl_float), (void*) hostMem);



    for (y = 0; y < feednode->shape.s1; y++)
    {
        for (x = 0; x < feednode->shape.s0; x++)
        {
            i = ((y * feednode->shape.s0) + x);

            if (x != 0)
                fputs(sep, f);

            fprintf(f, "%f", hostMem[i]);

        }//for x
        fputs("\n", f);
    }//for y

    fclose(f);

    if (filepathAndFilename)
        g_free((gpointer)filepathAndFilename);

    if (net->config->debugInfo)
    {
        g_print("Writing finished.\n");
    }

    return TRUE;
}

