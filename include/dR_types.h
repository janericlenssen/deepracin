#ifndef DR_TYPES_H
#define DR_TYPES_H

// dR_Graph Construction Structs as linked list

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenCL/cl.h>
#include <OpenCL/cl_gl.h>
#else
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif
#include <glib.h>
#include <string.h>
#include <stdlib.h>
#

typedef struct dR_Config dR_Config;
typedef struct dR_ClConfig dR_ClConfig;
typedef struct dR_Node dR_Node;
typedef struct dR_Graph dR_Graph;
typedef struct dR_ListElement dR_ListElement;
typedef struct dR_List dR_List;
typedef struct dR_DataFeedNode_Data dR_DataFeedNode_Data;
typedef struct dR_MemoryHandler dR_MemoryHandler;
typedef struct dR_Shape2 dR_Shape2;
typedef struct dR_Shape3 dR_Shape3;
typedef struct dR_Shape4 dR_Shape4;

// Function types for functions that need to/can be implemented for a node.
// Mandatory Node Functions
typedef gboolean (*fCompute)            (dR_Graph* net, dR_Node* layer);
typedef gboolean (*fSchedule)           (dR_Graph* net, dR_Node* layer);
typedef gboolean (*fPropagate)          (dR_Graph* net, dR_Node* layer);
typedef gint32   (*fMaxBuffer)          (dR_Node* layer);
typedef gboolean (*fCreateKernel)       (dR_Graph* net, dR_Node* layer);
typedef gboolean (*fAllocateBuffers)    (dR_Graph* net, dR_Node* layer);
typedef gboolean (*fFillBuffers)        (dR_Graph* net, dR_Node* layer);
typedef gboolean (*fCleanupBuffers)     (dR_Graph* net, dR_Node* layer);
typedef gboolean (*fCleanupLayer)       (dR_Graph* net, dR_Node* layer);
typedef gchar*   (*fPrintLayer)         (dR_Node* layer);
typedef gchar*   (*fSerializeNode)      (dR_Node* layer, gchar* params[], gint* numParams, gfloat* variables[], gint variableSizes[], gint* numVariables);
typedef dR_Node* (*fParseAppendNode)    (dR_Graph* net, dR_Node** iNodes, gint numINodes, gchar** params, gint numParams, gfloat** variables, gint numVariables);

// Optional Node Functions
typedef gboolean (*fGenerateKernel)     (dR_Graph* net, dR_Node* layer);
typedef gchar*   (*fCreateKernelName)   (dR_Node* layer);
typedef void     (*fSetVariables)       (dR_Node* layer, gfloat* weights, gfloat* bias);


///<\brief Generic Shape Struct (2 Elements)
struct dR_Shape2 {
    gint    s0;
    gint    s1;
};

///<\brief Generic Shape Struct (3 Elements)
struct dR_Shape3 {
    gint    s0;
    gint    s1;
    gint    s2;
};

///<\brief Generic Shape Struct (4 Elements)
struct dR_Shape4 {
    gint    s0;
    gint    s1;
    gint    s2;
    gint    s3;
};


enum dR_LayerType {
    tPooling,
    tBN,                           //Not Supported yet
    tFullyConnected,
    tConv2d,
    tPPFilter,
    tMaskDependentFilter,
    tLabelCreation,
    tNormalization,
    tUpscaling,
    tRGB2Gray,
    tMulScalar,
    tAddScalar,
    tDataFeedNode,
    tResolveRoI,
    tSoftmax,
    tSlice,
    tConcat,
    tCropOrPad,
    tElemWise2Op,
    tElemWise1Op,
    tFFT,
    tFFTShift,
    tFFTAbs,
    tSpecxture
};
typedef enum dR_LayerType dR_LayerType;

enum dR_ActivationType {
    tLinear,
    tReLU,
    tSigmoid,                       //Not Supported yet
    tTan                            //Not Supported yet
};
typedef enum dR_ActivationType dR_ActivationType;


struct dR_DataFeedNode_Data {
    dR_Shape3            shape;          // Size of Input Data
    cl_mem                      dataBuffer;
    dR_List*             next_layers;
};


struct dR_Node {
    dR_Shape3                   oshape;
    gint32                      layerID;
    dR_List*                    previous_layers;
    dR_List*                    next_layers;
    cl_kernel                   clKernel;
    dR_MemoryHandler*           outputBuf;
    void*                       layer;
    dR_LayerType                type;
    gboolean                    propagated;
    gboolean                    output;
    gboolean                    queued;

    // Functions
    fCompute                    compute;
    fSchedule                   schedule;
    fPropagate                  propagateShape;
    fMaxBuffer                  getRequiredOutputBufferSize;
    fCreateKernel               createKernel;
    fAllocateBuffers            allocateBuffers;
    fFillBuffers                fillBuffers;
    fCleanupBuffers             cleanupBuffers;
    fCleanupLayer               cleanupLayer;
    fPrintLayer                 printLayer;
    fSerializeNode              serializeNode;
    fParseAppendNode            parseAppendNode;

    fGenerateKernel             generateKernel;
    fCreateKernelName           createKernelName;
    fSetVariables               setVariables;
};

struct dR_ClConfig {
    cl_context          clContext;                  ///<\brief The OpenCL context
    cl_platform_id      clPlatformId;               ///<\brief The selected OpenCL platform.
    cl_command_queue    clCommandQueue;             ///<\brief The OpenCL command queue
    cl_program          clProgram;                  ///<\brief the OpenCL Program
    cl_uint             clDeviceType;               ///<\brief for providing the desired device type like GPU, CPU or ACCELERATOR
    cl_uint             clDeviceNumber;             ///<\brief The selected number in the clDeviceIds Array for getting the selected cdDeviceId
    gchar *             clPlatformName;             ///<\brief The OpenCL platform name
    gchar *             clKernelPath;               ///<\brief The path to the OpenCL kernels
    cl_device_id        clDeviceId;                 ///<\brief Holds the selected OpenCL device Id
    cl_uint             clNumDevices;               ///<\brief Hold the count of the found OpenCL devices
    cl_device_id        clDeviceIds[20];            ///<\brief Holds the found OpenCL device Ids in an array
    gchar *             clDeviceName;               ///<\brief Hold the device name of the selected OpenCL device
    cl_int              clError;                    ///<\brief Variable to catch the OpenCL error code which is then checked
    cl_event            clEventObject;              ///<\brief The event variable to catch some events from OpenCL
    cl_event *          clEvent;                    ///<\brief The event variable to catch some events from OpenCL
    gboolean            forceRecompile;             ///<\brief No precompiled binary usage if set to true
    gdouble             totalKernelGPUTime;         ///<\brief variable for profiling
    gdouble             queueKernelGPUTime;         ///<\brief variable for profiling
    gdouble             maxTotalKernelGPUTime;      ///<\brief variable for profiling
    gdouble             sumTotalKernelGPUTime;      ///<\brief variable for profiling
    gboolean            clInfo;                     ///<\brief Shows some information about the installed OpenCL devices if set to true.
    gboolean            fastRelaxedMath;            ///<\brief A flag to allow the use of faster relaxed amth calculations
    cl_build_status     build_status;               ///<\brief A variable to store the current OpenCL build status
    gboolean            cleanupDone;                ///<\brief A flag if the cleanup is already done and does not to be runned twice.
    gchar *             configH;                    ///<\brief Path to a config.h file when this should be used instead of config.h in "Kernels" folder
    glong               memConsumptionGPU;          ///<\brief variable for gpu memory consumption
};

struct dR_Config {
    gboolean        silent;                    ///<\brief Suppresses all console outputs if set to true
    gboolean        profilingGPU;              ///<\brief Activates GPU Profiling if set to true
    gboolean        profilingCPU;              ///<\brief Activates CPU Profiling if set to true
    gboolean        debugInfo;                 ///<\brief Shows some debug info if set to true.
    gdouble         totalNodeCPUTime;          ///<\brief variable for profiling
    gint            convLocalWorkSizeWindow;   ///<\brief Window size of workgroup (usually 16)
    gchar*          modelPath;                 ///<\brief Path to model files
};

struct dR_Graph {
    gboolean            prepared;
    cl_mem              swapBuffer1;
    cl_mem              swapBuffer2;
    dR_Config*   config;
    dR_ClConfig* clConfig;
    dR_List*     feed_layers;
    dR_List*     output_layers;
    dR_List*     streamBufferHandlers;
    gint32              number_of_layers;
    gint32              maxBufferSize;
    cl_float*           hostDebugBuffer;
    dR_List*     scheduledLayers;
    dR_List*     allNodes;
    gint                debugLayer;                 ///<\brief The content of layer with number "debugLayer" is downloaded into hostDebugBuffer every run
};

struct dR_MemoryHandler {
    dR_List*         outputFor;
    gint                    size;
    gint                    numberOfDependencies;
    cl_mem*                 bufptr;
    cl_mem                  buf;
    gboolean                outputBuf;
    gboolean                regionOfInterest;
    gboolean                useIndexBufferForRoI;
    cl_mem*                 indexBuffer;
    gint                    index;
    dR_Shape3        roiOrigin;

};

struct dR_ListElement {
    dR_ListElement*  next;
    dR_ListElement*  previous;
    void*                   element;
};

struct dR_List {
    dR_ListElement*  first;
    dR_ListElement*  last;
    gint                    length;
    dR_ListElement*  iterator;
};

/* CL_DEVICE_TYPES */
#define DEVICE_TYPE_DEFAULT (1 << 0)
#define DEVICE_TYPE_CPU (1 << 1)
#define DEVICE_TYPE_GPU (1 << 2)
#define DEVICE_TYPE_ACCELERATOR (1 << 3)
#define DEVICE_TYPE_ALL 0xFFFFFFFF

#define FORM_FACTOR_MIN_AREA 8.0f

#define TIME_SERIES_MAX_HISTORY 64
#define DYNAMIC_TIME_WARPING_MAX_PATTERN_SIZE 8

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

#ifndef M_2PI_F
#define M_2PI_F 6.28318530717958647692f
#endif

#ifndef M_SQRT_2PI_F
#define M_SQRT_2PI_F 2.50662827463100050241f
#endif

#ifndef FLT_EPSILON
#define FLT_EPSILON 0x1.0p23f
#endif



#endif // DR_TYPES_H
