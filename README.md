# deepRacin
<b>Deep Resource-aware OpenCL Inference Networks</b>
<ul>
<li>Deploy computation graphs (such as trained deep neural network models) to mobile or desktop OpenCL supporting platforms.
<li>Automated, resource-aware graph scheduling and parametrization
<li>Runs on Linux, Windows (and MacOS systems, not tested)
<li>Tested on Nvidia, AMD, Intel and mobile GPUs: Mali (OpenCL 1.1 required)
</ul>
<b>Note:</b> This library is under ongoing development and in alpha status. While the library should run on all OS and GPUs mentioned above in theory, it is not tested on all configurations.
 
If you have questions, feedback, suggestions or if you want to contribute, feel free to contact me!


<h2> Workflow </h2>
<ol>
<li> Define a computation graph in python, test it and store it in the deepRacin format
<li> Load the stored model in a C application 
<li> Initialize OpenCL or use an existing context and buffers
<li> In a processing loop: 
<ol>
<li> Feed data
<li> Run the graph
</ol>
</ol>

<h4>Basic examples</h4>
These are reduced examples to give the intuition of how to use deepRacin. Therefore, configuration and data loading code is omitted. See examples/vgg16 or examples/squeezenet-v1.1 for full examples of networks for classification of the 1000 ILSVRC2012 classes.

For <b>Step 1</b> in Python:
```python
import deepracin as dr
# Create empty graph
graph = dr.create_graph()

# Fill graph
# Feed node - Will be fed with data for each graph application
feed_node = dr.feed_node(graph, shape=(224, 224, 3))

# Conv2d node, given numpy arrays conv_weights and conv_biases
conv = dr.Conv2d(feed_node, shape, stride, activation='relu', weights=conv_weights, biases=conv_biases)

# MaxPooling node
pool = dr.Pooling(conv, pooling_type='max', shape, stride)

# FullyConnected node, given numpy arrays fc_weights and fc_biases
fc = dr.Fully_Connected(pool, shape, activation='relu', weights=fc_weights, biases=fc_biases)

# Mark output node
dr.mark_as_output(fc)

# Save deepracin graph
dr.save_graph(graph,model_path)

# Graph testing in python:
# Setup and schedule everything
dr.prepare(graph)

for img_data in img_paths:
    # Feed data
    dr.feed_data(feed_node,data)

    # Apply graph - returns one numpy array for each node marked as output
    fc_output = dr.apply(graph)
```

For <b>Steps 2</b>, <b>3</b> and <b>4</b> in C with a new OpenCL environment:
```c
// Load Graph
net = dR_NewGraph();
dR_loadGraph(net,model_path,&nodeslist,&numnodes,&feedlist,&numfeeds);

// Mark Output Node
dR_setAsOutput(net,nodeslist[numnodes-1]);

// Initialize OpenCL
dR_initCL(net);

// Setup and schedule everything
dR_prepare(net);

// Get OpenCL buffers for outputs
dR_getOutputBuffers(net,outbuffers);

for(int i = 0; i<numImages;i++)
{
    // Feed data
    dR_feedData(net,feedlist[0],(cl_float*)data[i],0,buffersize*sizeof(cl_float));
    // Apply graph
    dR_apply(net);
    // Get output data
    dR_downloadArray(net,"", outbuffers[0],0,out_size*sizeof(cl_float),data_out);
}
```
or with an existing OpenCL context and buffers:
```c
// Load Graph
net = dR_NewGraph();
dR_loadGraph(net,model_path,&nodeslist,&numnodes,&feedlist,&numfeeds);

// Use existing OpenCL context
dR_setClEnvironment(net, clContext, clPlatformId, clCommandQueue, clDeviceId);
dR_setDataFeedNodeBuffer(net,feedlist[0],existingCLMemPointer1);
dR_setPreexistingOutputBuffer(net,nodeslist[numnodes-1],existingCLMemPointer2);

// Setup and schedule everything
dR_prepare(net);

for(int i = 0; i<numImages;i++)
{
    ...
    // Apply graph
    dR_apply(net);
    ...
}
```
<h2> Getting Started </h2>

Dependencies of the C library:
<ul>
<li> OpenCL 1.1
<li> Glib 2.0 
</ul>

Dependencies of the Python interface:
<ul>
<li> Numpy
</ul>

Misc:
<ul>
<li>For the C part of the examples, libpng is required to load test images.
<li>For building, CMake 2.8 (3.4 on Windows) is required.
</ul>

<h4> Installation </h4>
On Linux:
<ol>
<li> Install glib > 2.6, OpenCL, libpng and zlib
<li> Checkout deepRacin git repository
<li> Navigate to checkout folder 
<li> Create build dir, navigate there

```sh
mkdir build
cd build
```
<li>  Apply cmake. Choose ON or OFF for options (without brackets). Note that Python and Numpy are required for installing the Python interface and libpng is required for building the examples

```sh
cmake .. -DINSTALL_PYTHON_INTERFACE=<ON|OFF> -DCOMPILE_EXAMPLES=<ON|OFF>
```
<li>  Install the library

```sh
sudo make install
```
</ol>

On Windows: (Overview, detailed version coming soon)
<ol>
<li> Download and compile glib > 2.6, libpng and zlib with Visual Studio and install OpenCL
<li> Checkout deepRacin git repository
<li> Use CMake to configure
<li> Set all missing paths to OpenCL, glib and libpng
<li> Adjust Install Prefix
<li> Generate Project
<li> Build INSTALL Target of the generated Visual Studio Project
</ol>

<h2> Currently implemented graph nodes </h2>
<ul>
  <li>DataFeedNode</li>
  <li>DNN Nodes</li>
  <ul>
    <li>Conv2d (direct, winograd(2x2, 3x3) and specialized 1x1 implementations)</li>
    <li>Pooling (currently Max, Avg)</li>
    <li>FullyConnected</li>
    <li>Activation fuctions (currently ReLU, Linear)</li>
    <li>Softmax</li>
  </ul>
  <li>Math Operations</li>
  <ul>
    <li>Add (with tensor or scalar)</li>
    <li>Sub (with tensor or scalar)</li>
    <li>Mul (with tensor or scalar)</li>
    <li>Div (with tensor or scalar)</li>
    <li>Pow (with tensor or scalar)</li>
    <li>Log</li>
    <li>Sqrt</li>
    <li>Exp</li>
    <li>Fill</li>
  </ul>
  <li>Transforms</li>
  <ul>
    <li>Concat</li>
    <li>Slice</li>
  </ul>
  <li>Image</li>
  <ul>
    <li>Normalization (per image to given mean and stddev) </li>
    <li>CropOrPad</li>
    <li>Upscaling</li>
    <li>RGBtoGray</li>
    <li>MaskDependentFilter (applies one of k image filters to each pixel, depending on integer mask)</li>
  </ul>
</ul>
All implementations are given as OpenCL host and device code.

<h2> Benchmarks </h2>
TODO

<h2> Acknowledgement </h2>
This work has been supported by Deutsche Forschungsgemeinschaft (DFG) within the Collaborative Research Center SFB 876 “Providing Information by Resource-Constrained Analysis”, project B2.
