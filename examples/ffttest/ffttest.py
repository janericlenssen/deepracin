import deepracin as dr
from scipy import misc
import numpy as np
from skimage import io

preferred_platform_name = 'Mesa'

with dr.Environment(preferred_platform_name) as env:
    # Properties
    # If true, all debug outputs are printed (Default: False)
    env.debuginfo = False

    # If true, the overall CPU runtimes are profiled and printed (Default: False)
    env.profileCPU = False

    # If true, the GPU kernel runtimes are profiled and printed (Default: False)
    env.profileGPU = False

    # If true, all outputs are supressed (Default: True)
    env.silent = False

    # If not set, a temporary folder will be created in location depending on system
    # Folder is used to store kernels, ptx, and (if model is exported) the exported model)
    env.model_path = 'model/'

# Create empty graph
graph = env.create_graph(interface_layout='CHW')

# Fill graph
# Feed node - Will be fed with data for each graph application
feed_node = dr.feed_node(graph, shape=(4, 4, 1))

###

###

# create FFT node
ffttest = dr.FFT(feed_node) # AttributeError: 'module' object has no attribute 'FFT'
#		test = dr.ElemWise2Operation(graph, feed_node, feed_node, Add)

# Mark output nodes (determines what dr.apply() returns)
dr.mark_as_output(ffttest)

# Print graph to console
dr.print_graph(graph)

# Save deepracin graph
dr.save_graph(graph,env.model_path)

dr.prepare(graph)

image_paths = ['4by4black.png']

for path in image_paths:

        # Feed Input
        img = io.imread(path)
        data = np.array(img).astype(np.float32)
        #dr.feed_data(feed_node,data) # data array  does not match node's dimension

        # Apply graph - returns one numpy array for each node marked as output
        feeddata = dr.apply(graph)
	io.imshow(img)
	#io.show()
