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
    env.silent = True

    # If not set, a temporary folder will be created in location depending on system
    # Folder is used to store kernels, ptx, and (if model is exported) the exported model)
    env.model_path = 'model/'

# Create empty graph
graph = env.create_graph(interface_layout='HWC')

# Fill graph
# Feed node - Will be fed with data for each graph application
#feed_node = dr.feed_node(graph, shape=(497, 303, 1))

feed_node = dr.feed_node(graph, shape=(224, 224, 1))

# create FFT node
ffttest = dr.FFT(feed_node)

# Mark output nodes (determines what dr.apply() returns)
dr.mark_as_output(ffttest)

# Print graph to console
dr.print_graph(graph)

# Save deepracin graph
dr.save_graph(graph,env.model_path)

dr.prepare(graph)

image_paths = ['allblack.png']

for path in image_paths:

    # Feed Input
    img = io.imread(path)
    print "Input image dimensions: " + '\n' + str(img.shape) + '\n'
    exp = np.expand_dims(img,2)
    type(img)
    print "Expanded input image dimensions: " + '\n' + str(exp.shape) + '\n'
    data = np.array(exp).astype(np.float32)
    print "np.array image dimensions: " + '\n' + str(data.shape) + '\n'
    dr.feed_data(feed_node,data)

    # Apply graph - returns one numpy array for each node marked as output
    fftout = dr.apply(graph)
    #print "fftout image dimensions: " + '\n' + str(fftout.shape) + '\n'

    dat = np.array(fftout[0]).astype(np.float32)
    print '\n' + "Output image dimensions: " + '\n' + str(dat.shape) + '\n'

    io.imshow(dat[:, :, 0])
    io.show()

    io.imshow(dat[:, :, 1])
    io.show()
