import deepracin as dr
import pywt as wt
from scipy import misc
import numpy as np
from skimage import io

preferred_platform_name = 'Beignet'

with dr.Environment(preferred_platform_name) as env:
    # Properties
    # If true, all debug outputs are printed (Default: False)
    env.debuginfo = True

    # If true, the overall CPU runtimes are profiled and printed (Default: False)
    env.profileCPU = True

    # If true, the GPU kernel runtimes are profiled and printed (Default: False)
    env.profileGPU = True

    # If true, all outputs are supressed (Default: True)
    env.silent = False

    # If not set, a temporary folder will be created in location depending on system
    # Folder is used to store kernels, ptx, and (if model is exported) the exported model)
    env.model_path = 'model/'

# Create empty graph
graph = env.create_graph(interface_layout='HWC')

# Fill graph
# Feed node - Will be fed with data for ea<ch graph application
#feed_node = dr.feed_node(graph, shape=(497, 303, 1))

#feed_node = dr.feed_node(graph, shape=(256, 256, 1))
feed_node = dr.feed_node(graph, shape=(8, 8, 1))

image_paths = ['dia8.png']
#image_paths = ['tigerbw64.png']

# create FFT node
ffttest = dr.FFT(feed_node)
hwt = dr.Haarwt(feed_node)
fftshifted = dr.FFTShift(ffttest)
fftmag = dr.FFTAbs(fftshifted)

# Mark output nodes (determines what dr.apply() returns)
dr.mark_as_output(hwt)

# Print graph to console
dr.print_graph(graph)

# Save deepracin graph
dr.save_graph(graph,env.model_path)

dr.prepare(graph)

for path in image_paths:

    # Feed Input
    img = io.imread(path)
    io.imshow(img)
    ##io.show()

    exp = np.expand_dims(img,2)

    data = np.array(exp).astype(np.float32)

    dr.feed_data(feed_node,data)

    # Apply graph - returns one numpy array for each node marked as output
    hwtOut = dr.apply(graph)
    dat = np.array(hwtOut[0]).astype(np.float32)

    #show output of specxture
    io.imshow(dat)
    io.show()

    # to use wavelets in python: pip install PyWavelets
    # from https://pywavelets.readthedocs.io/en/latest/
    x = [ [ 6, 12, 15, 15, 14, 12, 120, 116 ] ]

    #x = [ [1, 2, 3, 4], [0, 4, 3, 2], [1, 0, 4, 2], [1, 4, 3, 4] ]
    print '\n'
    print(x)

    c1 = wt.wavedec(x, 'haar')
    print '\n'
    print(c1)

    #print '\n'
    #print(c2)

    #print '\n'
    #print(c3)

    #print '\n'
    #print(c4)
