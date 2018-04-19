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

image_paths = ['tigerbw8.png']
#image_paths = ['tigerbw64.png']

# create wavelet node
hwt = dr.Haarwt(feed_node)

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
    #io.use_plugin('qt')
    io.imshow(img)
    ##io.show()

    exp = np.expand_dims(img,2)

    data = np.array(exp).astype(np.float32)

    dr.feed_data(feed_node,data)

    # Apply graph - returns one numpy array for each node marked as output
    hwtOut = dr.apply(graph)
    dat = np.array(hwtOut[0]).astype(np.float32)
    print(dat)
    #show output of specxture
    io.imshow(dat)
    ##io.show()

    # to use wavelets in python: pip install PyWavelets
    # from https://pywavelets.readthedocs.io/en/latest/
    x = [ 6, 12, 15, 15, 14, 12, 120, 116 ]

    #wavedec(x, 'haar', 3) = [array([[ 109.60155108]]), array([[-75.66042559]]), array([[  -6., -105.]]), array([[-4.24264069,  0.        ,  1.41421356,  2.82842712]])]

    tigerbw8 = [64, 91, 108, 123, 123, 136, 170, 133 ]
    # 1. stage: 109.60 163.34 183.14 214.25 -19.09 -10.61 -9.19 26.16


    #ulexp = [1.000, 2.000, 3.000, 1.000, 2.000, 3.000, 4.000, 0.000]
    # wavedec(tigerbw8, 'haar', 3) = [array([ 335.16861428]), array([-62.22539674]), array([-38., -22.]), array([-19.09188309, -10.60660172,  -9.19238816,  26.1629509 ])]

    #x = [ [1, 2, 3, 4], [0, 4, 3, 2], [1, 0, 4, 2], [1, 4, 3, 4] ]
    print('\n')
    #print(ulexp)

    c1 = wt.wavedec(tigerbw8, 'haar', level=3)
    print('\n')
    print(c1)

    #print '\n'
    #print(c2)

    #print '\n'
    #print(c3)

    #print '\n'
    #print(c4)
