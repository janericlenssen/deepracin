import deepracin as dr
import pywt as wt
from scipy import misc
import numpy as np
from skimage import io

preferred_platform_name = 'Beignet'

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



def fftfeatures(features, path, first):
    print(path)
    graph = env.create_graph(interface_layout='HWC')
    feed_node = dr.feed_node(graph, shape=(64, 64, 1))
    ffttest = dr.FFT(feed_node)
    fftshifted = dr.FFTShift(ffttest)
    fftmag = dr.FFTAbs(fftshifted)
    specxture = dr.Specxture(fftmag)
    dr.mark_as_output(specxture)
    #dr.save_graph(graph,env.model_path)
    dr.prepare(graph)
    # Create empty graph

    #graph = env.create_graph(interface_layout='HWC')

    # Fill graph
    # Feed node - Will be fed with data for ea<ch graph application
    #feed_node = dr.feed_node(graph, shape=(497, 303, 1))

    #feed_node = dr.feed_node(graph, shape=(256, 256, 1))
    #feed_node = dr.feed_node(graph, shape=(64, 64, 1))

    #image_paths = ['0.png']
    #image_paths = ['tigerbw64.png']

    # create FFT node
    #ffttest = dr.FFT(feed_node)
    #fftshifted = dr.FFTShift(ffttest)
    #fftmag = dr.FFTAbs(fftshifted)
    #specxture = dr.Specxture(fftmag)

    # Mark output nodes (determines what dr.apply() returns)
    #dr.mark_as_output(specxture)

    # Print graph to console
    #dr.print_graph(graph)

    # Save deepracin graph
    #dr.save_graph(graph,env.model_path)

    #dr.prepare(graph)


    # Feed Input
    img = io.imread(path)
    io.imshow(img)
    #io.show()

    exp = np.expand_dims(img,2)

    data = np.array(exp).astype(np.float32)

    dr.feed_data(feed_node,data)

    # Apply graph - returns one numpy array for each node marked as output
    specOut = dr.apply(graph)
    dat = np.array(specOut[0]).astype(np.float32)
    features.append(dat)
    #dat = np.array(fftmag[0]).astype(np.float32)
    #print(features)
    #show output of specxture
    #io.imshow(dat)
    #io.show()
