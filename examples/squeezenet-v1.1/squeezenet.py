import deepracin as dr
from scipy import misc
import numpy as np

# Can be a substring of actual platform name (NVIDIA, AMD, INTEL,...). If not set, first available device is chosen.
preferred_platform_name = 'NVIDIA'

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
    env.model_path = '/media/jan/DataExt4/deepRacinModels/test'



    # Create empty graph
    # interface_layout must be set. Determines how all communication with dR is interpreted.
    # For example: Tensorflow/Numpy data layout is HWC. deepRACIN uses CHW internally.
    graph = env.create_graph(interface_layout='HWC')

    # Get weights
    w = np.load('squeezenet11.npy', encoding='latin1').item()

    # Fill graph
    # Feed node - Will be fed with data for each graph application
    feed = dr.feed_node(graph, shape=(224, 224, 3))

    r, g, b = [feed[0:224, 0:224, 0] - 123.68,
               feed[0:224, 0:224, 1] - 116.779,
               feed[0:224, 0:224, 2] - 103.939]

    concat = dr.Concat([b, g, r], 2)

    # CNN layers
    filters, biases = w['conv1']['weights'], w['conv1']['biases']
    conv1 = dr.Conv2d(concat, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
    pool1 = dr.Pooling(conv1, 'average', [1, 2, 2, 1],[1, 2, 2, 1])
    pool2 = dr.Pooling(pool1, 'max', [1, 3, 3, 1],[1, 2, 2, 1])

    filters, biases = w['fire2_squeeze1x1']['weights'], w['fire2_squeeze1x1']['biases']
    fire2_squeeze1x1 = dr.Conv2d(pool2, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases = w['fire2_expand1x1']['weights'], w['fire2_expand1x1']['biases']
    fire2_expand1x1 = dr.Conv2d(fire2_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases = w['fire2_expand3x3']['weights'], w['fire2_expand3x3']['biases']
    fire2_expand3x3 = dr.Conv2d(fire2_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    fire2_concat = dr.Concat([fire2_expand1x1,fire2_expand3x3],concat_dim=2)

    filters, biases = w['fire3_squeeze1x1']['weights'], w['fire3_squeeze1x1']['biases']
    fire3_squeeze1x1 = dr.Conv2d(fire2_concat, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases = w['fire3_expand1x1']['weights'], w['fire3_expand1x1']['biases']
    fire3_expand1x1 = dr.Conv2d(fire3_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases = w['fire3_expand3x3']['weights'], w['fire3_expand3x3']['biases']
    fire3_expand3x3 = dr.Conv2d(fire3_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    fire3_concat = dr.Concat([fire3_expand1x1,fire3_expand3x3],concat_dim=2)

    pool3 = dr.Pooling(fire3_concat, 'max', [1, 3, 3, 1],[1, 2, 2, 1])

    filters, biases = w['fire4_squeeze1x1']['weights'], w['fire4_squeeze1x1']['biases']
    fire4_squeeze1x1 = dr.Conv2d(pool3, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases = w['fire4_expand1x1']['weights'], w['fire4_expand1x1']['biases']
    fire4_expand1x1 = dr.Conv2d(fire4_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases = w['fire4_expand3x3']['weights'], w['fire4_expand3x3']['biases']
    fire4_expand3x3 = dr.Conv2d(fire4_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    fire4_concat = dr.Concat([fire4_expand1x1,fire4_expand3x3],concat_dim=2)

    filters, biases = w['fire5_squeeze1x1']['weights'], w['fire5_squeeze1x1']['biases']
    fire5_squeeze1x1 = dr.Conv2d(fire4_concat, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases = w['fire5_expand1x1']['weights'], w['fire5_expand1x1']['biases']
    fire5_expand1x1 = dr.Conv2d(fire5_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases = w['fire5_expand3x3']['weights'], w['fire5_expand3x3']['biases']
    fire5_expand3x3 = dr.Conv2d(fire5_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    fire5_concat = dr.Concat([fire5_expand1x1,fire5_expand3x3],concat_dim=2)

    pool5 = dr.Pooling(fire5_concat, 'max', [1, 3, 3, 1],[1, 2, 2, 1])

    filters, biases = w['fire6_squeeze1x1']['weights'], w['fire6_squeeze1x1']['biases']
    fire6_squeeze1x1 = dr.Conv2d(pool5, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases = w['fire6_expand1x1']['weights'], w['fire6_expand1x1']['biases']
    fire6_expand1x1 = dr.Conv2d(fire6_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases = w['fire6_expand3x3']['weights'], w['fire6_expand3x3']['biases']
    fire6_expand3x3 = dr.Conv2d(fire6_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    fire6_concat = dr.Concat([fire6_expand1x1,fire6_expand3x3],concat_dim=2)

    filters, biases  = w['fire7_squeeze1x1']['weights'], w['fire7_squeeze1x1']['biases']
    fire7_squeeze1x1 = dr.Conv2d(fire6_concat, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases  = w['fire7_expand1x1']['weights'], w['fire7_expand1x1']['biases']
    fire7_expand1x1 = dr.Conv2d(fire7_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases  = w['fire7_expand3x3']['weights'], w['fire7_expand3x3']['biases']
    fire7_expand3x3 = dr.Conv2d(fire7_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    fire7_concat = dr.Concat([fire7_expand1x1,fire7_expand3x3],concat_dim=2)

    filters, biases  = w['fire8_squeeze1x1']['weights'], w['fire8_squeeze1x1']['biases']
    fire8_squeeze1x1 = dr.Conv2d(fire7_concat, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases  = w['fire8_expand1x1']['weights'], w['fire8_expand1x1']['biases']
    fire8_expand1x1 = dr.Conv2d(fire8_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases  = w['fire8_expand3x3']['weights'], w['fire8_expand3x3']['biases']
    fire8_expand3x3 = dr.Conv2d(fire8_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    fire8_concat = dr.Concat([fire8_expand1x1,fire8_expand3x3],concat_dim=2)

    filters, biases  = w['fire9_squeeze1x1']['weights'], w['fire9_squeeze1x1']['biases']
    fire9_squeeze1x1 = dr.Conv2d(fire8_concat, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases  = w['fire9_expand1x1']['weights'], w['fire9_expand1x1']['biases']
    fire9_expand1x1 = dr.Conv2d(fire9_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    filters, biases  = w['fire9_expand3x3']['weights'], w['fire9_expand3x3']['biases']
    fire9_expand3x3 = dr.Conv2d(fire9_squeeze1x1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
    fire9_concat = dr.Concat([fire9_expand1x1,fire9_expand3x3],concat_dim=2)

    filters, biases  = w['conv10']['weights'], w['conv10']['biases']
    conv10 = dr.Conv2d(fire9_concat, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)

    pool10 = dr.Pooling(conv10, 'average', [1, 14, 14, 1], [1, 14, 14, 1])

    softmax = dr.Softmax(pool10)

    # Mark output nodes (determines what dr.apply() returns)
    dr.mark_as_output(feed)
    dr.mark_as_output(softmax)

    # Print graph to console
    dr.print_graph(graph)

    # Save graph in dr format
    dr.save_graph(graph,'../../build/deepRacinModels/')

    # Prepare graph for execution (setup and initialize)
    dr.prepare(graph)


    # Graph application
    image_paths = ['tiger.png','puzzle.png']
    for path in image_paths:
        # Feed Input
        img = misc.imread(path)
        data = np.array(img).astype(np.float32)
        dr.feed_data(feed,data)

        # Apply graph - returns one numpy array for each node marked as output
        feeddata, logits = dr.apply(graph)
        classid = np.argmax(logits)

        # Display fed data and inference result
        synset = [l.strip() for l in open('/media/jan/DataExt4/deepRacinModels/vgg16/synset.txt').readlines()]

        print('Class: '+str(classid)+', '+synset[classid])
        misc.imshow(feeddata[:,:,0:3])

