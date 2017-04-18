import deepracin as dr
from scipy import misc
import numpy as np

# Properties

# If true, all debug outputs are printed (Default: False)
dr.env.debuginfo = False

# If true, the overall CPU runtimes are profiled and printed (Default: False)
dr.env.profileCPU = False

# If true, the GPU kernel runtimes are profiled and printed (Default: False)
dr.env.profileGPU = False

# If true, all outputs are supressed (Default: True)
dr.env.silent = False

# If not set, a temporary folder will be created in location depending on system
# Folder is used to store kernels, ptx, and (if model is exported) the exported model)
dr.env.model_path = '/media/jan/DataExt4/deepRacinModels/test'

# Can be a substring of actual platform name (NVIDIA, AMD, INTEL,...). If not set, first available device is chosen.
dr.env.preferred_platform_name = 'NVIDIA'

# Create empty graph
# interface_layout must be set. Determines how all communication with dR is interpreted.
# For example: Tensorflow/Numpy data layout is HWC. deepRACIN uses CHW internally.
graph = dr.create_graph(interface_layout='HWC')

# Get weights
w = np.load('/media/jan/DataExt4/deepRacinModels/vgg16/vgg16.npy', encoding='latin1').item()

# Fill graph
# Feed node - Will be fed with data for each graph application
feed = dr.feed_node(graph, shape=(224, 224, 3))

# VGG channel reorder and normalization
r, g, b = [feed[0:224, 0:224, 0] - 123.68,
           feed[0:224, 0:224, 1] - 116.779,
           feed[0:224, 0:224, 2] - 103.939]

concat = dr.Concat([b, g, r], 2)

# CNN layers
[filters, biases] = w['conv1_1'][0:2]
conv1_1 = dr.Conv2d(concat, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
[filters, biases] = w['conv1_2'][0:2]
conv1_2 = dr.Conv2d(conv1_1, filters.shape,[1, 1, 1, 1], 'relu', filters, biases)
pool1 = dr.Pooling(conv1_2, 'max', [1, 2, 2, 1],[1, 2, 2, 1])

[filters, biases] = w['conv2_1'][0:2]
conv2_1 = dr.Conv2d(pool1, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
[filters, biases] = w['conv2_2'][0:2]
conv2_2 = dr.Conv2d(conv2_1, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
pool2 = dr.Pooling(conv2_2, 'max', [1, 2, 2, 1], [1, 2, 2, 1])

[filters, biases] = w['conv3_1'][0:2]
conv3_1 = dr.Conv2d(pool2, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
[filters, biases] = w['conv3_2'][0:2]
conv3_2 = dr.Conv2d(conv3_1, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
[filters, biases] = w['conv3_3'][0:2]
conv3_3 = dr.Conv2d(conv3_2, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
pool3 = dr.Pooling(conv3_3, 'max', [1, 2, 2, 1], [1, 2, 2, 1])

[filters, biases] = w['conv4_1'][0:2]
conv4_1 = dr.Conv2d(pool3, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
[filters, biases] = w['conv4_2'][0:2]
conv4_2 = dr.Conv2d(conv4_1, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
[filters, biases] = w['conv4_3'][0:2]
conv4_3 = dr.Conv2d(conv4_2, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
pool4 = dr.Pooling(conv4_3, 'max', [1, 2, 2, 1], [1, 2, 2, 1])

[filters, biases] = w['conv5_1'][0:2]
conv5_1 = dr.Conv2d(pool4, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
[filters, biases] = w['conv5_2'][0:2]
conv5_2 = dr.Conv2d(conv5_1, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
[filters, biases] = w['conv5_3'][0:2]
conv5_3 = dr.Conv2d(conv5_2, filters.shape, [1, 1, 1, 1], 'relu', filters, biases)
pool5 = dr.Pooling(conv5_3, 'max', [1, 2, 2, 1], [1, 2, 2, 1])


[filters, biases] = w['fc6'][0:2]
fc6 = dr.Fully_Connected(pool5, filters.shape, 'relu', filters, biases)

[filters, biases] = w['fc7'][0:2]
fc7 = dr.Fully_Connected(fc6, filters.shape, 'relu', filters, biases)

[filters, biases] = w['fc8'][0:2]
fc8 = dr.Fully_Connected(fc7, filters.shape, 'linear', filters, biases)

logits = dr.Softmax(fc8)

# Mark output nodes (determines what dr.apply() returns)
dr.mark_as_output(concat)
dr.mark_as_output(logits)

# Print graph in console
dr.print_graph(graph)

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

