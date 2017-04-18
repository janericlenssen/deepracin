import deepracin as dr
from scipy import misc
import numpy as np

# Properties
dr.env.debuginfo = False
dr.env.profileCPU = False
dr.env.profileGPU = False
dr.env.silent = False
# Can be a substring of actual platform name (NVIDIA, AMD, INTEL,...). If not set, first available device is chosen.
dr.env.preferred_platform_name = 'NVIDIA'

# Create empty graph
# interface_layout must be set. Determines how all communication with dR is interpreted.
# For example: Tensorflow/Numpy data layout is HWC. deepRACIN uses CHW internally.
graph = dr.create_graph(interface_layout='HWC')

# Load graph
nodes, feednodes = dr.load_graph(graph, '/media/jan/DataExt4/deepRacinModels/vgg16_whole/')
print(len(nodes),len(feednodes))
# Mark output nodes (determines what dr.apply() returns)
dr.mark_as_output(feednodes[0])
dr.mark_as_output(nodes[29])

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
    dr.feed_data(feednodes[0],data)

    # Apply graph - returns one numpy array for each node marked as output
    feeddata, logits = dr.apply(graph)
    classid = np.argmax(logits)

    # Display fed data and inference result
    synset = [l.strip() for l in open('/media/jan/DataExt4/deepRacinModels/vgg16/synset.txt').readlines()]
    print('Class: '+str(classid)+', '+synset[classid])
    misc.imshow(feeddata)

