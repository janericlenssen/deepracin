import deepracin as dr
import numpy as np

save_path = 'variables.npy'

with dr.Environment('INTEL') as env:
    # Properties
    graph = env.create_graph(interface_layout='HWC')
    env.debuginfo = False
    env.profileCPU = True
    env.profileGPU = False
    env.silent=True
    # Get weights
    w = np.load(save_path, encoding='latin1').item()

    # Fill graph
    # Feed node - Will be fed with data for each graph application
    feed = dr.feed_node(graph, shape=(32, 32, 1))
    std = dr.Normalization(feed,'target_mean_stddev')

    conv1 = dr.Conv2d(std,w['conv1_w'].shape, [1,1,1,1], 'relu', w['conv1_w'], w['conv1_b'])


    pool1 = dr.Pooling(conv1, 'max', [1, 2, 2, 1], [1, 2, 2, 1])

    fire_squeeze1x1 = dr.Conv2d(pool1, w['conv2_w'].shape, [1, 1, 1, 1], 'relu', w['conv2_w'], w['conv2_b'])

    fire_expand1x1 = dr.Conv2d(fire_squeeze1x1, w['conv3_w'].shape, [1, 1, 1, 1], 'relu', w['conv3_w'],
                                w['conv3_b'])

    fire_expand3x3 = dr.Conv2d(fire_squeeze1x1, w['conv4_w'].shape, [1, 1, 1, 1], 'relu', w['conv4_w'],
                                w['conv4_b'])
    fire_concat = dr.Concat([fire_expand1x1, fire_expand3x3], concat_dim=2)

    pool2 = dr.Pooling(fire_concat, 'max', [1,2,2,1], [1,2,2,1])

    fire1_squeeze1x1 = dr.Conv2d(pool2, w['conv5_w'].shape, [1, 1, 1, 1], 'relu', w['conv5_w'], w['conv5_b'])

    fire1_expand1x1 = dr.Conv2d(fire1_squeeze1x1, w['conv6_w'].shape, [1, 1, 1, 1], 'relu', w['conv6_w'],
                                w['conv6_b'])

    fire1_expand3x3 = dr.Conv2d(fire1_squeeze1x1, w['conv7_w'].shape, [1, 1, 1, 1], 'relu', w['conv7_w'],
                                w['conv7_b'])
    fire1_concat = dr.Concat([fire1_expand1x1, fire1_expand3x3], concat_dim=2)

    pool3 = dr.Pooling(fire1_concat, 'max', [1, 2, 2, 1], [1, 2, 2, 1])

    conv2 = dr.Conv2d(pool3, w['conv8_w'].shape, [1, 1, 1, 1], 'linear', w['conv8_w'], w['conv8_b'])



    pool4 = dr.Pooling(conv2, 'average', [1, 4, 4, 1], [1, 4, 4, 1])

    logits = dr.Softmax(pool4)


    # Mark output nodes (determines what dr.apply() returns)
    dr.mark_as_output(feed)
    dr.mark_as_output(logits)

    # Print graph in console
    dr.print_graph(graph)

    # Store dR graph for loading in C or python
    #dr.save_graph(graph,'/media/jan/Data/Projects/test/')
    dr.prepare(graph)

    for i in range(10):
        x = np.random.rand(32, 32, 1).astype(np.float32)
        dr.feed_data(feed, x)
        dr.apply(graph)
