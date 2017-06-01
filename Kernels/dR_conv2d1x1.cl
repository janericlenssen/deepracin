#ifndef kerneldef_conv2d1x1
#define kerneldef_conv2d1x1

__kernel void conv2d1x1(
    const __global float * biases,
    const __global  float * gInputImage,
    const __global float * gFilter,
    __global  float * gOutputImage,
    __local float* lFilter,
    int bias,           // 0 no bias, 1 bias
    int activation,     // 0 linear, 1 relu
    int numO,              // Number of Filters
    int numI ,             // Number of Input Channels
    int depthPartitionSize, // Number of filters applied per workitem
    int bW,             // Input Buffer Width
    int bH             // Input Buffer Height
    )
{
    if(get_global_id(0)>=bW||get_global_id(1)>=bH)
        return;
    const int depthstart =  depthPartitionSize*get_global_id(2);
    int biasindex = depthstart;
    int inputindex = 0;
    int outputindex = 0;
    int widthXheight = bW*bH;
    const int inImageOffset = bW*get_global_id(1)+get_global_id(0);

    for(outputindex=0; outputindex<depthPartitionSize;outputindex++)
    {
        gOutputImage[biasindex*widthXheight+inImageOffset] = 0.0;
        biasindex++;
    }
    biasindex=depthstart;
    for(inputindex = 0; inputindex<numI; inputindex++)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<depthPartitionSize;i+=get_local_size(0)*get_local_size(1))
            lFilter[i] = gFilter[inputindex*numO + depthstart + i];
        barrier(CLK_LOCAL_MEM_FENCE);

        int findex = 0;
        for(outputindex=0; outputindex<depthPartitionSize;outputindex++)
        {
            gOutputImage[biasindex*widthXheight+inImageOffset] += gInputImage[inputindex*widthXheight + inImageOffset] * lFilter[findex];
            biasindex++;
            findex++;
        }
        biasindex = depthstart;

    }

    if(bias)
    {
        biasindex = depthstart;
        for(outputindex=0; outputindex<depthPartitionSize;outputindex++)
        {
            gOutputImage[biasindex*widthXheight+inImageOffset] += biases[biasindex];
            biasindex++;
        }
    }
    if(activation)
    {
        biasindex = depthstart;
        for(outputindex=0; outputindex<depthPartitionSize;outputindex++)
        {
            gOutputImage[biasindex*widthXheight+inImageOffset] = clamp(gOutputImage[biasindex*widthXheight+inImageOffset], 0.0f, FLT_MAX);
            biasindex++;
        }
    }

}
#endif

