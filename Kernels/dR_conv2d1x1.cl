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
    int bW,             // Input Buffer Width
    int bH             // Input Buffer Height
    )
{

    const int filter_idx = get_global_id(2);
    int inputindex = 0;
    int outputindex = 0;
    int widthXheight = bW*bH;
    float out = 0.0f;
    const int inImageOffset = bW*get_global_id(1)+get_global_id(0);

    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<numI;i+=get_local_size(0)*get_local_size(1))
        lFilter[i] = gFilter[filter_idx*numI + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_global_id(0)>=bW||get_global_id(1)>=bH)
        return;

    for(inputindex = 0; inputindex<numI; inputindex++)
    {
        out += gInputImage[inputindex*widthXheight + inImageOffset] * lFilter[inputindex];
    }

    if(bias)
    {
        out += biases[filter_idx];
    }

    if(activation)
    {
        gOutputImage[filter_idx*widthXheight+inImageOffset] = clamp(out, 0.0f, FLT_MAX);
    }
    else
    {
        gOutputImage[filter_idx*widthXheight+inImageOffset] = out;
    }

}
#endif

