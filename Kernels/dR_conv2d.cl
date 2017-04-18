

/**
* \brief Copy parts of the input image from global to local memory
* \param[in] gImage The input image
* \param[in] lImage A local cache for the image of the size of the local work group enhanced by filterWidth - 1 and filterHeight -1 in width and height.
* \param[in] floatsToReadInThisWG Number of floats to read
* \param[in] numberOfWorkItems Number of work items in serialized work group
* \param[in] localID Serialized local_id
* \param[in] copyWidth Width of the block to read
* \param[in] blx Bottom-left x-coordinate of block to read
* \param[in] bly Bottom-left y-coordinate of block to read
* \param[in] width Width of the image
* \param[in] height Height of the image
* \author jan
*/

void copyiImageToLocal(
    const __global float * gInputImage,
    __local float * lImage,
    int filterWidth,
    int filterHeight,
    int width,
    int height
    )
{
    int floatsToReadInThisWG = (get_local_size(0)+filterWidth-1)*(get_local_size(1)+filterHeight-1);
    int numberOfWorkItems = get_local_size(1)*get_local_size(0);
    int localID = get_local_id(1)*get_local_size(0)+get_local_id(0);
    int copyWidth = get_local_size(0)+ filterWidth-1;
    int blx = get_group_id(0)/get_num_groups(0)-filterWidth/2;
    int bly = get_group_id(1)/get_num_groups(1)-filterHeight/2;
    for(int i = localID; i<floatsToReadInThisWG;i+=numberOfWorkItems)
    {
        //lImage[i] = getClampedFloat(gInputImage, blx + (i % copyWidth), bly + (i / copyWidth), width, height, width);
        lImage[i] = gInputImage[mad24(clamp(bly + (i / copyWidth),0,height), width,clamp(blx + (i % copyWidth),0,width))];
    }
}

/**
* \brief Copy the assigned parts from global to local memory
* \param[in] gInputImage The input image
* \param[in] gFilter Constant buffer with the filter that should be applied.
* \param[in] lImage A local cache for the image of the size of the local work group enhanced by filterWidth - 1 and filterHeight -1 in width and height.
* \param[in] width Width of the image without padding
* \param[in] height Height of the image without padding
* \param[in] filterWidth Width of the gaussian kernel
* \param[in] filterHeight Height of the gaussian kernel
* \author jan
*/
void copyFilterToLocal(
    const __global float * gFilter,
    __local float * lFilter,
    int floatsToReadInThisWG,
    int numberOfWorkItems,
    int localID
    )
{
    //nt floatsToReadInWI = (floatsToReadInThisWG/numberOfWorkItems);
    /*for(int i = floatsToReadInThisWG*localID; i<floatsToReadInWI*(localID+1);i++)
    {
        lFilter[i] = gFilter[i];
    }*/
    for(int i = localID; i<floatsToReadInThisWG;i+=numberOfWorkItems)
    {
        lFilter[i] = gFilter[i];
    }
}





/**
* \brief Convolution of an image with a filter element
* \param[in] gInputImage The input image
* \param[in] gFilter Constant buffer with the filter that should be applied.
* \param[in] lImage A local cache for the image of the size of the local work group enhanced by filterWidth - 1 and filterHeight -1 in width and height.
* \param[in] width Width of the image without padding
* \param[in] height Height of the image without padding
* \param[in] filterWidth Width of the gaussian kernel
* \param[in] filterHeight Height of the gaussian kernel
* \author jan
*/
void convolutionWithImageCopy(
    const __global  float * gInputImage,
    const __global float * gFilter,
    __local float* lImage,
    __local float* lFilter,
    const int width,
    const int height,
    const int inputDepth,
    const int filterWidth,
    const int filterHeight,
    const int outputDepth,
    float* outputArr
    )
{
    const int lengthindepth = outputDepth/get_global_size(2);
    const int depthstart = lengthindepth*get_global_id(2);
    const int halfFilterWidth = filterWidth/2;
    const int halfFilterHeight = filterHeight/2;
    const int blx = get_group_id(0)/get_num_groups(0)-halfFilterWidth;
    const int bly = get_group_id(1)/get_num_groups(1)-halfFilterHeight;
    const int serializedLocalSize = get_local_size(1)*get_local_size(0);
    const int lwidth = get_local_size(0)+ filterWidth-1;
    const int indexstart = mad24(get_local_id(1), lwidth, get_local_id(0));
    const int jumpint = lwidth - filterWidth;
    const int filterxy = filterWidth*filterHeight;
    const int widthheight = mul24(width,height);
    const int numberOfFloatsPerWGImg = (get_local_size(0)+filterWidth-1)*(get_local_size(1)+filterHeight-1);
    const int serializedLocalId = get_local_id(1)*get_local_size(0)+get_local_id(0);
    const int numberOfFloatsPerWGFil = mul24(lengthindepth,filterxy);
    int fid = 0;
    int index = 0;
    float sum = 0.0f;
    //float pWindow[25];

    for(int inputindex = 0; inputindex<inputDepth; inputindex++)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        copyiImageToLocal(
            gInputImage + mul24(inputindex, widthheight),
            lImage,
            filterWidth,
            filterHeight,
            width,
            height
            );
        copyFilterToLocal(
            gFilter+mad24(mul24(inputindex,filterWidth), mul24(filterHeight,outputDepth), mul24(depthstart,filterxy)),
            lFilter,
            numberOfFloatsPerWGFil,
            serializedLocalSize,
            serializedLocalId
            );
        /*copylWindowToPrivate(
            lImage + indexstart,
            pWindow,
            filterHeight,
            filterWidth,
            lwidth
            );*/
        barrier(CLK_LOCAL_MEM_FENCE);
        fid = 0;
        for(int outputindex = 0; outputindex<lengthindepth;outputindex++)
        {
        index = indexstart;

            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            index+=jumpint;

            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            index+=jumpint;

            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            index+=jumpint;

            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            index+=jumpint;

            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            sum += lImage[index] * lFilter[fid];
            fid++;
            index++;
            index+=jumpint;
            //outputArr[outputindex] = sum;
            float x = sum+2.0;
        }
    }
}

/**
* \brief Convolution of an image with a filter element without Local Image Copy
* \param[in] gInputImage The input image
* \param[in] gFilter Constant buffer with the filter that should be applied.
* \param[in] width Width of the image without padding
* \param[in] height Height of the image without padding
* \param[in] filterWidth Width of the gaussian kernel
* \param[in] filterHeight Height of the gaussian kernel
* \author jan
*/
void convolutionWithoutImageCopy(
    const __global  float * gInputImage,
    const __global float * gFilter,
    __local float* lFilter,
    const int width,
    const int height,
    const int inputDepth,
    const int filterWidth,
    const int filterHeight,
    const int outputDepth,
    float* outputArr
    )
{

    int lengthindepth = outputDepth/get_global_size(2);
    int depthstart = lengthindepth*get_global_id(2);
    int serializedLocalSize = get_local_size(1)*get_local_size(0);
    int halfFilterWidth = filterWidth/2;
    int halfFilterHeight = filterHeight/2;
    int localwidth = get_local_size(0)+ filterWidth-1;

    for(int inputindex = 0; inputindex<inputDepth; inputindex++)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        copyFilterToLocal(
            gFilter+(inputindex*filterWidth*filterHeight*outputDepth)+(depthstart*filterWidth*filterHeight),
            lFilter,
            lengthindepth*filterWidth*filterHeight,
            serializedLocalSize,
            get_local_id(1)*get_local_size(0)+get_local_id(0)
            );
        barrier(CLK_LOCAL_MEM_FENCE);
        int fid = 0;
        for(int outputindex = depthstart; outputindex<depthstart+lengthindepth;outputindex++)
        {
            for (int yy = - halfFilterHeight; yy <= halfFilterHeight; yy++)
            {
                for (int xx = - halfFilterWidth; xx <= halfFilterWidth; xx++)
                {
                    int index = mad24(get_global_id(1) + halfFilterHeight + yy, localwidth, get_local_id(0) + halfFilterWidth + xx);
                    outputArr[outputindex-depthstart] += sampleZeroPaddedFloat(gInputImage,get_global_id(0)+xx,get_global_id(1)+yy,width,height) * lFilter[fid];
                    fid++;
                }
            }
        }
    }
}


/**
* \brief Convolutional Layer with ReLU activation with LMEM
* \param[in] gInputImage The input image
* \param[in] gFilter Constant buffer with the filter that should be applied.
* \param[out] gOutputImage Filtered output image
* \param[in] lImage A local cache for the image of the size of the local work group enhanced by filterWidth - 1 and filterHeight -1 in width and height.
* \param[in] width Width of the image without padding
* \param[in] height Height of the image without padding
* \param[in] filterWidth Width of the gaussian kernel
* \param[in] filterHeight Height of the gaussian kernel
* \author jan
*/
__kernel void conv2dReLU(
    const __global  float * gInputImage,
    const __global float * gFilter,
    const __global float * biases,
    __global  float * gOutputImage,
    __local float * lImage,
    __local float * lFilter,
    const int width,
    const int height,
    const int depth,
    const int filterWidth,
    const int filterHeight,
    const int outputDepth
)
{
    int lengthindepth = outputDepth/get_global_size(2);
    int depthstart = lengthindepth*get_global_id(2);
    float outArr[256];
    outArr[0] = 0.0;
    convolutionWithImageCopy(gInputImage, gFilter, lImage, lFilter, width, height, depth, filterWidth, filterHeight, outputDepth, outArr);

    int inImgOffset = mad24(width,get_global_id(1),get_global_id(0));
    for(int i = depthstart; i< depthstart+lengthindepth; i++)
    {
        int layerpt = i*width*height;
        gOutputImage[layerpt+inImgOffset] = clamp(outArr[0]+biases[layerpt+inImgOffset], 0.0f, FLT_MAX);
    }
}

/**
* \brief Convolutional Layer with ReLU activation without LMEM (for stride>=filtersize)
* \param[in] gInputImage The input image
* \param[in] gFilter Constant buffer with the filter that should be applied.
* \param[out] gOutputImage Filtered output image
* \param[in] lImage A local cache for the image of the size of the local work group enhanced by filterWidth - 1 and filterHeight -1 in width and height.
* \param[in] width Width of the image without padding
* \param[in] height Height of the image without padding
* \param[in] filterWidth Width of the gaussian kernel
* \param[in] filterHeight Height of the gaussian kernel
* \author jan
*/
__kernel void conv2dReLUwoLMEM(
    const __global  float * gInputImage,
    const __global float * gFilter,
    const __global float * biases,
    __global  float * gOutputImage,
    __local float * lImage,
    __local float * lFilter,
    const int width,
    const int height,
    const int depth,
    const int filterWidth,
    const int filterHeight,
    const int outputDepth
)
{
    int lengthindepth = outputDepth/get_global_size(2);
    int depthstart = lengthindepth*get_global_id(2);
    float outArr[256];
    convolutionWithoutImageCopy(gInputImage, gFilter, lFilter, width, height, depth, filterWidth, filterHeight, outputDepth, outArr);
    int inImgOffset = mad24(width,get_global_id(1),get_global_id(0));
    for(int i = depthstart; i< depthstart+lengthindepth; i++)
    {
        int layerpt = i*width*height;
        gOutputImage[layerpt+inImgOffset] = clamp(outArr[i-depthstart]+biases[layerpt+inImgOffset], 0.0f, FLT_MAX);
    }
}

/**
* \brief Convolutional Layer with Linear activation with LMEM
* \param[in] gInputImage The input image
* \param[in] gFilter Constant buffer with the filter that should be applied.
* \param[out] gOutputImage Filtered output image
* \param[in] lImage A local cache for the image of the size of the local work group enhanced by filterWidth - 1 and filterHeight -1 in width and height.
* \param[in] width Width of the image without padding
* \param[in] height Height of the image without padding
* \param[in] filterWidth Width of the gaussian kernel
* \param[in] filterHeight Height of the gaussian kernel
* \author jan
*/
__kernel void conv2dLinear(
    const __global  float * gInputImage,
    const __global float * gFilter,
    const __global float * biases,
    __global  float * gOutputImage,
    __local float * lImage,
    __local float * lFilter,
    const int width,
    const int height,
    const int depth,
    const int filterWidth,
    const int filterHeight,
    const int outputDepth
)
{
    int lengthindepth = outputDepth/get_global_size(2);
    int depthstart = lengthindepth*get_global_id(2);
    float outArr[256];
    convolutionWithImageCopy(gInputImage, gFilter, lImage, lFilter, width, height, depth, filterWidth, filterHeight, outputDepth, outArr);
    int inImgOffset = mad24(width,get_global_id(1),get_global_id(0));
    for(int i = depthstart; i< depthstart+lengthindepth; i++)
    {
        int layerpt = i*width*height;
        gOutputImage[layerpt+inImgOffset] = outArr[i-depthstart]+biases[layerpt+inImgOffset];
    }
}

/**
* \brief Convolutional Layer with Linear activation without LMEM (for stride>=filtersize)
* \param[in] gInputImage The input image
* \param[in] gFilter Constant buffer with the filter that should be applied.
* \param[out] gOutputImage Filtered output image
* \param[in] lImage A local cache for the image of the size of the local work group enhanced by filterWidth - 1 and filterHeight -1 in width and height.
* \param[in] width Width of the image without padding
* \param[in] height Height of the image without padding
* \param[in] filterWidth Width of the gaussian kernel
* \param[in] filterHeight Height of the gaussian kernel
* \author jan
*/
__kernel void conv2dLinearwoLMEM(
    const __global  float * gInputImage,
    const __global float * gFilter,
    const __global float * biases,
    __global  float * gOutputImage,
    __local float * lImage,
    __local float * lFilter,
    const int width,
    const int height,
    const int depth,
    const int filterWidth,
    const int filterHeight,
    const int outputDepth
)
{
    int lengthindepth = outputDepth/get_global_size(2);
    int depthstart = lengthindepth*get_global_id(2);
    float outArr[256];
    convolutionWithoutImageCopy(gInputImage, gFilter, lFilter, width, height, depth, filterWidth, filterHeight, outputDepth, outArr);
    int inImgOffset = mad24(width,get_global_id(1),get_global_id(0));
    for(int i = depthstart; i< depthstart+lengthindepth; i++)
    {
        int layerpt = i*width*height;
        gOutputImage[layerpt+inImgOffset] = outArr[i-depthstart]+biases[layerpt+inImgOffset];
    }
}
