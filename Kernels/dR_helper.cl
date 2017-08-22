/*


*/

/**
* \brief Copy the assigned parts from global to local memory
* \param[in] gImage The input image
* \param[in] x x-Coordinate of pixel to read
* \param[in] y y-Coordinate of pixel to read
* \param[in] width Width of the image
* \param[in] height Height of the image
* \author jan
*/
float sampleZeroPaddedFloat(
    const __global float * gInputImage,
    int offset,
    int x,
    int y,
    int width,
    int height
    )
{
    if(x<0||x>=width||y<0||y>=height)
        return 0.0f;
    return gInputImage[offset + (y*width+x)];

}


/**
* \brief Copy the assigned parts from global to local memory
* \param[in] gImage The input image
* \param[in] x x-Coordinate of pixel to read
* \param[in] y y-Coordinate of pixel to read
* \param[in] z z-Coordinate of pixel to read
* \param[in] width Width of the image
* \param[in] height Height of the image
* \param[in] depth Width of the image
* \author jan
*/
float getClampedFloat3d(
    const __global float * gInputImage,
    int x,
    int y,
    int z,
    int width,
    int height,
    int depth
    )
{

    if(x<0)
        x = 0;
    if(x>=width)
        x = width-1;
    if(y<0)
        y = 0;
    if(y>=height)
        y = height-1;
    if(z<0)
        z = 0;
    if(z>=depth)
        z = depth-1;
    return gInputImage[z*width*height+y*width+x];

}


/**
* \brief Copy the assigned parts from global to local memory
* \param[in] index index to be mapped
* \param[in] width width of 2d-array
* \param[in] widthRoI roi of 2darray
* \author jan
*/
int mapToRoI(
    int index,
    int width,
    int widthRoI,
    int height,
    int heightRoI,
    int depth,
    int depthRoI
    )
{
    // Transform to index for depth minor ([y,x,z]) layout
    int z = index%depthRoI;
    int x = (index/depthRoI)%widthRoI;
    int y = ((index/depthRoI)/widthRoI);
    return z*width*height+y*width+x;
}

/**
* \brief Maps index from [w,h,d] to [d,w,d]
* \param[in] index index to be mapped
* \param[in] width width of dd-array
* \author jan
*/
int mapToDepthMajor(
    int index,
    int width,
    int height,
    int depth
    )
{
    int z = index%depth;
    int x = (index/depth)%width;
    int y = ((index/depth)/width);
    return z*width*height+y*width+x;
}


/**
* \brief Resolves a specified region of interest (copies block of data)
* \param[in] gInput input opencl buffer
* \param[in] outputArr output opencl buffer
* \param[in] width width of original buffer
* \param[in] height height of original buffer
* \param[in] origin originpoint of RoI in original buffer
* \author jan
*/
__kernel void resolveRoI(
    const __global  float * gInput,
    __global float* outputArr,
    int width,
    int height,
    int depth,
    int originX,
    int originY,
    int originZ
    )
{
    int x = originX+get_global_id(0);
    int y = originY+get_global_id(1);
    int z = originZ+get_global_id(2);
    int indexout = get_global_id(0) + get_global_id(1)*get_global_size(0) + get_global_id(2)*get_global_size(0)*get_global_size(1);
    // Transform to index for depth major layout
    outputArr[indexout] = getClampedFloat3d(gInput,x,y,z,width,height,depth);
}


/**
* \brief Resolves a specified region of interest (copies block of data)
* \param[in] gInput input opencl buffer
* \param[in] outputArr output opencl buffer
* \param[in] width width of original buffer
* \param[in] height height of original buffer
* \param[in] origin originpoint of RoI in original buffer
* \author jan
*/
__kernel void resolveRoIIndexBuffer(
    const __global  float * gInput,
    __global float* outputArr,
    __global float* indexBuffer,
    int index,
    int width,
    int height,
    int depth
    )
{
    float originX = indexBuffer[index];
    float originY = indexBuffer[index+1];
    int x = originX+get_global_id(0);
    int y = originY+get_global_id(1);
    int z = get_global_id(2);
    int indexout = get_global_id(0) + get_global_id(1)*get_global_size(0) + get_global_id(2)*get_global_size(0)*get_global_size(1);
    // Transform to index for depth major layout
    outputArr[indexout] = getClampedFloat3d(gInput,x,y,z,width,height,depth);
}


/**
* \brief Creates Confidence(Label!=0) Image out of Softmax Distribution over 3 Classes per Pixel
* \param[in] gInput Softmax Distribution per Pixel
* \param[out] outputArr Buffer in which class labels are stored
* \author jan
*/
__kernel void toLabelImage3to2ClassesConf(
    const __global  float * gInput,
    __global float* outputArr,
    float class0offset,
    float class1offset,
    float class2offset
    )
{
    int index = get_global_id(1)*get_global_size(0)+get_global_id(0);
    int widthheight = get_global_size(0)* get_global_size(1);
    float prob0 = gInput[index] +class0offset;
    float prob1 = gInput[widthheight + index]+class1offset;
    float prob2 = gInput[widthheight*2 + index]+class2offset;
    if(prob1>prob2)
        outputArr[index] = prob1;//prob1/(prob1+prob0);
    else
        outputArr[index] = prob2;//prob2/(prob2+prob0);
}


/**
* \brief Creates Confidence(Label=1) Image out of Softmax Distribution over 2 Classes per Pixel
* \param[in] gInput Softmax Distribution per Pixel
* \param[out] outputArr Buffer in which class labels are stored
* \author jan
*/
__kernel void toLabelImage2ClassesConf(
    const __global  float * gInput,
    __global float* outputArr,
    float class0offset,
    float class1offset
    )
{
    int index = get_global_id(1)*get_global_size(0)+get_global_id(0);
    int widthheight = get_global_size(0)* get_global_size(1);
    float prob0 = gInput[index] + class0offset;
    float prob1 = gInput[widthheight + index] + class1offset;
    outputArr[index] = prob1/(prob1+prob0);
}

/**
* \brief Creates Binary Image (Label=0 => 0 / Label!=0 => 1) out of Softmax Distribution over 3 Classes per Pixel
* \param[in] gInput Softmax Distribution per Pixel
* \param[out] outputArr Buffer in which class labels are stored
* \author jan
*/
__kernel void toLabelImage3to2ClassesBin(
    const __global  float * gInput,
    __global float* outputArr,
    float class0offset,
    float class1offset,
    float class2offset
    )
{
    int index = get_global_id(1)*get_global_size(0)+get_global_id(0);
    int widthheight = get_global_size(0)* get_global_size(1);
    float prob0 = gInput[index] + class0offset;
    float prob1 = gInput[widthheight + index] + class1offset;
    float prob2 = gInput[widthheight*2 + index] + class2offset;
    if(prob1>prob0||prob2>prob0)
        outputArr[index] = 1.0f;
    else
        outputArr[index] = 0.0f;
}

/**
* \brief Creates Binary Image out of Softmax Distribution over 2 Classes per Pixel
* \param[in] gInput Softmax Distribution per Pixel
* \param[out] outputArr Buffer in which class labels are stored
* \author jan
*/
__kernel void toLabelImage2ClassesBin(
    const __global  float * gInput,
    __global float* outputArr,
    float class0offset,
    float class1offset
    )
{
    int index = get_global_id(1)*get_global_size(0)+get_global_id(0);
    int widthheight = get_global_size(0)* get_global_size(1);
    float prob0 = gInput[index] + class0offset;
    float prob1 = gInput[widthheight + index] + class1offset;
    if(prob1>prob0)
        outputArr[index] = 1.0f;
    else
        outputArr[index] = 0.0f;
}



/**
* \brief
* \details Data will be reduced from sumInput to lsum and then to sumResults/maxResults.
* This needs to be repeated until only one value is left in minResults[0]/maxResults[0]
* this is the case after only one workgroup gets submitted to the GPU.
* Alternatively all results in minResults/maxResults needs to be reduced to one result on the CPU.
* \warning The work group size must be a power of two
* \param[in] sumInput Input for the sum calculation.
* \param[out] sumResults size must be [get_num_groups(0) * get_num_groups(1)], so that each workgroup can calculate one sum.
* \param[in] channel Only the channel of the input int4 as specified by channel is processed.
* \param[in] gWidth Width of the image without padding
* \param[in] gHeight Height of the image without padding
* \param[in] gidOfLastValidElement ID of the last element that should be processed
* \author pascal
*/
__kernel void getAvg(
    __global float * sumInput,
    __global float * sumResults,
    __local float * lsum,
    int gWidth,
    int gHeight,
    int gidOfLastValidElement
    )
{
    // global position:
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, gWidth*gHeight, mad24(gy, gWidth, gx));

    //local position:
    int lid = mad24((int)get_local_id(1), (int)get_local_size(0), (int)get_local_id(0));

    int nActiveThreads = (int)get_local_size(0) * (int)get_local_size(1); // Total number of active threads (eg 256)
    int halfPoint = 0;

    //Initialize with neutral element
    lsum[lid] = 0.0f;
    gidOfLastValidElement = gidOfLastValidElement+((int)get_global_id(2)*(int) get_global_size(0)*(int)get_global_size(1));

    if (! (gx >= gWidth || gy >= gHeight || gid > gidOfLastValidElement))
    {
        //read to local memory
        lsum[lid] = sumInput[gid];
    }

    //wait for all threads to read one value of the (e.g.) 256 values
    barrier(CLK_LOCAL_MEM_FENCE);

    while(nActiveThreads > 1)
    {
        halfPoint = (nActiveThreads >> 1);	// divide by two
        // only the first half of the threads will be active.
        // in this round the first half of threads reads what the last half of threads calculated in the previous round
        // it compares the lmax[lid + halfPoint] value with the value lmax[lid]
        // continues until only one thread is left (all other threads are still running but inactive)
        if (lid < halfPoint)
        {
            // Get the shared value stored by another thread (of the last half) then compare the two values and keep the lesser one
            lsum[lid] += lsum[lid + halfPoint];

        } //if

        barrier(CLK_LOCAL_MEM_FENCE);

        nActiveThreads = halfPoint;	//only first half stays active
    } //while(nActiveThreads > 1)

    // At this point in time, thread zero has the min, max of the work group
    // It's time for thread zero to write it's final results.
    // Note that the address structure of the results is different, because
    // there is only one value for every work group.

    if (lid == 0)
    {
        gid = mad24((int)get_group_id(2), (int)(get_num_groups(0)*get_num_groups(1)), mad24((int)get_group_id(1), (int)get_num_groups(0), (int)get_group_id(0)));
        sumResults[gid] = lsum[0];
    }
} //getAvg



/**
* \brief
* \details Data will be reduced from sumInput to lsum and then to sumResults/maxResults.
* This needs to be repeated until only one value is left in minResults[0]/maxResults[0]
* this is the case after only one workgroup gets submitted to the GPU.
* Alternatively all results in minResults/maxResults needs to be reduced to one result on the CPU.
* \warning The work group size must be a power of two
* \param[in] sumInput Input for the sum calculation.
* \param[out] sumResults size must be [get_num_groups(0) * get_num_groups(1)], so that each workgroup can calculate one sum.
* \param[in] channel Only the channel of the input int4 as specified by channel is processed.
* \param[in] gWidth Width of the image without padding
* \param[in] gHeight Height of the image without padding
* \param[in] gidOfLastValidElement ID of the last element that should be processed
* \author pascal, jan
*/
__kernel void getSumDev(
    __global float * sumInput,
    __global float * sumResults,
    __local float * lsum,
    int gWidth,
    int gHeight,
    int gidOfLastValidElement,
    float avg,
    int first
    )
{
    // global position:
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, gHeight*gWidth, mad24(gy, gWidth, gx));

    //local position:
    int lid = mad24((int)get_local_id(1), (int)get_local_size(0), (int)get_local_id(0));

    int nActiveThreads = (int)get_local_size(0) * (int)get_local_size(1); // Total number of active threads (eg 256)
    int halfPoint = 0;

    //Initialize with neutral element
    lsum[lid] = 0.0f;


    if (! (gx >= gWidth || gy >= gHeight || gid > gidOfLastValidElement))
    {
        //read to local memory
        if(first==1)
        {
            float dif = sumInput[gid]-avg;
            lsum[lid] = dif*dif;
        }
        else
        {
            lsum[lid] = sumInput[gid];
        }
    }

    //wait for all threads to read one value of the (e.g.) 256 values
    barrier(CLK_LOCAL_MEM_FENCE);

    while(nActiveThreads > 1)
    {
        halfPoint = (nActiveThreads >> 1);	// divide by two
        // only the first half of the threads will be active.
        // in this round the first half of threads reads what the last half of threads calculated in the previous round
        // it compares the lmax[lid + halfPoint] value with the value lmax[lid]
        // continues until only one thread is left (all other threads are still running but inactive)
        if (lid < halfPoint)
        {
            // Get the shared value stored by another thread (of the last half) then compare the two values and keep the lesser one
            lsum[lid] += lsum[lid + halfPoint];

        } //if

        barrier(CLK_LOCAL_MEM_FENCE);

        nActiveThreads = halfPoint;	//only first half stays active
    } //while(nActiveThreads > 1)

    // At this point in time, thread zero has the min, max of the work group
    // It's time for thread zero to write it's final results.
    // Note that the address structure of the results is different, because
    // there is only one value for every work group.

    if (lid == 0)
    {
        gid = mad24((int)get_group_id(2), (int)(get_num_groups(0)*get_num_groups(1)), mad24((int)get_group_id(1), (int)get_num_groups(0), (int)get_group_id(0)));
        sumResults[gid] = lsum[0];
    }
} //getAvg
