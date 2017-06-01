/*


*/

/*
__kernel void conv2dwinograd2_iTrans(
    const __global float * biases,
    const __global float * gInputImage,
    const __global float * gTransformedFilters,
    __global  float * gOutputImage,
    __local float* lTransformedFilters,
    __local float* lInput,
    int activation,     // 0 linear, 1 relu
    int bias,           // 0 no bias, 1 bias
    int NperDim,        // Number of blocks per Dimensions (expected to be 1-5, for N = 1, 4, 9, 16, 25)
    int OPartitionSize,              // Number of Filters
    int O,              // Number of Filters
    int I,              // Number of Input Channels
    int bInputWidth,    // Input Buffer Width
    int bInputHeight,    // Input Buffer Height
    int numberOfWGsInX, // Number of Workgroups in X Dimension
    int cInputWidth,    // Clipped Input Width
    int cInputHeight    // Clipped Input Height
    )
{
    int superblockwh = NperDim*2+2;
    int superblocksize = superblockwh*superblockwh;
    int N = NperDim*NperDim;


    // Copy Input to Local (Only for non-strided calculation with N>1)
    // I think, here we have to decide between coalescing global memory access and no local memory bank conflicts...
    // I chose coalescing global memory access

    const int blx = NperDim*2*(get_group_id(0)%numberOfWGsInX)-1;
    const int bly = NperDim*2*(get_group_id(0)/numberOfWGsInX)-1;

    for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<superblocksize*I;i+=get_local_size(0)*get_local_size(1))
    {
        int inblock = i % (superblocksize);
        int blockid = i / (superblocksize);
        lInput[i] = getClampedFloat(gInputImage, (blockid* bInputWidth*bInputHeight), blx + (inblock % superblockwh), bly + (inblock / superblockwh), cInputWidth, cInputHeight, bInputWidth);
    }




    // Needed for Output storing (hiding gmem access latency a bit)
    int oy = bly+1+2*(get_local_id(0)/NperDim);
    int ox = blx+1+2*(get_local_id(0)%NperDim);
    int index = (get_group_id(1)*OPartitionSize+get_local_id(1))*(bInputWidth*bInputHeight)+oy*bInputWidth+ox;
    // Copy Transforms and Filters to Local

    barrier(CLK_LOCAL_MEM_FENCE);



    {
        float m[16];

        // Input Transform
        // We have N*I Input patches (4x4) to transform and N*max(I,O) threads to accomplish that


        int i = get_local_id(1)*get_local_size(0)+get_local_id(0);

        // If O>I, some threads don't perform a transformation here
        if(get_local_id(1)<I)
        {
            // Transform Input Step 1
            // Matmul from the left with:
            //    1.0, 0.0,-1.0, 0.0,
            //    0.0, 1.0, 1.0, 0.0,
            //    0.0,-1.0, 1.0, 0.0,
            //    0.0,-1.0, 0.0, 1.0
            // we can skip the 0*x
            int j = i*2 +(i/NperDim)*(superblockwh+2)+(i/N)*(superblockwh*2); // Calculate the index of the first value in overlapping inputs in local memory

            m[0]   = lInput[j]                             - lInput[j+2*superblockwh];
            m[1]   = lInput[j+1]                           - lInput[j+2*superblockwh+1];
            m[2]   = lInput[j+2]                           - lInput[j+2*superblockwh+2];
            m[3]   = lInput[j+3]                           - lInput[j+2*superblockwh+3];

            m[4]   =              lInput[j+superblockwh]   + lInput[j+2*superblockwh];
            m[5]   =              lInput[j+superblockwh+1] + lInput[j+2*superblockwh+1];
            m[6]   =              lInput[j+superblockwh+2] + lInput[j+2*superblockwh+2];
            m[7]   =              lInput[j+superblockwh+3] + lInput[j+2*superblockwh+3];

            m[8]   =             -lInput[j+superblockwh]   + lInput[j+2*superblockwh];
            m[9]   =             -lInput[j+superblockwh+1] + lInput[j+2*superblockwh+1];
            m[10]  =             -lInput[j+superblockwh+2] + lInput[j+2*superblockwh+2];
            m[11]  =             -lInput[j+superblockwh+3] + lInput[j+2*superblockwh+3];

            m[12]  =             -lInput[j+superblockwh]                                  +lInput[j+3*superblockwh];
            m[13]  =             -lInput[j+superblockwh+1]                                +lInput[j+3*superblockwh+1];
            m[14]  =             -lInput[j+superblockwh+2]                                +lInput[j+3*superblockwh+2];
            m[15]  =             -lInput[j+superblockwh+3]                                +lInput[j+3*superblockwh+3];


        }


        barrier(CLK_LOCAL_MEM_FENCE);
        // If O>I, some threads don't perform a transformation here
        if(get_local_id(1)<I)
        {
            //Transform Input Step 2
            // Matmul from the right with:
            //    1.0, 0.0, 0.0, 0.0,
            //    0.0, 1.0,-1.0,-1.0,
            //   -1.0, 1.0, 1.0, 0.0,
            //    0.0, 0.0, 0.0, 1.0
            // we can skip 0*x and identities
            float temp = m[1];
            m[0]  =  m[0]  - m[2];
            m[1]  =  m[1]  + m[2];
            m[2]  = -temp  + m[2];
            m[3]  = -temp  + m[3];

            temp = m[5];
            m[4]  =  m[4]  - m[6];
            m[5]  =  m[5]  + m[6];
            m[6]  = -temp  + m[6];
            m[7]  = -temp  + m[7];

            temp = m[9];
            m[8]  =  m[8]  - m[10];
            m[9]  =  m[9]  + m[10];
            m[10] = -temp  + m[10];
            m[11] = -temp  + m[11];

            temp = m[13];
            m[12] =  m[12] - m[14];
            m[13] =  m[13] + m[14];
            m[14] = -temp  + m[14];
            m[15] = -temp  + m[15];

            // Store in Local Memory (avoiding bank conflicts now and later)
            #pragma unroll
            for(int k = 0; k<16;k++)
            {
                lInput[k*N*I+i] = m[k];
            }
        }

        // Local memory Layout: [h,w,I,N]
    }
}


__kernel void conv2dwinograd2_mul_oTrans(
    const __global float * biases,
    const __global float * gInputImage,
    const __global float * gTransformedFilters,
    __global  float * gOutputImage,
    __local float* lTransformedFilters,
    __local float* lInput,
    int activation,     // 0 linear, 1 relu
    int bias,           // 0 no bias, 1 bias
    int NperDim,        // Number of blocks per Dimensions (expected to be 1-5, for N = 1, 4, 9, 16, 25)
    int OPartitionSize,              // Number of Filters
    int O,              // Number of Filters
    int I,              // Number of Input Channels
    int bInputWidth,    // Input Buffer Width
    int bInputHeight,    // Input Buffer Height
    int numberOfWGsInX, // Number of Workgroups in X Dimension
    int cInputWidth,    // Clipped Input Width
    int cInputHeight    // Clipped Input Height
    )
{
    // Multiplications
    // 16*I Multiplications per thread
    // If I>O, some threads don't perform computations here
    // For I>O probably not efficient in some cases... but happens rarely anyway with a 3x3 convolutional layer.
    if(get_local_id(1)<OPartitionSize)
    {
        float m[16] = {0.0};

        for(int i = 0; i<I;i++)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int j = get_local_id(1)*get_local_size(0)+get_local_id(0); j<16*OPartitionSize;j+=get_local_size(0)*get_local_size(1))
            {
                lTransformedFilters[j] = gTransformedFilters[i*O*16+(j/OPartitionSize)*(O-OPartitionSize)+get_group_id(1)*OPartitionSize+j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for(int j = 0; j<16; j++)
            {
                m[j] += lInput[j*N*I+i*N+get_local_id(0)]*lTransformedFilters[j*OPartitionSize+get_local_id(1)];

            }
        }




        // Output Transform Step 1
        // Matmul from the right with:
        //   1.0, 0.0,
        //   1.0, 1.0,
        //   1.0,-1.0,
        //   0.0, 1.0

        m[0] =  m[0]  + m[1]  + m[2];
        m[1] =  m[1]  - m[2]  + m[3];
        m[2] =  m[4]  + m[5]  + m[6];
        m[3] =  m[5]  - m[6]  + m[7];
        m[4] =  m[8]  + m[9]  + m[10];
        m[5] =  m[9]  - m[10] + m[11];
        m[6] =  m[12] + m[13] + m[14];
        m[7] =  m[13] - m[14] + m[15];


        // Output Transform Step 2
        // Matmul from the left with:
        //   1.0, 1.0, 1.0, 0.0
        //   0.0, 1.0,-1.0, 1.0
        //

        m[0] =  m[0] + m[2] + m[4];
        m[1] =  m[1] + m[3] + m[5];
        m[2] =  m[2] - m[4] + m[6];
        m[3] =  m[3] - m[5] + m[7];


        // Bias and ReLU-Activation

        if(bias)
        {
            float bias = biases[get_group_id(1)*OPartitionSize+get_local_id(1)];

            m[0] += bias;
            m[1] += bias;
            m[2] += bias;
            m[3] += bias;
        }

        if(activation)
        {

            m[0] = clamp(m[0], 0.0f, FLT_MAX);
            m[1] = clamp(m[1], 0.0f, FLT_MAX);
            m[2] = clamp(m[2], 0.0f, FLT_MAX);
            m[3] = clamp(m[3], 0.0f, FLT_MAX);
        }

        // Store Output

        gOutputImage[index] = m[0];
        gOutputImage[index+1] = m[1];
        gOutputImage[index+bInputWidth] = m[2];
        gOutputImage[index+bInputWidth+1] = m[3];
    }
}
*/
// This kernel does not work. Tried to parallize over I but then there are synchronizing issues.
/*
__kernel void conv2dwinograd2I(
    const __global float * biases,
    const __global float * gInputImage,
    const __global float * gTransformedFilters,
    __global  float * gOutputImage,
    __local float* lOutput,
    __local float* lInput,
    int activation,     // 0 linear, 1 relu
    int bias,           // 0 no bias, 1 bias
    int NperDim,        // Number of blocks per Dimensions (expected to be 1-5, for N = 1, 4, 9, 16, 25)
    int O,              // Number of Filters
    int I,              // Number of Input Channels
    int bInputWidth,    // Input Buffer Width
    int bInputHeight,    // Input Buffer Height
    int numberOfWGsInX, // Number of Workgroups in X Dimension
    int cInputWidth,    // Clipped Input Width
    int cInputHeight    // Clipped Input Height
    )
{
    int superblockwh = NperDim*2+2;
    int superblocksize = superblockwh*superblockwh;
    int N = NperDim*NperDim;

    // Initialize Output LMEM with zeros
    for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<O*N*16;i+=get_local_size(0)*get_local_size(1))
    {
        lOutput[i] = 0.0;
    }
    // Copy Input to Local (Only for non-strided calculation with N>1)
    // I think, here we have to decide between coalescing global memory access and no local memory bank conflicts...
    // I chose coalescing global memory access

    const int blx = NperDim*2*(get_group_id(1)%numberOfWGsInX)-1;
    const int bly = NperDim*2*(get_group_id(1)/numberOfWGsInX)-1;

    for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<superblocksize*I;i+=get_local_size(0)*get_local_size(1))
    {
        int inblock = i % (superblocksize);
        int blockid = i / (superblocksize);
        lInput[i] = getClampedFloat(gInputImage, (blockid* bInputWidth*bInputHeight), blx + (inblock % superblockwh), bly + (inblock / superblockwh),bInputWidth , cInputHeight, bInputWidth);
    }




    // Needed for Output storing (hiding gmem access latency a bit)
    int oy = bly+1+2*(get_local_id(0)/NperDim);
    int ox = blx+1+2*(get_local_id(0)%NperDim);



    barrier(CLK_LOCAL_MEM_FENCE);


    float m[16];

    // Input Transform
    // We have N*I Input patches (4x4) to transform and N*O threads to accomplish that
    // If I>O, some threads have to compute more than one transform (happens rarely)
    // If O<I, some threads do nothing

    int i = get_local_id(1)*get_local_size(0)+get_local_id(0);

    // Transform Input Step 1
    // Matmul from the left with:
    //    1.0, 0.0,-1.0, 0.0,
    //    0.0, 1.0, 1.0, 0.0,
    //    0.0,-1.0, 1.0, 0.0,
    //    0.0,-1.0, 0.0, 1.0
    // we can skip the 0*x
    int j = i*2 +(i/NperDim)*(superblockwh+2)+(i/N)*(superblockwh*2); // Calculate the index of the first value in overlapping inputs in local memory

    m[0]   = lInput[j]                             - lInput[j+2*superblockwh];
    m[1]   = lInput[j+1]                           - lInput[j+2*superblockwh+1];
    m[2]   = lInput[j+2]                           - lInput[j+2*superblockwh+2];
    m[3]   = lInput[j+3]                           - lInput[j+2*superblockwh+3];

    m[4]   =              lInput[j+superblockwh]   + lInput[j+2*superblockwh];
    m[5]   =              lInput[j+superblockwh+1] + lInput[j+2*superblockwh+1];
    m[6]   =              lInput[j+superblockwh+2] + lInput[j+2*superblockwh+2];
    m[7]   =              lInput[j+superblockwh+3] + lInput[j+2*superblockwh+3];

    m[8]   =             -lInput[j+superblockwh]   + lInput[j+2*superblockwh];
    m[9]   =             -lInput[j+superblockwh+1] + lInput[j+2*superblockwh+1];
    m[10]  =             -lInput[j+superblockwh+2] + lInput[j+2*superblockwh+2];
    m[11]  =             -lInput[j+superblockwh+3] + lInput[j+2*superblockwh+3];

    m[12]  =             -lInput[j+superblockwh]                                  +lInput[j+3*superblockwh];
    m[13]  =             -lInput[j+superblockwh+1]                                +lInput[j+3*superblockwh+1];
    m[14]  =             -lInput[j+superblockwh+2]                                +lInput[j+3*superblockwh+2];
    m[15]  =             -lInput[j+superblockwh+3]                                +lInput[j+3*superblockwh+3];



    float temp = m[1];
    m[0]  =  m[0]  - m[2];
    m[1]  =  m[1]  + m[2];
    m[2]  = -temp  + m[2];
    m[3]  = -temp  + m[3];

    temp = m[5];
    m[4]  =  m[4]  - m[6];
    m[5]  =  m[5]  + m[6];
    m[6]  = -temp  + m[6];
    m[7]  = -temp  + m[7];

    temp = m[9];
    m[8]  =  m[8]  - m[10];
    m[9]  =  m[9]  + m[10];
    m[10] = -temp  + m[10];
    m[11] = -temp  + m[11];

    temp = m[13];
    m[12] =  m[12] - m[14];
    m[13] =  m[13] + m[14];
    m[14] = -temp  + m[14];
    m[15] = -temp  + m[15];


    // Expecting filter layout [h,w,O,I]
    // Multiplications
    // 16*I Multiplications per thread
    for(int o = 0; o<O; o++)
    {
        #pragma unroll
        for(int j = 0; j<16; j++)
        {
            lOutput[j*N*O+o*N+get_local_id(0)] += m[j]*gTransformedFilters[O*I*j+o*I+get_local_id(1)];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<O*N; i+=get_local_size(0)*get_local_size(1))
    {

        int indexout = (i/N)*(bInputWidth*bInputHeight)+oy*bInputWidth+ox;
        // Output Transform Step 1
        // Matmul from the right with:
        //   1.0, 0.0,
        //   1.0, 1.0,
        //   1.0,-1.0,
        //   0.0, 1.0
        int index = i;
        int offset = N*O;
        float temp0,temp1,temp2,temp3;

        temp0 = lOutput[index]; index +=offset;
        temp1 = lOutput[index]; index +=offset;
        temp2 = lOutput[index]; index +=offset;
        temp3 = lOutput[index]; index +=offset;
        m[0] =  temp0  + temp1  + temp2;
        m[1] =  temp1  - temp2  + temp3;

        temp0 = lOutput[index]; index +=offset;
        temp1 = lOutput[index]; index +=offset;
        temp2 = lOutput[index]; index +=offset;
        temp3 = lOutput[index]; index +=offset;
        m[2] =  temp0  + temp1  + temp2;
        m[3] =  temp1  - temp2  + temp3;

        temp0 = lOutput[index]; index +=offset;
        temp1 = lOutput[index]; index +=offset;
        temp2 = lOutput[index]; index +=offset;
        temp3 = lOutput[index]; index +=offset;
        m[4] =  temp0  + temp1  + temp2;
        m[5] =  temp1  - temp2  + temp3;

        temp0 = lOutput[index]; index +=offset;
        temp1 = lOutput[index]; index +=offset;
        temp2 = lOutput[index]; index +=offset;
        temp3 = lOutput[index];
        m[6] =  temp0  + temp1  + temp2;
        m[7] =  temp1  - temp2  + temp3;


        // Output Transform Step 2
        // Matmul from the left with:
        //   1.0, 1.0, 1.0, 0.0
        //   0.0, 1.0,-1.0, 1.0
        //

        m[0] =  m[0] + m[2] + m[4];
        m[1] =  m[1] + m[3] + m[5];
        m[2] =  m[2] - m[4] + m[6];
        m[3] =  m[3] - m[5] + m[7];


        // Bias and Activation

        if(bias)
        {
            float bias = biases[i/N];

            m[0] += bias;
            m[1] += bias;
            m[2] += bias;
            m[3] += bias;
        }

        if(activation)
        {

            m[0] = clamp(m[0], 0.0f, FLT_MAX);
            m[1] = clamp(m[1], 0.0f, FLT_MAX);
            m[2] = clamp(m[2], 0.0f, FLT_MAX);
            m[3] = clamp(m[3], 0.0f, FLT_MAX);
        }

        // Store Output

        gOutputImage[indexout] = m[0];
        gOutputImage[indexout+1] = m[1];
        gOutputImage[indexout+bInputWidth] = m[2];
        gOutputImage[indexout+bInputWidth+1] = m[3];

    }

}
*/
