

#ifndef kerneldef_conv2dwinograd2
#define kerneldef_conv2dwinograd2

// Notation: N - Number of 2x2 blocks in superblock, O - Number of Filters, I - Number of Input channels, NperDim - sqrt(N)
//           h - height of block (4), w - width of block (4)
//           H - height of input    , W - width of input
// Layout notation: [x,y,z] -> x major, z minor (increasing z leads to adjacent memory slots)
//
// Parallelization works as follows:
// WorkGroup for N blocks (N blocks of 4x4 Input -> 2x2 Output)
// One workgroup has OxN Work Items and computes a NperDim*2 x NperDim*2 block of outputs
// N should be chosen based on I/O (to match optimal architecture workgroup size and maxLocalMemory)

// Expected Layout for pretransformed Filters: [I,h,w,O]
// Expected Layout for gInput: [I,H,W]
// Expected Layout for lInput: [NperDim*(h/2)+2,NperDim*(w/2)+2,N,I] (no bank conflicts)

// Therefore, expected "Work Item in local index space" layout: dim0:[N], dim1:[max(I,O)]


__kernel void conv2dwinograd2(
    const __global float * biases,
    const __global float * gI,
    const __global float * gTransformedFilters,
    __global  float * gO,
    __local float* lTransformedFilters,
    __local float* lInput,
    int activation,     // 0 linear, 1 relu
    int bias,           // 0 no bias, 1 bias
    int NperDim,        // Number of blocks per Dimensions (expected to be 1-5, for N = 1, 4, 9, 16, 25)
    int OPartitionSize, // Number of Filterapplications per WorkGroup
    int INumPartitions, // Number of Inputtransforms per WorkItem
    int O,              // Number of Filters
    int I,              // Number of Input Channels
    int bW,             // Input Buffer Width
    int bH,             // Input Buffer Height
    int numWGsInX      // Number of Workgroups in X Dimension
    )
{
    int N = NperDim*NperDim;
    // x and y indices for bottom left of superblock in gI (InputImage)
    const int x_p = NperDim*2*(get_group_id(0)%numWGsInX)-1;
    const int y_p = NperDim*2*(get_group_id(0)/numWGsInX)-1;
    // offset for bottom left of 4x4 block
    int x_o = (get_local_id(0)%NperDim)*2;
    int y_o = (get_local_id(0)/NperDim)*2;
    // Output indices
    int ox = x_p+1+x_o;
    int oy = y_p+1+y_o;
    int index = (get_group_id(1)*OPartitionSize+get_local_id(1))*(bW*bH)+oy*bW+ox;
    int bc_offset = get_local_id(1)%32;


    if(get_local_id(1)*INumPartitions<I)
    {
        // x and y index offsets for block based in local_id(0)

        for(int i = 0;i< INumPartitions;i++)
        {
            float m[16];
            float temp[8];
            int z_o = ((get_local_id(1)*INumPartitions+i)* bW*bH);
            // Transform Input Step 1
            // Matmul from the left with:
            //    1.0, 0.0,-1.0, 0.0,
            //    0.0, 1.0, 1.0, 0.0,
            //    0.0,-1.0, 1.0, 0.0,
            //    0.0,-1.0, 0.0, 1.0
            // we can skip the 0*x

            temp[0]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+1, bW, bH);
            temp[1]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+1, bW, bH);
            temp[2]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+1, bW, bH);
            temp[3]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+1, bW, bH);

            temp[4]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+2, bW, bH);
            temp[5]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+2, bW, bH);
            temp[6]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+2, bW, bH);
            temp[7]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+2, bW, bH);

            m[0]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o, bW, bH)   - temp[4];
            m[1]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o, bW, bH) - temp[5];
            m[2]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o, bW, bH) - temp[6];
            m[3]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o, bW, bH) - temp[7];

            m[4]  =  temp[0] + temp[4];
            m[5]  =  temp[1] + temp[5];
            m[6]  =  temp[2] + temp[6];
            m[7]  =  temp[3] + temp[7];

            m[8]  = -temp[0] + temp[4];
            m[9]  = -temp[1] + temp[5];
            m[10] = -temp[2] + temp[6];
            m[11] = -temp[3] + temp[7];

            m[12] = -temp[0] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+3, bW, bH);
            m[13] = -temp[1] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+3, bW, bH);
            m[14] = -temp[2] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+3, bW, bH);
            m[15] = -temp[3] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+3, bW, bH);


            /* non-optimized version (but more readable)
            m[0]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o, bW, bH)   - sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+2, bW, bH);
            m[1]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o, bW, bH) - sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+2, bW, bH);
            m[2]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o, bW, bH) - sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+2, bW, bH) ;
            m[3]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o, bW, bH) - sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+2, bW, bH);

            m[4]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+1, bW, bH)   + sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+2, bW, bH);
            m[5]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+1, bW, bH) + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+2, bW, bH);
            m[6]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+1, bW, bH) + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+2, bW, bH) ;
            m[7]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+1, bW, bH) + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+2, bW, bH);
            // - y+1 y+2
            m[8]   = - sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+1, bW, bH)   + sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+2, bW, bH);
            m[9]   = - sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+1, bW, bH) + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+2, bW, bH);
            m[10]  = - sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+1, bW, bH) + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+2, bW, bH) ;
            m[11]  = - sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+1, bW, bH) + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+2, bW, bH);

            // - y+1 y+3
            m[12]  = - sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+1, bW, bH)   + sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+3, bW, bH);
            m[13]  = - sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+1, bW, bH) + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+3, bW, bH);
            m[14]  = - sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+1, bW, bH) + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+3, bW, bH) ;
            m[15]  = - sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+1, bW, bH) + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+3, bW, bH);
            */



            //Transform Input Step 2
            // Matmul from the right with:
            //    1.0, 0.0, 0.0, 0.0,
            //    0.0, 1.0,-1.0,-1.0,
            //   -1.0, 1.0, 1.0, 0.0,
            //    0.0, 0.0, 0.0, 1.0
            // we can skip 0*x and identities
            temp[0] = m[1];
            m[0]  =  m[0]     - m[2];
            m[1]  =  m[1]     + m[2];
            m[2]  = -temp[0]  + m[2];
            m[3]  = -temp[0]  + m[3];

            temp[0] = m[5];
            m[4]  =  m[4]     - m[6];
            m[5]  =  m[5]     + m[6];
            m[6]  = -temp[0]  + m[6];
            m[7]  = -temp[0]  + m[7];

            temp[0] = m[9];
            m[8]  =  m[8]     - m[10];
            m[9]  =  m[9]     + m[10];
            m[10] = -temp[0]  + m[10];
            m[11] = -temp[0]  + m[11];

            temp[0] = m[13];
            m[12] =  m[12]    - m[14];
            m[13] =  m[13]    + m[14];
            m[14] = -temp[0]  + m[14];
            m[15] = -temp[0]  + m[15];

            // Store in Local Memory (avoiding bank conflicts now and later)
            #pragma unroll
            for(int k = 0; k<16;k++)
            {
                lInput[(get_local_id(1)*INumPartitions+i)*N*16+k*N+get_local_id(0)] = m[k];
            }
            // Local memory Layout: [I,h,w,N]
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Multiplications
    // 16*I Multiplications per thread
    // gFilter Layout: [I,h,w,O]
    // If I>O, some threads don't perform computations here
    // For I>O probably not efficient in some cases... but happens rarely anyway with a 3x3 convolutional layer.
    if(get_local_id(1)<OPartitionSize&&ox<bW&&oy<bH)
    {

        float m[16] = {0.0};
        int f_ind = get_group_id(1)*OPartitionSize;
        int i_ind = 0;
        for(int i = 0; i<I;i++)
        {
            #pragma unroll
            for(int j = 0; j<16; j++)
            {
                m[j] = mad(lInput[i_ind+get_local_id(0)],gTransformedFilters[f_ind+get_local_id(1)],m[j]);
                f_ind += O;
                i_ind += N;
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

        gO[index] = m[0];
        gO[index+1] = m[1];
        gO[index+bW] = m[2];
        gO[index+bW+1] = m[3];

    }
}




//Test for large images - iterate over NperDim
__kernel void conv2dwinograd2_wide(
    const __global float * biases,
    const __global float * gI,
    const __global float * gTransformedFilters,
    __global  float * gO,
    __local float* lTransformedFilters,
    __local float* lInput,
    int activation,     // 0 linear, 1 relu
    int bias,           // 0 no bias, 1 bias
    int NperDim,        // Number of blocks per Dimensions (expected to be 1-5, for N = 1, 4, 9, 16, 25)
    int OPartitionSize, // Number of Filterapplications per WorkGroup
    int INumPartitions, // Number of Inputtransforms per WorkItem
    int O,              // Number of Filters
    int I,              // Number of Input Channels
    int bW,             // Input Buffer Width
    int bH,             // Input Buffer Height
    int numWGsInX      // Number of Workgroups in X Dimension
    )
{
    for(int j = get_local_id(1)*get_local_size(0)+get_local_id(0); j<16*OPartitionSize*I;j+=get_local_size(0)*get_local_size(1))
    {
        lTransformedFilters[j] = gTransformedFilters[j];
    }
    // lFilter layout like gFilter layout: [I,h,w,O]
    int N = NperDim*NperDim;
    // x and y indices for bottom left of superblock in gI (InputImage)
    const int x_p = NperDim*2*(get_group_id(0)%numWGsInX)-1;
    const int y_p = NperDim*2*(get_group_id(0)/numWGsInX)-1;

    for(int n = 0; n<NperDim; n++)
    {
        // offset for bottom left of 4x4 block
        int x_o = get_local_id(0)*2;
        int y_o = n*2;
        // Output indices
        int ox = x_p+1+x_o;
        int oy = y_p+1+y_o;
        int index = (get_group_id(1)*OPartitionSize+get_local_id(1))*(bW*bH)+oy*bW+ox;


        if(get_local_id(1)*INumPartitions<I)
        {
            // x and y index offsets for block based in local_id(0)

            for(int i = 0;i< INumPartitions;i++)
            {
                float m[16];
                float temp[8];
                int z_o = ((get_local_id(1)*INumPartitions+i)* bW*bH);
                // Transform Input Step 1
                // Matmul from the left with:
                //    1.0, 0.0,-1.0, 0.0,
                //    0.0, 1.0, 1.0, 0.0,
                //    0.0,-1.0, 1.0, 0.0,
                //    0.0,-1.0, 0.0, 1.0
                // we can skip the 0*x

                temp[0]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+1, bW, bH);
                temp[1]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+1, bW, bH);
                temp[2]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+1, bW, bH);
                temp[3]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+1, bW, bH);

                temp[4]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+2, bW, bH);
                temp[5]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+2, bW, bH);
                temp[6]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+2, bW, bH);
                temp[7]   = sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+2, bW, bH);

                m[0]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o, bW, bH)   - temp[4];
                m[1]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o, bW, bH) - temp[5];
                m[2]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o, bW, bH) - temp[6];
                m[3]  =  sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o, bW, bH) - temp[7];

                m[4]  =  temp[0] + temp[4];
                m[5]  =  temp[1] + temp[5];
                m[6]  =  temp[2] + temp[6];
                m[7]  =  temp[3] + temp[7];

                m[8]  = -temp[0] + temp[4];
                m[9]  = -temp[1] + temp[5];
                m[10] = -temp[2] + temp[6];
                m[11] = -temp[3] + temp[7];

                m[12] = -temp[0] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o, y_p+y_o+3, bW, bH);
                m[13] = -temp[1] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+1, y_p+y_o+3, bW, bH);
                m[14] = -temp[2] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+2, y_p+y_o+3, bW, bH);
                m[15] = -temp[3] + sampleZeroPaddedFloat(gI, z_o, x_p+x_o+3, y_p+y_o+3, bW, bH);




                //Transform Input Step 2
                // Matmul from the right with:
                //    1.0, 0.0, 0.0, 0.0,
                //    0.0, 1.0,-1.0,-1.0,
                //   -1.0, 1.0, 1.0, 0.0,
                //    0.0, 0.0, 0.0, 1.0
                // we can skip 0*x and identities
                temp[0] = m[1];
                m[0]  =  m[0]     - m[2];
                m[1]  =  m[1]     + m[2];
                m[2]  = -temp[0]  + m[2];
                m[3]  = -temp[0]  + m[3];

                temp[0] = m[5];
                m[4]  =  m[4]     - m[6];
                m[5]  =  m[5]     + m[6];
                m[6]  = -temp[0]  + m[6];
                m[7]  = -temp[0]  + m[7];

                temp[0] = m[9];
                m[8]  =  m[8]     - m[10];
                m[9]  =  m[9]     + m[10];
                m[10] = -temp[0]  + m[10];
                m[11] = -temp[0]  + m[11];

                temp[0] = m[13];
                m[12] =  m[12]    - m[14];
                m[13] =  m[13]    + m[14];
                m[14] = -temp[0]  + m[14];
                m[15] = -temp[0]  + m[15];

                // Store in Local Memory (avoiding bank conflicts now and later)
                #pragma unroll
                for(int k = 0; k<16;k++)
                {
                    lInput[(get_local_id(1)*INumPartitions+i)*NperDim*16+k*NperDim+get_local_id(0)] = m[k];
                }
                // Local memory Layout: [I,h,w,N]
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Multiplications
        // 16*I Multiplications per thread
        // gFilter Layout: [I,h,w,O]
        // If I>O, some threads don't perform computations here
        // For I>O probably not efficient in some cases... but happens rarely anyway with a 3x3 convolutional layer.

        if(get_local_id(1)<OPartitionSize&&ox<bW&&oy<bH)
        {
            float m[16] = {0.0};
            int f_ind = get_group_id(1)*OPartitionSize;
            int i_ind = 0;
            for(int i = 0; i<I;i++)
            {
                #pragma unroll
                for(int j = 0; j<16; j++)
                {
                    m[j] = mad(lInput[i_ind+get_local_id(0)],lTransformedFilters[f_ind+get_local_id(1)],m[j]);
                    f_ind += O;
                    i_ind += NperDim;
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
            gO[index] = m[0];
            gO[index+1] = m[1];
            gO[index+bW] = m[2];
            gO[index+bW+1] = m[3];

        }
    }
}


#endif
