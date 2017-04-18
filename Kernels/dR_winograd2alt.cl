// Input Transform (Matrix B)
/*float inputTransform2x2[4*4] = { 1.0, 0.0, 0.0, 0.0,
                                  0.0,-0.5, 0.5,-1.0,
                                 -1.0, 0.5, 0.5, 0.0,
                                  0.0, 0.0, 0.0, 1.0};

*/
// Output Transform (Matrix A)
/*float outputTransform2x2[4*2] = { 1.0, 0.0,
                                   1.0,-1.0,
                                   1.0, 1.0,
                                   0.0, 1.0};
*/


// Notation: N - Number of blocks, O - Number of Filters, I - Number of Input channels
//           h - height of block (4), w - width of block (4)
// Layout notation: [x,y,z] -> x major, z minor (increasing z leads to adjacent memory slots)
//
// Parallelization works as follows:
// WorkGroup for N superlocks (N blocks of 4x4 Input -> 2x2 Output)
// One workgroup has 4xOxN Work Items and computes a sqrt(N)*2 x sqrt(N)*2 block of outputs
// N should be chosen based on O*4 (to match optimal architecture workgroup size and maxLocalMemory)

// Expected Layout for Filters: [I,h,w,O]
// Expected Layout for gInput: [I,H,W]
// Expected Layout for lInput: [
// Expected Layout for lInput: [N,I,h,w] (coalescing memory access when looping over h in one Work Item)
// Therefore, expected "Work Item in local index space" layout: dim0:[4,O], dim1:[N]

__kernel void conv2dwinograd2(
    const __global float * biases,
    const __global float * gInputImage,
    const __global float * gTransformsAndFilters,   // Layout: 1.(Adress 0) 4x4 InputTransform, 2.(Adress 16) 4x2 Outputtransform, 3.(Adress 24) Filters
    __global  float * gOutputImage,
    __local float* lTransformsAndFilters,           // Layout: 1.(Adress 0) 4x4 InputTransform, 2.(Adress 16) 4x2 Outputtransform, 3.(Adress 24) Filters
    __local float* lInput1,
    __local float* lInput2,
    int NperDim, // Number of blocks per Dimensions (expected to be 1-5, for N = 1, 4, 9, 16, 25)
    int O, // Number of Filters
    int I, // Number of Input Channels
    int bInputWidth, // Input Buffer Width
    int bInputHeight, // Input Buffer Height
    int cInputWidth, // Clipped Input Width
    int cInputHeight // Clipped Input Height
    )
{
    int blockwh = NperDim*2+2;
    int blocksize = blockwh*blockwh;
    int N = NperDim*NperDim;

    // Copy Transforms and Filters to Local
    for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<24+4*4*I*O;i+=get_local_size(0)*get_local_size(1))
    {
        lTransformsAndFilters[i] = gTransformsAndFilters[i];
    }

    // Copy Input to Local (Only for non-strided calculation with N>1)
    int numberOfWGsInX = inputWidth/2*NperDim;
    const int blx = NperDim*2*(get_group_id(0)%numberOfWGsInX)-1;
    const int bly = NperDim*2*(get_group_id(0)/numberOfWGsInX)-1;
    for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0); i<blocksize*I;i+=get_local_size(0)*get_local_size(1))
    {
        int inblock = i % (blocksize);
        int blockid = i / (blocksize);
        lInput1[i] = getClampedFloat(gInputImage, (blockid* blocksize), blx + (inblock % blockwh), bly + (inblock / blockwh), cInputWidth, cInputHeight, bInputWidth);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Transform Input Step 1
    // Matmul from the left with:
    /*    1.0, 0.0, 0.0, 0.0,
          0.0,-0.5, 0.5,-1.0,
         -1.0, 0.5, 0.5, 0.0,
          0.0, 0.0, 0.0, 1.0 */
    // We have N*I Input patches (4x4) to transform and N*4*O threads to accomplish that
    // Each thread one result column of one matrix (so we can skip the 0*x)
    // After this step: Memory Layout [I,N,h,w]
    // Most columns are computed twice here (due to overlapping), maybe this can be avoided?
    // If I>O, some threads have to compute more than one column (happens rarely)
    // If O<I, some threads do nothing

    for(int i = get_local_id(1)*get_local_size(0)+get_local_id(0);i<4*N*I;i+=get_local_size(0)*get_local_size(1))
    {
        int j = i-(i/4)*2+(i/8)*(blockwh+2); // Calculate the index of the first vector value in overlapping inputs in local memory
        lInput2[i]                = lInput1[j];
        lInput2[i+I*N*4]   = -0.5*lInput1[j+blockwh]+0.5*lInput1[j+2*blockwh]-lInput1[j+3*blockwh];
        lInput2[i+I*N*8]   = -lInput1[j] + 0.5*lInput1[j+blockwh] + 0.5*lInput[j+2*blockwh];
        lInput2[i+I*N*16]  = lInput1[j+3*blockwh];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //Transform Input Step 2
    // Matmul from the right with:
    /*    1.0, 0.0,-1.0, 0.0,
          0.0,-0.5, 0.5, 0.0,
          0.0, 0.5, 0.5, 0.0,
          0.0,-1.0, 0.0, 1.0 */
    //Each thread one row of one matrix (so we can skip 0*x)
    // Here, too many bank conflicts happen, do not know if they can be avoided
    for(int i = (get_local_id(1)*get_local_size(0)+get_local_id(0))*4;i<4*N*I*4;i+=get_local_size(0)*get_local_size(1)*4)
    {
        lInput1[i]      = lInput2[i];
        lInput1[i+1]    = -0.5*lInput2[i+1] + 0.5*lInput2[i+2]-lInput2[i+3];
        lInput1[i+2]    = -lInput2[i] + 0.5*lInput2[i+1] + 0.5*lInput2[i+2];
        lInput1[i+3]    = lInput2[i+3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    // Multiplications
    // 4*I Multiplications per thread
    // Each thread one row (can directly be used to compute one row of output transform step 1, without using local memory)
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;
    for(int i = 0; i<I;i++)
    {
        int j = i*16*N+get_local_id(0)+get_local_id(1)*16;
        sum1 += lInput1[j]*lTransformsAndFilters[get_local_id(0)];
        sum2 += lInput1[j+1]*lTransformsAndFilters[get_local_id(0)+1];
        sum3 += lInput1[j+2]*lTransformsAndFilters[get_local_id(0)+2];
        sum4 += lInput1[j+3]*lTransformsAndFilters[get_local_id(0)+3];
    }





    // Output Transform Step 1
    // Matmul from the right with:
    /*   1.0, 0.0,
         1.0,-1.0,
         1.0, 1.0,
         0.0, 1.0 */
    int i = 2*get_local_id(1)+get_local_id(0)*N*2;
    lInput2[i]   = sum1 + sum2 + sum3;
    lInput2[i+1] = -sum2 + sum3 + sum4;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Output Transform Step 2
    // Matmul from the left with:
    /*   1.0, 1.0, 1.0, 0.0
         0.0,-1.0, 1.0, 1.0 */
    //
    i = 2*get_local_id(1)+(get_local_id(0)/2)*N*2+get_local_id(0)%2;
    result1 = lInput2[i] + lInput2[i+N*2] + lInput2[i+N*4];
    result2 = - lInput2[i+N*2] + lInput2[i+N*4] + lInput2[i+N*6];



    // Bias and Activation
    float bias = biases[get_local_id(0)/4];
    result1 += bias;
    result2 += bias;



}
