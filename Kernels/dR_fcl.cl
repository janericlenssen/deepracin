/*
FullyConnectedLayer implementation as matrix-vector multiplication

Visualization of Calculation:
(Input data serialized and input filters serialized per output)

(                    )   (   )   (   )
| ---- Filter 1 ---- |   | i |   |   |
|         :          |   | n |   | o |
|         :          | * | p | = | u |
|         :          |   | u |   | t |
| ---- Filter n ---- |   | t |   |   |
(                    )   (   )   (   )

First Step:
- One thread makes calculations based on a fixed sized part of one filter row
- One Workgroup contains threads using the same columns but different rows
- The required parts of input are stored in local memory to speed up reusing the values

Example:

FilterMatrix:

xxxxxxxxxxxxxxx
xxxxxxxxxxxxxxx
xxxxxxxxxxxxxxx
xxxxxxxxxxxxxxx
yyyyy11111zzzzz   1: WorkItem
xxxxx22222xxxxx   2: WorkItem
xxxxx33333xxxxx   3: WorkItem
xxxxx44444xxxxx   4: WorkItem
                  1-4: WorkGroup with 4 WorkItems reusing 5 values of input out of local memory


Note: For this to work efficiently, the filter matrix has to be stored in column major order! (memory coalescing!)


Second Step:
- One Thread per Outputvalue
- Reduce sum over blocks calculated in first step

Example for line 5 (above):
Result of WI yyyyy + result of WI 11111 + Result of WI zzzzz
*/




/**
* \brief Matrix-vector multiplication step one as described above
* \param[in] gInput The input data
* \param[in] gFilter Constant buffer with the filter that should be applied.
* \param[in] lInput A local cache for the parts of the input data
* \param[in] inputSize length of input vector (and number of cols of the filtermatrix)
* \param[in] outputSize length of output vector (and number of rows of the filtermatrix)
* \param[out] outputArr Buffer to store the partial sums
* \author jan
*/
__kernel void matrixVectorMultFirst(
    const __global  float * gInput,
    const __global float * gFilter,
    __local float* lInput,
    __global float* outputArr,
    const int numInputNeurons,
    const int numOutputNeurons,
    const int partitionSize,
    const int inputSizeX,
    const int inputSizeY,
    const int inputSizeZ
    )
{
    int lindex = 0;
    //Copy to LMEM
    int startindex = partitionSize*get_group_id(1);
    for(int i = get_local_id(0); i<partitionSize; i+=get_local_size(0))
    {
        if(startindex+i<numInputNeurons)
            lInput[i] = gInput[mapToDepthMajor(startindex+i,inputSizeX,inputSizeY,inputSizeZ)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Compute Multiplications
    float sum = 0.0;
    for(int i = 0; i< partitionSize; i++)
    {
        sum += gFilter[mad24(numOutputNeurons,startindex+i,(int)get_global_id(0))] * lInput[i];
    }
    outputArr[mad24((int)get_global_id(1),(int)get_global_size(0),(int)get_global_id(0))] = sum;
}


/**
* \brief Matrix-vector multiplication step two as described above and bias add
* \param[in,out] inputOutput Buffer with (partial) sums
* \param[in] outputSize size of the output vector
* \param[in] numberOfPartitions The number of partitions per row of the filter matrix
* \author jan
*/
__kernel void matrixVectorMultSecond(
    __global  float * inputOutput,
    __global  float * biases,
    const int useBias,
    const int activationtype,
    const int outputSize,
    const int numberOfPartitions
    )
{
    float sum = 0.0;
    int currentRow = get_global_id(0);
    for(int i = 0; i<numberOfPartitions; i++)
    {
        sum+=inputOutput[mad24(outputSize,i,currentRow)];
    }
    if(useBias==1)
        sum += biases[currentRow];
    if(activationtype==1)
        inputOutput[currentRow] = clamp(sum, 0.0f, FLT_MAX);
    else
        inputOutput[currentRow] = clamp(sum, -FLT_MAX, FLT_MAX);
}
