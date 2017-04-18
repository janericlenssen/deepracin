/**
* \brief Extracts a slice from a Buffer
* \author jan
*/
__kernel void extractSlice(
    const __global  float * gInput,
    __global float* gOutput,
    int origin0,
    int origin1,
    int origin2,
    int oshape0,
    int oshape1,
    int oshape2,
    int ishape0,
    int ishape1,
    int ishape2
    )
{
    int g0 = origin0 + (int)get_global_id(0)%oshape0;
    int g1 = origin1 + (int)(get_global_id(0)/oshape0)%oshape1;
    int g2 = origin2 + (int)(get_global_id(0)/(oshape0*oshape1));
    int gid = g2*ishape1*ishape0 + g1*ishape0 + g0;

    gOutput[get_global_id(0)] = gInput[gid];
}


/**
* \brief Crops or Pads a Buffer
* \author jan
*/
__kernel void cropOrPad(
    const __global  float * gInput,
    __global float* gOutput,
    int oshape0,
    int oshape1,
    int oshape2,
    int ishape0,
    int ishape1,
    int ishape2
    )
{
    int g0 = (int)get_global_id(0)%oshape0;
    int g1 = (int)(get_global_id(0)/oshape0)%oshape1;
    int g2 = (int)(get_global_id(0)/(oshape0*oshape1));

    gOutput[get_global_id(0)] = sampleZeroPaddedFloat(gInput,g2*ishape0*ishape1,g0,g1,ishape0,ishape1);
}
