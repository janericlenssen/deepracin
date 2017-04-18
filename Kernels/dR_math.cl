/**
* \brief Convertes RGB Image to Grayscale by averaging over depth
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \author jan
*/
__kernel void RGB2Gray(
    const __global  float * gInput,
    __global float* outputArr
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int gid = gy*width + gx;

    outputArr[gid] = (gInput[gid]+gInput[width*height+gid]+gInput[2*width*height+gid])/3;
}



/**
* \brief Adds to nodes element-wise
* \param[in] gInput1 Input Buffer 1
* \param[in] gInput2 Input Buffer 2
* \param[out] outputArr Output Image
* \author jan
*/
__kernel void elemWiseAdd(
    const __global  float * gInput1,
    const __global  float * gInput2,
    __global float* outputArr
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = gInput1[gid]+gInput2[gid];
}

/**
* \brief Subtracts input 2 from input 1 element-wise
* \param[in] gInput1 Input Buffer 1
* \param[in] gInput2 Input Buffer 2
* \param[out] outputArr Output Image
* \author jan
*/
__kernel void elemWiseSub(
    const __global  float * gInput1,
    const __global  float * gInput2,
    __global float* outputArr
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = gInput1[gid]-gInput2[gid];
}

/**
* \brief Multiplies to nodes element-wise
* \param[in] gInput1 Input Buffer 1
* \param[in] gInput2 Input Buffer 2
* \param[out] outputArr Output Image
* \author jan
*/
__kernel void elemWiseMul(
    const __global  float * gInput1,
    const __global  float * gInput2,
    __global float* outputArr
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = gInput1[gid]*gInput2[gid];
}

/**
* \brief Divides input1 through input2 element-wise
* \param[in] gInput1 Input Buffer 1
* \param[in] gInput2 Input Buffer 2
* \param[out] outputArr Output Image
* \author jan
*/
__kernel void elemWiseDiv(
    const __global  float * gInput1,
    const __global  float * gInput2,
    __global float* outputArr
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = gInput1[gid]/gInput2[gid];
}

/**
* \brief Computes input1^input2 element-wise
* \param[in] gInput1 Input Buffer 1
* \param[in] gInput2 Input Buffer 2
* \param[out] outputArr Output Image
* \author jan
*/
__kernel void elemWisePow(
    const __global  float * gInput1,
    const __global  float * gInput2,
    __global float* outputArr
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = pow(gInput1[gid],gInput2[gid]);
}


/**
* \brief Computes (Element+scalaradd)*scalarmul for every image element
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \param[in] scalaradd scalar value
* \param[in] scalarmul scalar value
* \author jan
*/
__kernel void addMulScalar(
    const __global  float * gInput,
    __global float* outputArr,
    float scalaradd,
    float scalarmul
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = (gInput[gid]+scalaradd)*scalarmul;
}


/**
* \brief Computes Element*scalarmul+scalaradd for every image element
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \param[in] scalarmul scalar value
* \param[in] scalaradd scalar value
* \author jan
*/
__kernel void madScalar(
    const __global  float * gInput,
    __global float* outputArr,
    float scalarmul,
    float scalaradd
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = gInput[gid]*scalarmul+scalaradd;
}

/**
* \brief Adds scalar value to every image element
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \param[in] scalar scalar value
* \author jan
*/
__kernel void addScalar(
    const __global  float * gInput,
    __global float* outputArr,
    float scalar
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = gInput[gid]+scalar;
}

/**
* \brief Subtracts scalar value from every buffer element
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \param[in] scalar scalar value
* \author jan
*/
__kernel void subScalar(
    const __global  float * gInput,
    __global float* outputArr,
    float scalar
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = gInput[gid]-scalar;
}

/**
* \brief Multiplies every element of buffer by a scalar value
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \param[in] scalar scalar value
* \author jan
*/
__kernel void mulScalar(
    __global  float * gInput,
    __global float* outputArr,
    float scalar
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = gInput[gid]*scalar;
}

/**
* \brief Divides every element from image through scalar value
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \param[in] scalar scalar value
* \author jan
*/
__kernel void divScalar(
    __global  float * gInput,
    __global float* outputArr,
    float scalar
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = gInput[gid]/scalar;
}

/**
* \brief Computes x^scalar for each element x in input
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \param[in] scalar scalar value
* \author jan
*/
__kernel void powScalar(
    __global  float * gInput,
    __global float* outputArr,
    float scalar
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = pow(gInput[gid],scalar);
}

/**
* \brief Computs log(x) (natural) for every image element
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \param[in] scalar scalar value
* \author jan
*/
__kernel void computeLog(
    const __global  float * gInput,
    __global float* outputArr
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = log(gInput[gid]);
}


/**
* \brief Computes e^x for every element of input image
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \author jan
*/
__kernel void computeExp(
    __global float* gInput,
    __global float* outputArr
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = exp(gInput[gid]);
}

/**
* \brief Computes sqrt(x) for every element of input buffer
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \author jan
*/
__kernel void computeSqrt(
    __global float* gInput,
    __global float* outputArr
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = sqrt(gInput[gid]);
}

/**
* \brief Fills buffer with constant
* \param[in] gInput Input Image
* \param[out] outputArr Output Image
* \author jan
*/
__kernel void fill(
    __global float* gInput,
    __global float* outputArr,
    float scalar
    )
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int gid = mad24(gz, get_global_size(0)*get_global_size(1), mad24(gy, get_global_size(0), gx));
    outputArr[gid] = scalar;
}
