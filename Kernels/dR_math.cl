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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
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
    int gid = mad24(gz, (int)get_global_size(0)*(int)get_global_size(1), mad24(gy, (int)get_global_size(0), gx));
    outputArr[gid] = scalar;
}

/* --------------FFT: Out of place-------------- */

/* complex numbers definitions https://stackoverflow.com/questions/10016125/complex-number-support-in-opencl */
#ifndef M_PI
#define M_PI 3.1416
#endif
/* Return Real (Imaginary) component of complex number */
inline float real(float2 a){
     return a.x;
}
inline float imag(float2 a){
     return a.y;
}

/* Complex multiplication */
#define MUL(a, b, tmp) { tmp = a; a.x = tmp.x*b.x - tmp.y*b.y; a.y = tmp.x*b.y + tmp.y*b.x; }
/*
inline cfloat  cmult(float2 a, float2 b){
    return (float2)( a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
*/

/* Butterfly operation */
#define DFT2( a, b, tmp) { tmp = a - b; a += b; b = tmp; }

/* Return cos(alpha)+I*sin(alpha) */
float2 exp_alpha(float alpha)
{
  float cs,sn;
  sn = sincos(alpha,&cs);
  return (float2)(cs,sn);
}

/* Dimensions of input */
/*
#define X 4
#define Y 4
#define Z 3 // 3 for RGB
*/
//static const unsigned char BitReverseTable[4] = { 0x0, 0x2, 0x1, 0x3 };

__kernel void fft(
    const __global float * gInput,
    __global float2 * outputArr
    )
{
    /* For RGB do */
    /*  Bit reverse rows  */
    /*  FFT on rows */
    /*  Bit reverse columns */
    /*  FFT on columns */

    /* Input size N, for 1D FFT N/2 threads, each thread doing one DFT2
     * Input size X*Y*Z, for Z times 2D FFT, Z = 3 for RGB
     * Output size X*Y*6
     * 1) rows: Y*(X/2)*Z work items at once
     * 2) columns: X*(Y/2)*Z work items at once
     */

    /* Each work item has a three dimensional identifier */
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int gz = (int) get_global_id(2);
    int width = (int) get_global_size(0);
    int height = (int) get_global_size(1);
    int depth = (int) get_global_size(2);
    int gid = height*(width/2)*gz + (width/2)*gy + gx;

    /* For Z = 1,2,3 store rows in bitreverse order */
    /*
    int lookup = BitReverseTable[gid - (gy*width)];
    int newindex = gy*width + lookup;

    outputArr[ newindex ] = gInput[gid];
    */

    /* first stage: rows */
    /* Caluclate log2(X) */

    //int log_x = log2((int)get_global_size(0));

    /*
    for (int stage = 1; stage <= log_x; stage++ )
    {
      int m = pow(2,stage);
      float2 twiddle_base = exp_alpha((2*M_PI)/m);

      float2 top = outputArr[gid];
      float2 bottom = outputArr[gid + m/2];
    }
    */

    /* rows, insert barrier between invocations */
    for(int p = 1; p <= width; p *= 2)
    {
        int k = gid & (p-1); // index in input sequence, in 0..P-1

        float2 u0 = gInput[gid];
        float2 u1 = gInput[gid+t];

        float2 twiddle = 1; //pow_theta(k,p);

        float2 tmp;

        MUL(u1,twiddle,tmp);

        DFT2(u0,u1,tmp);

        int j = (gid<<1) - k; // = ((i-k)<<1)+k

        out[j] = u0;

        out[j+p] = u1;
    }

    /* inser barrier here */

    /* collumns may need different work items */

    /*
    for(int p = 1; p <= width; p *= 2)
    {
      fft_radix2_second_stage(outputArr, p)
    }
    */

}

/* helper function */

// void fft_radix2_first_stage(__global const float * in,__global float2 * out, int p)
// {
//   /* id of current thread */
//   int i = gx;
//   /* nr of all threads */
//   int t = width;
//
//   int k = i & (p-1); // index in input sequence, in 0..P-1
//
//   float2 u0 = in[i];
//   float2 u1 = in[i+t];
//
//   float2 twiddle = pow_theta(k,p);
//
//   float2 tmp;
//
//   MUL(u1,twiddle,tmp);
//
//   DFT2(u0,u1,tmp);
//
//   int j = (i<<1) - k; // = ((i-k)<<1)+k
//
//   out[j] = u0;
//
//   out[j+p] = u1;
// }

/* helper function */
void fft_radix2_second_stage(__global float2 * out, int p)
{
  /* id of current thread */
  int i = gx;
  /* nr of all threads */
  int t = width;

  int k = i & (p-1); // index in input sequence, in 0..P-1

  float2 u0 = out[i];
  float2 u1 = out[i+t];

  float2 twiddle = pow_theta(k,p);

  float2 tmp;

  MUL(u1,twiddle,tmp);

  DFT2(u0,u1,tmp);

  int j = (i<<1) - k; // = ((i-k)<<1)+k

  out[j] = u0;

  out[j+p] = u1;
}
