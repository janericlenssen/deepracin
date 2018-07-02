/* FFT and Wavelet kernels */

/* The implementation is inspired by http://www.bealto.com/gpu-fft_opencl-1.html */


#define SQRT2 1.41421356237309504880f

/* Return real or imaginary component of complex number */
inline float real(float2 a){
     return a.x;
}

inline float imag(float2 a){
     return a.y;
}

/* Complex multiplication */
#define MUL(a, b, tmp) { tmp = a; a.x = tmp.x*b.x - tmp.y*b.y; a.y = tmp.x*b.y + tmp.y*b.x; }

/* Butterfly operation */
#define DFT2(a, b, tmp) { tmp = a - b; a += b; b = tmp; }

/* Return e^(i*alpha) = cos(alpha) + I*sin(alpha) */
float2 exp_alpha(float alpha)
{
  float cs,sn;
  sn = sincos(alpha,&cs);
  return (float2)(cs,sn);
}

/* For zeropadded acces on arrays */
float sampleZeroPaddedFloatFFT(
  __global float * in,
  int imag_offset,
  int width_offset,
  int x,
  int y,
  int width,
  int real_width,
  int real_height
)
{
  if(x<0 || (x + width_offset)>=(real_width)|| y<0 || y>=real_height)
  {
    return 0.0f;
  }
  else
  {
    return in[imag_offset + width_offset + (y*width + x)];
  }
}

/**
* \brief 2D grayscale out-of-place FFT algorithm
* \param[in] in Input image, grayscale or spectral. In the first half of the array are the real parts, and in the second half the imaginary parts. E.g., for 4x4 images, in the first 16 elements are the real parts, and in the seconds 16 elements are the imaginary parts. In case of a grayscale image, only one array with 16 elements is allowed as input aswell.
* \param[out] out Output of the transformation, in the first half of the array are the real parts, and in the second half the imaginary parts. E.g., for 4x4 images, in the first 16 elements are the real parts, and in the seconds 16 elements are the imaginary parts
* \param[in] p A power of two, denotes the level of the fft algorithm
* \param[in] real_width The real width of the array, used for zeropadding
* \param[in] real_height The real height of the array, used for zeropadding
* \param[in] r_c Value to determine whether input real or complex, r_c == 1 means real, 0 means complex
* \author mikail
*/
//TODO: zeropadding does not work yet
__kernel void fft(
  __global float * in,
  __global float * out,
  int p,
  int real_width,
  int real_height,
  int r_c
)
{
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  int width = (int) get_global_size(0)*2;
  int height = (int) get_global_size(1);
  int offset = gy*width;
  int gid = width*gy + gx;
  int imag_offset = width*height;

  int k = (gx) & (p-1);

  float2 u0;
  float2 u1;

  // first fft iteration, input is real
  if (p == 1 && r_c == 1)
  {
    u0.x = sampleZeroPaddedFloatFFT(in,0,0,gx,gy,width,real_width,real_height);
    u0.y = 0;

    u1.x = sampleZeroPaddedFloatFFT(in,0,(width/2),gx,gy,width,real_width,real_height);
    u1.y = 0;
  }
  else
  {
    // not first fft iteration, get real and imaginary part
    u0.x = sampleZeroPaddedFloatFFT(in,0,0,gx,gy,width,real_width,real_height);
    u0.y = sampleZeroPaddedFloatFFT(in,imag_offset,0,gx,gy,width,real_width, real_height);

    u1.x = sampleZeroPaddedFloatFFT(in,0,(width/2),gx,gy,width,real_width, real_height);
    u1.y = sampleZeroPaddedFloatFFT(in,imag_offset,(width/2),gx,gy,width,real_width, real_height);
  }

  float2 twiddle;
  float2 tmp;

  twiddle = exp_alpha( (float)(k)*(-1)*M_PI_F / (float)(p) );

  MUL(u1,twiddle,tmp);

  DFT2(u0,u1,tmp);

  int j = ((gx) << 1) - k;
  j += offset;


  out[j] = real(u0);
  if(!(gx<0 || (gx + p)>=(real_width)|| gy<0 || gy>=real_height))
    out[j + p] = real(u1);

  out[j + imag_offset] = imag(u0);
  if(!(gx<0 || (gx + p)>=(real_width)|| gy<0 || gy>=real_height))
    out[j + p + imag_offset] = imag(u1);
}

/**
* \brief 2D grayscale out-of-place inverse FFT algorithm
* \param[in] in Input image, spectral. In the first half of the array are the real parts, and in the second half the imaginary parts. E.g., for 4x4 images, in the first 16 elements are the real parts, and in the seconds 16 elements are the imaginary parts
* \param[out] out Output of the transformation, in the first half of the array are the real parts, and in the second half the imaginary parts. E.g., for 4x4 images, in the first 16 elements are the real parts, and in the seconds 16 elements are the imaginary parts
* \param[in] p A power of two, denotes the level of the fft algorithm
* \author mikail
*/
// TODO: works, but currently not integrated with sampleZeroPaddedFloatFFT
__kernel void fft_inv(
  __global float * in,
  __global float * out,
  int p
)
{
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  int width = (int) get_global_size(0)*2;
  int height = (int) get_global_size(1);
  int offset = gy*width;
  int gid = width*gy + gx;
  int imag_offset = width*height;

  int k = (gx) & (p-1);

  float2 u0;
  float2 u1;

  u0.x = in[gid]; //real part
  u0.y = in[gid + imag_offset]; //imag part

  u1.x = in[gid + (width/2)];
  u1.y = in[gid + (width/2) + imag_offset];

  float2 twiddle;
  float2 tmp;

  twiddle = exp_alpha( (float)(k)*M_PI_F / (float)(p) );

  MUL(u1,twiddle,tmp);

  DFT2(u0,u1,tmp);

  int j = ((gx) << 1) - k;
  j += offset;

  out[j] = real(u0);
  out[j + p] = real(u1);

  out[j + imag_offset] = imag(u0);
  out[j + p + imag_offset] = imag(u1);
}

/**
* \brief copy kernel which just copies an array into another array
* \param[in] in Input array
* \param[out] out Output array
* \param[in] real_width The real width of the array, used for zeropadding
* \param[in] real_height The real height of the array, used for zeropadding
* \author mikail
*/
// TODO: (1) implement these helper kernels as string in dR_nodes_fft.c ? (2) rename kernels to fft_x or x_fft, e.g. fft_copy or copy_fft ?
 __kernel void copy(
  __global float * in,
  __global float * out,
  int real_width,
  int real_height
)
{
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  int gz = get_global_id(2);
  int width = (int) get_global_size(0);
  int height = (int) get_global_size(1);
  int gid = width*height*gz + width*gy + gx;

  if(!(gx<0 || (gx)>=(real_width)|| gy<0 || gy>=real_height))
    out[gid] = in[gid];
}

/**
* \brief Transposes the input into the output
* \param[in] in Input array
* \param[out] out Output array
* \param[in] real_width The real width of the array, used for zeropadding
* \param[in] real_height The real height of the array, used for zeropadding
* \author mikail
*/
// TODO: use more optimized version
__kernel void transpose(
  __global float * in,
  __global float * out,
  int real_width,
  int real_height
)
{
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  int gz = get_global_id(2);
  int width = (int) get_global_size(0);
  int height = (int) get_global_size(1);

  if( !(gx<0 || gx>=real_width || gy<0 || gy>=real_height) )
  {
    int gid = width*height*gz + width*gy + gx;
    int t_gid = width*height*gz + width*gx + gy;
    out[t_gid] = in[gid];
  }
}

/**
* \brief Normalizes the fft
* \param[in] in Input array
* \param[out] out Output array
* \param[in] real_width The real width of the array, used for zeropadding
* \param[in] real_height The real height of the array, used for zeropadding
* \author mikail
*/
__kernel void normalizeFFT(
  __global float *out,
  int real_width,
  int real_height
)
{
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  int gz = get_global_id(2);
  int width = (int) get_global_size(0);
  int height = (int) get_global_size(1);
  int gid = width*height*gz + width*gy + gx;

  if(!(gx<0 || (gx)>=(real_width)|| gy<0 || gy>=real_height))
    out[gid] /= (width*height);
}

// for fftshift: https://www.researchgate.net/publication/271457994_High_performance_multi-dimensional_2D3D_FFT-Shift_implementation_on_Graphics_Processing_Units_GPUs
// https://www.researchgate.net/publication/278847958_CufftShift_High_performance_CUDA-accelerated_FFT-shift_library
// TODO: (1) could be done in-place. out-of-place has less temporal values (2) if only quadratic images, use only width

/*
1 2  becomes  4 3
3 4           2 1
*/
/**
* \brief Shifts the magnitudes of the fft to the middle
* \param[in] in Input array
* \param[out] out Output array
* \author mikail
*/
__kernel void shiftFFT(
  __global float *in,
  __global float *out
  )
  {
    int width = (int) get_global_size(0);
    int height = (int) get_global_size(1);
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gid = gy*width + gx;
    int imag_offset = width*height;
    int eq1 = ((width*height) + width)/2;
    int eq2 = ((width*height) - height)/2;

    if (gx < width/2)
    {
      if (gy < height/2)
      {
        // swap first quadrant with fourth
        out[gid] = in[gid+eq1];
        out[gid+eq1] = in[gid];

        out[gid + imag_offset] = in[gid + eq1 + imag_offset];
        out[gid + eq1 + imag_offset] = in[gid + imag_offset];
      }
    }
    else
    {
      if (gy < height/2)
      {
        // swap second quadrant with third
        out[gid] = in[gid + eq2];
        out[gid+eq2] = in[gid];

        out[gid + imag_offset] = in[gid + eq2 + imag_offset];
        out[gid + eq2 + imag_offset] = in[gid + imag_offset];
      }
    }
  }

  /**
  * \brief Calculates the magnitudes of a fft
  * \param[in] in Input array
  * \param[out] out Output array
  * \author mikail
  */
  __kernel void absFFT(
    __global float *in,
    __global float *out
    )
    {
      int width = (int) get_global_size(0);
      int height = (int) get_global_size(1);
      int gx = get_global_id(0);
      int gy = get_global_id(1);
      int gid = gy*width + gx;
      int imag_offset = width*height;

      float real = in[gid];
      float imag = in[gid + imag_offset];
      real *= real;
      imag *= imag;
      out[gid] = sqrt(real + imag);
    }

    /**
    * \brief Computes the 2D Haar Wavelet transformation
    * \param[in] in Input array
    * \param[out] out Output array
    * \param[in] img_width The width of the image
    * \author mikail
    */
  __kernel void haarwt(
    __global float *in,
    __global float *out,
    int img_width
    )
    {
      int width = (int) get_global_size(0);
      int gx = get_global_id(0);
      int gy = get_global_id(1);

      out[img_width*gy + gx] = in[img_width*gy + 2*gx] + in[img_width*gy + 2*gx + 1];
      out[img_width*gy + gx] /= SQRT2;
      out[img_width*gy + width + gx] = in[img_width*gy + 2*gx] - in[img_width*gy + 2*gx + 1];
      out[img_width*gy + width + gx] /= SQRT2;
    }

    /**
    * \brief Simple copy for the haarwt transformation
    * \param[in] in Input array
    * \param[out] out Output array
    * \author mikail
    */
    __kernel void hwtcopy(
     __global float * in,
     __global float * out
   )
   {
     int gx = get_global_id(0);
     int gy = get_global_id(1);
     int width = (int) get_global_size(0);
     int gid = width*gy + gx;

     out[gid] = in[gid];
   }

   /**
   * \brief Simple transpose for the haarwt transformation
   * \param[in] in Input array
   * \param[out] out Output array
   * \author mikail
   */
   __kernel void hwttranspose(
     __global float * in,
     __global float * out
   )
   {
     int gx = get_global_id(0);
     int gy = get_global_id(1);
     int width = (int) get_global_size(0);
     int gid =  width*gy + gx;
     int t_gid = width*gx + gy;

     out[t_gid] = in[gid];
   }

   /**
   * \brief Calculate all energy summands in-place
   * \param[in] in Input array, is also the output array
   * \author mikail
   */
   __kernel void wenergy2All(
     __global float * in
   )
   {
     int gx = get_global_id(0);
     int gy = get_global_id(1);
     int width = (int) get_global_size(0);
     int height = (int) get_global_size(1);
     int gid =  width*gy + gx;

     in[gid] = in[gid]*in[gid];
    }

   // inspired by https://github.com/maoshouse/OpenCL-reduction-sum
   // and https://dournac.org/info/gpu_sum_reduction
   // http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
   /**
   * \brief Parallel sum reduction
   * \param[in] in Input array
   * \param[out] out Output array
   * \param[in] reductionSums Local work groups
   * \author mikail
   */
   // not really worth doing this on GPU
  __kernel void wenergy2Sum(
    __global float *input,
    __global float *output,
    __local float *reductionSums
  )
  {
      const int globalID = (int)get_global_id(1) * (int)get_global_size(0) + (int)get_global_id(0);

      const int localID = (int)get_local_id(1)*(int)get_local_size(0) + (int)get_local_id(0);

      const int localSize = get_local_size(0)*get_local_size(1);
      const int globalSize = get_global_size(0)*get_local_size(1);
      const int width = get_global_size(0);

      const int workgroupID = globalID / localSize;

    	reductionSums[localID] = input[globalID];
      barrier(CLK_LOCAL_MEM_FENCE);

    	for(int offset = localSize / 2; offset > 0; offset /= 2)
      {
      		if(localID < offset)
          {
      			  reductionSums[localID] += reductionSums[localID + offset];
      		}
      		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all other work-items to finish previous iteration.
    	}
      // TODO: for more speedup, do this last summation on CPU
    	if(localID == 0)
      {
          output[workgroupID] = reductionSums[0];
    	}
  }

   // TODO: to sum all values, do parallel sum reduction
   // https://dournac.org/info/gpu_sum_reduction

   // Output energy on 3 levels.
   /*
   Image segments in Wavelet decimation
   1 2
   3 4
   */
   /*
   __kernel void wenergy2_3(
     __global float * in,
     __global float * feat
   )
   {
     int gx = get_global_id(0);
     int gy = get_global_id(1);
     int width = (int) get_global_size(0);
     int height = (int) get_global_size(1);
     int gid =  width*gy + gx;

     // calculate total energy


     // Level 1 Energy
     if (gx < width/2)
     {
       // left part of image
       if (gy < height/2)
       {
         // left top: 1. quadrant
         //Call again for w/2, h/2
         #if 0 // LVL 2
         // Level 2 energy
         if (gx < width/4)
         {
           // left part of image
           if (gy < height/4)
           {
             // left top: 1. quadrant lvl 2
             // Call again
             #if 1 // LVL 3
             // Level 3 energy
             if (gx < width/8)
             {
               // left part of image
               if (gy < height/8)
               {
                 // left top: 1. quadrant lvl 3
                 // No more recursions

               }
               else
               {
                 // left bottom: 3. quadrant lvl 3
               }
             }
             else
             {
               // right part of image
               if (gy < height/8)
               {
                 // right top: 2. quadrant lvl 3
               }
               else
               {
                 // right bottom: 4. quadrant lvl 3
               }
               // End of Level 3 decimation
               #endif
           }
           else
           {
             // left bottom: 3. quadrant
           }
         }
         else
         {
           // right part of image
           if (gy < height/4)
           {
             // right top: 2. quadrant lvl 2
           }
           else
           {
             // right bottom: 4. quadrant lvl 2
           }
           // End of level 2 Decimation
           #endif
       }
       else
       {
         // left bottom: 3. quadrant

       }
     }
     else
     {
       // right part of image
       if (gy < height/2)
       {
         // right top: 2. quadrant
       }
       else
       {
         // right bottom: 4. quadrant
       }
     }
     // End of level 1 decimation
   }
   */
