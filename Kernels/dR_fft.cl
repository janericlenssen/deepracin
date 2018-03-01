/**
* \brief Transforms grayscale image to frequency spectrum
* \param[in] Input, real and imaginary parts
* \param[out] Output, real and imaginary parts
* \author
*/

/* The implementation is inspired by http://www.bealto.com/gpu-fft_opencl-1.html */

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
#define DFT2( a, b, tmp) { tmp = a - b; a += b; b = tmp; }

/* Return e^(i*alpha) = cos(alpha) + I*sin(alpha) */
float2 exp_alpha(float alpha)
{
  float cs,sn;
  sn = sincos(alpha,&cs);
  return (float2)(cs,sn);
}

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

/******************** 2D grayscale fft *********************/
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

  if (p == 1 && r_c == 1) // first fft iteration, input is real
  {
    //sampleZeroPaddedFloatFFT(in,imag_offset,width_offset,gx,gy,width,real_width, real_height);
    //TODO: check if really needed for upper element of butterfly
    u0.x = sampleZeroPaddedFloatFFT(in,0,0,gx,gy,width,real_width,real_height);
    u0.y = 0;

    u1.x = sampleZeroPaddedFloatFFT(in,0,(width/2),gx,gy,width,real_width,real_height);
    u1.y = 0;
  }
  else // get real and imaginary part
  {
    u0.x = sampleZeroPaddedFloatFFT(in,0,0,gx,gy,width,real_width,real_height); //real part
    u0.y = sampleZeroPaddedFloatFFT(in,imag_offset,0,gx,gy,width,real_width, real_height); //imag part

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

// TODO: could do one kernel for forward and inv fft
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
