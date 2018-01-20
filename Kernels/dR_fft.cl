/**
* \brief Transforms grayscale image to frequency spectrum
* \param[in] Input, real and imaginary parts
* \param[out] Output, real and imaginary parts
* \author
*/

/* The implementation is inspired by http://www.bealto.com/gpu-fft_opencl-1.html */

#ifndef M_PI
#define M_PI 3.1416 // TODO: more exact PI
#endif

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

/******************** 2D grayscale fft *********************/
__kernel void fft(
  __global float * in,
  __global float * out,
  int p,
  int r_c
)
{
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  int w = (int) get_global_size(0)*2;
  int h = (int) get_global_size(1);
  int offset = gy*w;
  int gid = w*gy + gx;
  int imag_offset = w*h;

  int k = (gx) & (p-1);

  float2 u0;
  float2 u1;

  if (p == 1 && r_c == 1) // first fft iteration, input is real
  {
    u0.x = in[gid];
    u0.y = 0;

    u1.x = in[gid + (w/2)];
    u1.y = 0;
  }
  else // get real and imaginary part
  {
    u0.x = in[gid]; //real part
    u0.y = in[gid + imag_offset]; //imag part

    u1.x = in[gid + (w/2)];
    u1.y = in[gid + (w/2) + imag_offset];
  }

  float2 twiddle;
  float2 tmp;

  twiddle = exp_alpha( (float)(k)*(-1)*M_PI / (float)(p) );

  MUL(u1,twiddle,tmp);

  DFT2(u0,u1,tmp);

  int j = ((gx) << 1) - k;
  j += offset;

  out[j] = real(u0);
  out[j + p] = real(u1);

  out[j + imag_offset] = imag(u0);
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
  int w = (int) get_global_size(0)*2;
  int h = (int) get_global_size(1);
  int offset = gy*w;
  int gid = w*gy + gx;
  int imag_offset = w*h;

  int k = (gx) & (p-1);

  float2 u0;
  float2 u1;

  u0.x = in[gid]; //real part
  u0.y = in[gid + imag_offset]; //imag part

  u1.x = in[gid + (w/2)];
  u1.y = in[gid + (w/2) + imag_offset];

  float2 twiddle;
  float2 tmp;

  twiddle = exp_alpha( (float)(k)*M_PI / (float)(p) );

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
  __global float * out
)
{
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  int gz = get_global_id(2);
  int w = (int) get_global_size(0);
  int h = (int) get_global_size(1);
  int gid = w*h*gz + w*gy + gx;

  out[gid] = in[gid];
}

// TODO: use more optimized version
__kernel void transpose(
  __global float * in,
  __global float * out
)
{
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  int gz = get_global_id(2);
  int w = (int) get_global_size(0);
  int h = (int) get_global_size(1);
  int gid = w*h*gz + w*gy + gx;
  int t_gid = w*h*gz + w*gx + gy;

  out[t_gid] = in[gid];
}

__kernel void normalize(
  __global float *out
)
{
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  int gz = get_global_id(2);
  int w = (int) get_global_size(0);
  int h = (int) get_global_size(1);
  int gid = w*h*gz + w*gy + gx;

  out[gid] /= (w*h);
}
