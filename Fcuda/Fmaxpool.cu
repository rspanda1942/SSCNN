

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// -------------------------------------------------------------------------------------
__global__ void FMaxPoolForward(const float* ori_data, float* pool_data, int* indice_data,
    const int num, const int channels, 
    const int height, const int width, 
    const int pooled_height, const int pooled_width,
    const int poolsize, const int poolstrike) {

  int poolx = threadIdx.x + blockIdx.x * blockDim.x;
  int pooly = threadIdx.y + blockIdx.y * blockDim.y;
  int poolz = blockIdx.z;

  if (pooly < pooled_height && poolx < pooled_width) {

    int hstart = pooly * poolstrike;
    int hend = hstart + poolsize;
    int wstart = poolx * poolstrike;
    int wend = wstart + poolsize;
    float maxval = -9999;
    int indice = 0;
    ori_data += poolz * height * width;
    for (int h = hstart; h < hend; h++) {
      for (int w = wstart; w < wend; w++) {
         if( ori_data[h * width + w] > maxval){
             maxval = ori_data[h * width + w];
             indice = (h - hstart) * poolsize + w - wstart ;
         }
      }
    }
    pool_data[poolx + pooly * pooled_width + poolz * pooled_height * pooled_width] = maxval;
    indice_data[poolx + pooly * pooled_width + poolz * pooled_height * pooled_width] = indice;
  } 
}


//---------------------------------------------------------------------------------------


__global__ void FMaxPoolBackward(float* reverse_data, float* pool_data, int* indice_data,
    const int num, const int channels, 
    const int height, const int width, 
    const int pooled_height, const int pooled_width,
    const int poolsize, const int poolstrike) {

  int poolx = threadIdx.x + blockIdx.x * blockDim.x;
  int pooly = threadIdx.y + blockIdx.y * blockDim.y;
  int poolz = blockIdx.z;

  if (pooly < pooled_height && poolx < pooled_width) {

    float maxdata = pool_data[poolx + pooly * pooled_width + poolz * pooled_height * pooled_width] ;
    int posit = indice_data[poolx + pooly * pooled_width + poolz * pooled_height * pooled_width] ;

    int hstart = pooly * poolstrike;
    int wstart = poolx * poolstrike;

    int woffset = posit % poolsize;
    int hoffset = int(posit / poolsize);

    int h = hstart + hoffset;
    int w = wstart + woffset;
    reverse_data += poolz * height * width;
    reverse_data[h * width + w] = maxdata;

  }  
}

//--------------------------------------------------------------------------------------------

__global__ void FMaxPoolForwardFix(const float* ori_data, float* pool_data, int* indice_data,
    const int num, const int channels, 
    const int height, const int width, 
    const int pooled_height, const int pooled_width,
    const int poolsize, const int poolstrike) {

  int poolx = threadIdx.x + blockIdx.x * blockDim.x;
  int pooly = threadIdx.y + blockIdx.y * blockDim.y;
  int poolz = blockIdx.z;

  if (pooly < pooled_height && poolx < pooled_width) {

    int posit = indice_data[poolx + pooly * pooled_width + poolz * pooled_height * pooled_width] ;

    int hstart = pooly * poolstrike;
    int wstart = poolx * poolstrike;

    int woffset = posit % poolsize;
    int hoffset = int(posit / poolsize);

    int h = hstart + hoffset;
    int w = wstart + woffset;
    ori_data += poolz * height * width;
    pool_data[poolx + pooly * pooled_width + poolz * pooled_height * pooled_width] = ori_data[h * width + w];

  }  
}
