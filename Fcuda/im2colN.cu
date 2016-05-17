
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void im2colN(float* data_col, const float* ori_data,
  const int height, const int width, const int channels, const int ksize,
  const int height_col, const int width_col) {

  int poolx = threadIdx.x + blockIdx.x * blockDim.x;
  int pooly = threadIdx.y + blockIdx.y * blockDim.y;
  int poolz = blockIdx.z;
  if (pooly < height_col && poolx < width_col) {

    int hstart = pooly;
    int hend = pooly + ksize;
    int wstart = poolx;
    int wend = poolx + ksize;
    int patchNum = pooly * height_col + poolx;
    ori_data += poolz * height * width;
    data_col += height_col * width_col * poolz;
    data_col += patchNum;
 
    for (int h = hstart; h < hend; h++) {
      for (int w = wstart; w < wend; w++) {
             
             *data_col = ori_data[h * width + w];
              data_col += height_col * width_col * channels;

         }
      }
    }
}


