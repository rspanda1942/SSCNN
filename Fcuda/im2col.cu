// Copyright 2013 Yangqing Jia

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void im2col(const int n, const float * data_im,
  const int height, const int width, const int ksize,
  const int stride, const int height_col, const int width_col, float * data_col) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride;
    int w_in = w_out * stride;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        *data_col = data_im[i * width + j];
        data_col += height_col * width_col;
      }
    }
  }
}



