// Copyright 2013 

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void im2col(const int n, const float * data_im,
  const int height, const int width, const int imagenum,
  float * data_col) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    int image_in = index/width;
    index -=image_in * width;
    data_col += height * width * image_in +  index * height ;
    data_im += height * image_in + imagenum * height * index;
    for (int i = 0; i < height; ++i) {

        *data_col = data_im[i];
        data_col += 1;
    }
  }
}



