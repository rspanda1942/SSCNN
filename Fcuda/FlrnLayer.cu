

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// -------------------------------------------------------------------------------------

__global__ void LRNFillScale(const int nthreads, const float* in,
    const int num, const int channels, const int height,
    const int width, const int size, const float alpha_over_size,
    float* scale) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    in += offset;
    scale += offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    float accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_scale += in[head * step] * in[head * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
  }
}

//------------------------------------------------------------------------------

__global__ void LRNComputeOutput(const int nthreads, const float* in,
    const float* scale, const float negative_beta, float* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}




//-------------------------------------------------------------------------------------

__global__ void LRNComputeDiff(const int nthreads, const float* bottom_data,
    const float* top_data, const float* scale, const float* top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const float negative_beta,
    const float cache_ratio,
    float* bottom_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    bottom_data += offset;
    top_data += offset;
    scale += offset;
    top_diff += offset;
    bottom_diff += offset;
    int head = 0;
    int pre_pad = size - (size + 1) / 2;
    int post_pad = size - pre_pad - 1;
    float accum_ratio = 0;
    // accumulate values
    while (head < post_pad) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}


