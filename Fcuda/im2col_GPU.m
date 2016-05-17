function coldata = im2col_GPU(x, cudakernel, height_col, width_col, ksize, channels) 

 height = size(x,1);
 width = size(x,2);


 num_kernels = channels * height_col * width_col;   %The using  CUDA thread number
 CUDA_NUM_THREADS = 1024;
 GET_BLOCKS = (num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

 cudakernel.ThreadBlockSize = [CUDA_NUM_THREADS];
 cudakernel.GridSize = [floor(GET_BLOCKS)];

  coldata = gpuArray.zeros(num_kernels/channels, ksize * ksize * channels, 'single') ;
 
  coldata = feval(cudakernel,num_kernels, x, height, width, ksize, 1, height_col, width_col, coldata);

end
