function coldata = col2im_GPU(x, cudakernel, height, width,  imagenum) 

 num_kernels = width * imagenum;   %The using  CUDA thread number
 CUDA_NUM_THREADS = 1024;
 GET_BLOCKS = (num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

 cudakernel.ThreadBlockSize = [CUDA_NUM_THREADS];
 cudakernel.GridSize = [floor(GET_BLOCKS)];

  coldata = gpuArray.zeros(sqrt(height), sqrt(height), width , imagenum, 'single') ;
 
  coldata = feval(cudakernel,num_kernels, x, height, width, imagenum, coldata);

end