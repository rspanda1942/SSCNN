function delta= LRNComputeDiff_GPU(lastdelta, lastdata, bottom_data, curscale , Xrosize, alpha, beta, LRNDiffkernel)


 cache_ratio = 2.0 * alpha * beta / Xrosize ;
%  cache_ratio = 2.0 * alpha * beta ;
 height = size(lastdelta,1);
 width = size(lastdelta,2);
 channels = size(lastdelta,3);
 imagenum = size(lastdelta,4);
%     alpha: 0.00005
%     beta: 0.75
%  alpha_over_size = alpha / Xrosize;
 
 num_kernels = imagenum * height * width;   %The using  CUDA thread number
 Cuda_Thread = 1024;
 Cuda_Bolck = (num_kernels + Cuda_Thread - 1) / Cuda_Thread;

 LRNDiffkernel.ThreadBlockSize = [Cuda_Thread];
 LRNDiffkernel.GridSize = [floor(Cuda_Bolck)];


  delta = gpuArray.zeros(height, width, channels, imagenum, 'single' );


         delta = feval(LRNDiffkernel, num_kernels, bottom_data, lastdata, curscale, ...
           lastdelta, imagenum, channels, height, width, Xrosize, -1 * beta, cache_ratio, delta);  
  

       
%  LrnScalekernel = parallel.gpu.CUDAKernel('FlrnLayer.ptx','FlrnLayer.cu','LRNFillScale'); % reverse Maxpooling cuda kernel


end


