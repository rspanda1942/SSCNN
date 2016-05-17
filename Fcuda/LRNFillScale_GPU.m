function [lrndata ,scaledata]= LRNFillScale_GPU(data, Xrosize, alpha, beta, LrnScalekernel, LRNComputekernel)


 height = size(data,1);
 width = size(data,2);
 channels = size(data,3);
 imagenum = size(data,4);
%     alpha: 0.00005
%     beta: 0.75
 alpha_over_size = alpha / Xrosize;
 
 num_kernels = imagenum * height * width;   %The using  CUDA thread number
 num_kernels2 = imagenum * channels * height * width;   %The using  CUDA thread number
 Cuda_Thread = 1024;
 Cuda_Bolck = (num_kernels + Cuda_Thread - 1) / Cuda_Thread;
 Cuda_Bolck2 = (num_kernels2 + Cuda_Thread - 1) / Cuda_Thread;
 
 LrnScalekernel.ThreadBlockSize = [Cuda_Thread];
 LrnScalekernel.GridSize = [floor(Cuda_Bolck)];
 LRNComputekernel.ThreadBlockSize = [Cuda_Thread];
 LRNComputekernel.GridSize = [floor(Cuda_Bolck2)];

  scaledata = gpuArray.zeros(height, width, channels, imagenum, 'single' );
  lrndata = gpuArray.zeros(height, width, channels, imagenum, 'single' );

         scaledata = feval(LrnScalekernel, num_kernels, data, imagenum, channels, ...
           height, width, Xrosize, alpha_over_size, scaledata);  
  
           lrndata = feval(LRNComputekernel, num_kernels2, data, scaledata, ...
           -1 * beta, lrndata);       
       
%  LrnScalekernel = parallel.gpu.CUDAKernel('FlrnLayer.ptx','FlrnLayer.cu','LRNFillScale'); % reverse Maxpooling cuda kernel


end


