function reversedata = ReverseMaxpool_GPU(pooldata, indice, cudakernel, imagenum, channels, height, width, poolheight, poolwidth, poolsize, pstrike)


       Cuda_Thread_X = 32;
       Cuda_Thread_Y = 32;     
       Cuda_Block_X = floor( (poolwidth-1)/Cuda_Thread_X ) + 1;
       Cuda_Block_Y = floor( (poolheight-1)/Cuda_Thread_Y ) + 1;
       Cuda_Block_Z = channels * imagenum;     
       
       cudakernel.ThreadBlockSize = [Cuda_Thread_X Cuda_Thread_Y];
       cudakernel.GridSize = [Cuda_Block_X Cuda_Block_Y Cuda_Block_Z];
 
       reversedata = gpuArray.zeros(height, width, channels, imagenum,'single');

       reversedata = feval(cudakernel, reversedata, pooldata, indice, ...
           imagenum, channels, height, width, poolheight, poolwidth, poolsize, pstrike);  

end