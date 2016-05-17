function coldata = im2colN_GPU(x, cudakernel,height_col, width_col, psize, channels) 

 height = size(x,1);
 width = size(x,2);



       Cuda_Thread_X = 32;
       Cuda_Thread_Y = 32;     
       Cuda_Block_X = floor( (height_col-1)/Cuda_Thread_X ) + 1;
       Cuda_Block_Y = floor( (width_col-1)/Cuda_Thread_Y ) + 1;
       Cuda_Block_Z = channels;     
       
       cudakernel.ThreadBlockSize = [Cuda_Thread_X Cuda_Thread_Y];
       cudakernel.GridSize = [Cuda_Block_X Cuda_Block_Y Cuda_Block_Z];
       
       coldata = gpuArray.zeros(height_col*width_col*channels, psize*psize,'single');

       coldata= feval(cudakernel, coldata, x, height, width, channels, psize, height_col, width_col);  

end