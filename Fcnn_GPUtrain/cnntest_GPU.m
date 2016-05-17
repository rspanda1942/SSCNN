function pred = cnntest_GPU(net, x, opts, im2colkernel, maxFkernel, LrnScalekernel, LRNComputekernel)


m = size(x, 4);  % number of all testing data
%% 

  switch opts.dataAug
      %% 
      
    case 'normal'

   pred = [];
   for i=1:opts.testbatchsize:m %X输入的样本个数，分成小块
 
     lastIndex=min(i+opts.testbatchsize-1, m);     
     bolb = x(:,:,:,i:lastIndex);
     bolb = gpuArray( single(bolb) );     
     net = cnnForward_GPU(net, bolb,'CTest', im2colkernel, maxFkernel, LrnScalekernel, LRNComputekernel);
     
     [a b]=max(gather(net.layers{numel(net.layers)}.p));
      pred = [pred ; b'];
     
   end
   %% 
   
    case 'crop'
   pred = [];
   for i=1:opts.testbatchsize:m %X输入的样本个数，分成小块
 
     lastIndex=min(i+opts.testbatchsize-1, m);     
     bolbOri = x(:,:,:,i:lastIndex);
     bolb = zeros(opts.cropsize, opts.cropsize, size(x,3), opts.testbatchsize * 10);
     nn = 1;
     
     for iii = 1: opts.testbatchsize
      posi(1,:) = [1 , 1];
      posi(2,:) = [size(x,1)-opts.cropsize+1 , 1];
      posi(3,:) = [1 , size(x,1)-opts.cropsize+1];
      posi(4,:) = [size(x,1)-opts.cropsize+1 , size(x,1)-opts.cropsize+1];   
      midx = floor((size(x,1)-opts.cropsize)/2);
      posi(5,:) = [midx , midx];          
      
         for jjj = 1: 5
             bolb(:,:,:,nn) = bolbOri(posi(jjj,1):posi(jjj,1)+opts.cropsize-1, posi(jjj,2):posi(jjj,2)+opts.cropsize-1, : , iii);
             nn = nn+1;
             for badn = 1 : size(x,3)
             bolb(:,:,badn,nn) = rot90(bolb(:,:,badn,nn - 1) , 2);
             end
             nn  = nn+1;
         end
         
     end
     
     
     bolb = gpuArray( single(bolb) );     
     net = cnnForward_GPU(net, bolb,'CTest', im2colkernel, maxFkernel, LrnScalekernel, LRNComputekernel);
     
     [a b]=max(gather(net.layers{numel(net.layers)}.p));
     
     newb = reshape(b ,10,[] );
     lll=[];
     for i = 1 : net.layers{numel(net.layers)}.classnum
         temp=uint8(newb==i);
         lll(:,i)=sum(temp,1);    
         
     end
     
      [aa, labelnew]=max(lll');  
      
      pred = [pred ; labelnew'];

   end   
   %% 
       case 'elastic'

   pred = [];
   for i=1:opts.testbatchsize:m %X输入的样本个数，分成小块
 
     lastIndex=min(i+opts.testbatchsize-1, m);     
     bolb = x(:,:,:,i:lastIndex);
     bolb = gpuArray( single(bolb) );          
     net = cnnForward_GPU(net, bolb,'CTest', im2colkernel, maxFkernel, LrnScalekernel, LRNComputekernel);
     
     [a b]=max(gather(net.layers{numel(net.layers)}.p));
      pred = [pred ; b'];
     
   end
             
             

  end
  %% 
   
   

end

