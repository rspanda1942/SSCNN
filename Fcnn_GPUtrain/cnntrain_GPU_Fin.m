function [net ,opts]= cnntrain_GPU_Fin(net, x, y, Testx, Testy, opts)


tempIter = 1;
savenum = 1;

im2colkernel = parallel.gpu.CUDAKernel('im2col.ptx','im2col.cu');  % image 2 col format cuda kernel
maxFkernel = parallel.gpu.CUDAKernel('Fmaxpool.ptx','Fmaxpool.cu','FMaxPoolForward');  % Maxpooling cuda kernel
maxBkernel = parallel.gpu.CUDAKernel('Fmaxpool.ptx','Fmaxpool.cu','FMaxPoolBackward'); % reverse Maxpooling cuda kernel
im2colNkernel = parallel.gpu.CUDAKernel('im2colN.ptx','im2colN.cu'); % reverse Maxpooling cuda kernel

LrnScalekernel = parallel.gpu.CUDAKernel('FlrnLayer.ptx','FlrnLayer.cu','LRNFillScale'); 
LRNComputekernel = parallel.gpu.CUDAKernel('FlrnLayer.ptx','FlrnLayer.cu','LRNComputeOutput'); 
LRNDiffkernel = parallel.gpu.CUDAKernel('FlrnLayer.ptx','FlrnLayer.cu','LRNComputeDiff'); 
%% 

  for iter = 1 : opts.Maxiter

    % batch is split to samll minibatch,to save memory 
     if strcmp(opts.fixweight, 'on')   
     net = cnnfilterLN_GPU(net  ,  opts.weight_const);     % fixed weights constrain
     end
      
      %% 

     % random sample the training data or data augmentation
     [bolb ,bolbLabel] = dataSample(x, y, opts);

     % color manipulation
     if strcmp(opts.colormul, 'on')  
     bolb = colormul_AIR(bolb);   
     end

     % mean removal
     if strcmp(opts.meansubt, 'on')       
     posix = randi(size(x,1) - opts.cropsize + 1);
     posiy = randi(size(x,2) - opts.cropsize + 1);     
     bolb = bsxfun(@minus, bolb, opts.meanimage(posix:posix+opts.cropsize-1, posiy:posiy+opts.cropsize-1,:)   );   
     end

     bolb = gpuArray( single(bolb) );     

     net = cnnForward_GPU(net, bolb, 'CTrain', im2colkernel, maxFkernel, LrnScalekernel, LRNComputekernel);
     
     net = cnnbp_GPU(net, bolbLabel, im2colkernel, maxBkernel, im2colNkernel, LRNDiffkernel);     

     net = cnnapplygrads_GPU(net, opts);

     
     %% 
     % current learning rate
     Cbase_lr = opts.base_lr * ( (1 + opts.gamma * opts.iter)^(-opts.power) );

     %% 
     
       if (rem(opts.iter, opts.display)==0)                   
         disp(['epoch: ' num2str(opts.iter) '/LR: ' num2str(Cbase_lr) '/Cost: ' num2str(net.Cost) '/lost: ' num2str(net.lost)]);
         net.err(tempIter) = net.lost;
         tempIter = tempIter + 1;         
         
           if (net.layers{1}.outputmaps == 1)
             figure(1)
             display_network(gather(net.layers{2}.k));
             drawnow;     
           else
             figure(1)
             displayColorNetworkNew(gather(net.layers{2}.k));
             drawnow;        
           end
         
         
         
       end
       %% 
       
       if (rem(opts.iter, opts.test_interval)==0)
           
         opts.base_lr = opts.base_lr * opts.lr_decay;
         % show learned weight 
           if (net.layers{1}.outputmaps == 1)
             figure(1)
             display_network(gather(net.layers{2}.k));
             drawnow;     
           else
             figure(1)
             displayColorNetwork(gather(net.layers{2}.k));
             drawnow;        
           end
           TestxSubm = Testx;
           if strcmp(opts.meansubt, 'on')      
         TestxSubm = bsxfun(@minus, Testx, opts.meanimage);     
           end
         pred = cnntest_GPU(net, TestxSubm, opts,  im2colkernel, maxFkernel, LrnScalekernel, LRNComputekernel); 
         accuracy = 100 * sum(pred ~= Testy)/size(pred,1);
%          [w,OA,KAPPA] = pingding3(Testy,pred)  ;
         disp(['Testing' ':.....................' ]);
         disp(['errorRate: ' num2str(accuracy) '%']);
         [w,OA,KAPPA] = pingding3(Testy,pred)  ;
         disp(['Testing' ':.....................' ]);
         
         figure(2)
         plot(net.err);
         xlabel('batchNum', 'FontSize', 20);
         ylabel('Cost', 'FontSize', 20); 
         set(gcf,'color',[1 1 1]);
         drawnow;
%          if accuracy<=10
         tempnet=[num2str(savenum) '_tempnet_' num2str(accuracy) '_.mat'];
         net = delenet(net);
         save ( '-v7.3',tempnet ,'net','opts','w','pred') ;
%          end
         savenum =savenum + 1;
       end
       
       %%    
       if opts.iter==200
           opts.base_lr = opts.base_lr * 0.3 ;         
       end
       if opts.iter==500
           opts.base_lr = opts.base_lr * 0.3 ;         
       end      
       if opts.iter==12000
           opts.base_lr = opts.base_lr * 0.3 ;         
       end      
       %% 
         
         opts.iter = opts.iter + 1;
     
 end
 
end