function net = cnnForward_GPU(net, x , cnncase, im2colkernel, maxFkernel, LrnScalekernel, LRNComputekernel) 

n = numel(net.layers);  % number of layer
net.layers{1}.a = x;    % training data   data = n * m * channel * number
Sbatch = size(x,4);     % data size
% inputsmaps = net.layers{1}.outputmaps;  % input image channel
% mapsize = [net.layers{1}.height net.layers{1}.width];   % input image size

    for L = 1 : n   %  for each layer        
%%        
        if strcmp(net.layers{L}.type, 'conv')  
            
            
           Cksize = net.layers{L}.kernelsize;   % Current layer kernel size
           Comap = net.layers{L}.outputmaps;   % Current layer outputmap size
           inputsmaps = net.layers{L-1}.outputmaps; 

           net.layers{L}.a = gpuArray.zeros(net.layers{L}.height, net.layers{L}.width, Comap, Sbatch,'single');

           for numbatch = 1:Sbatch   % Sbatch = data size
               
           % image data format transform to colum format   
           % using im2colkernel
           coldata = im2col_GPU(squeeze(net.layers{L-1}.a(:, :, :, numbatch)), im2colkernel, ...
               net.layers{L}.height, net.layers{L}.width, Cksize, inputsmaps);             
           
           temp = coldata * net.layers{L}.k;   % k - n * m  n: kernelsize^2 * inputsmaps , m: number of feature map
           temp = bsxfun(@plus, temp, net.layers{L}.b);   

           net.layers{L}.a(:,:,:,numbatch) = reshape(temp, net.layers{L}.height, net.layers{L}.width, Comap);
                                
           end
        end
%%        
        if strcmp(net.layers{L}.type, 'pool')  
            
       height = net.layers{L-1}.height; % original image size
       width  = net.layers{L-1}.width;
       psize = net.layers{L}.poolsize;   % pool size
       pstrike = net.layers{L}.stride;   % pool strike
       channels = net.layers{L}.outputmaps;  % image band
       imagenum = Sbatch;  % 

       poolheight = net.layers{L}.height;
       poolwidth = net.layers{L}.width;
   
       [net.layers{L}.a ,net.layers{L}.indice] = Maxpool_GPU(net.layers{L-1}.a, maxFkernel, ...
           imagenum, channels, height, width, poolheight, poolwidth, psize , pstrike) ;       
%        mapsize = floor( (mapsize - net.layers{L}.poolsize)/ net.layers{L}.stride) + 1;           
       %%%%                  
%        net.layers{L-1} = rmfield(net.layers{L-1},'a');
       
        end
        
        %%       
        if strcmp(net.layers{L}.type, 'relu')             

            net.layers{L}.a = max(net.layers{L-1}.a, 0);         
        %%%%    
%             net.layers{L-1} = rmfield(net.layers{L-1},'a');
            
        end 
        
        
        %% 
        if strcmp(net.layers{L}.type, 'lrn')            
            
            [net.layers{L}.a ,net.layers{L}.scale] = LRNFillScale_GPU(net.layers{L-1}.a, ...
                net.layers{L}.local_size, net.layers{L}.lrn_alpha, net.layers{L}.lrn_beta, LrnScalekernel, LRNComputekernel);     
            
        end         
        
         %% 
        if strcmp(net.layers{L}.type, 'padding')             

           net.layers{L}.a = padarray(net.layers{L-1}.a, [net.layers{L}.padsize net.layers{L}.padsize]);
%          net.layers{L-1} = rmfield(net.layers{L-1},'a');  
        end          
        
        
        %% 
        if strcmp(net.layers{L}.type, 'sigmoid')             
           if strcmp(net.layers{L-1}.type, 'pool')            
               net.layers{L}.height = net.layers{L-1}.height;
               net.layers{L}.width = net.layers{L-1}.width;
           end  
            net.layers{L}.a = 1./(1+exp(-net.layers{L-1}.a));
        end         
       
        %%         
         if strcmp(net.layers{L}.type, 'dropout')      
             
           switch cnncase
               case 'CTrain'
           net.layers{L}.dropOutMask = ( rand( size( net.layers{L-1}.a ) ) > net.layers{L}.fraction);    
           net.layers{L}.dropOutMask = gpuArray( single(net.layers{L}.dropOutMask));
           net.layers{L}.a = net.layers{L-1}.a .* net.layers{L}.dropOutMask;         
               case 'CTest'
           net.layers{L}.a = net.layers{L-1}.a * net.layers{L}.fraction;                
           end
           
         end        
%% 
        if strcmp(net.layers{L}.type, 'full')  
            
         net.layers{L}.a = net.layers{L}.k' * reshape( net.layers{L-1}.a , [] , Sbatch);   % Sbatch = data size
         net.layers{L}.a = bsxfun(@plus, net.layers{L}.a, net.layers{L}.b');   
         
        end
        
 %%       
         if strcmp(net.layers{L}.type, 'loss')  
            
         net.layers{L}.p = bsxfun(@minus,net.layers{L-1}.a,max(net.layers{L-1}.a, [], 1));
         net.layers{L}.p = exp(net.layers{L}.p);
         net.layers{L}.p = bsxfun(@rdivide, net.layers{L}.p, sum(net.layers{L}.p));
                  
        end
%%        
         if strcmp(net.layers{L}.type, 'mseloss')  
            
         net.layers{L}.p = net.layers{L-1}.a;
                  
        end
                     
    end


end