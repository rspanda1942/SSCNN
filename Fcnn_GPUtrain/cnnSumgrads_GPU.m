function net = cnnSumgrads_GPU(net, opts, minib)

n = numel(net.layers); 

    %%   sum gradiant
   if (minib == 1)
    for L = fliplr ( 2 : n-1 )  %  for each layer                    
          if strcmp(net.layers{L}.type, 'conv')  
              
             net.layers{L}.Wsum = gpuArray.zeros( size(net.layers{L}.Wgrad),'single');
             net.layers{L}.bsum = gpuArray.zeros( size(net.layers{L}.bgrad),'single');      
                
          end
          if strcmp(net.layers{L}.type, 'full')  
              
             net.layers{L}.Wsum = gpuArray.zeros( size(net.layers{L}.Wgrad),'single');
             net.layers{L}.bsum = gpuArray.zeros( size(net.layers{L}.bgrad),'single');      
                
          end       
    end
   end
   
   %% 
    for L = fliplr ( 2 : n-1 )  %  for each layer                    
          if strcmp(net.layers{L}.type, 'conv')  
              
             net.layers{L}.Wsum = net.layers{L}.Wsum + net.layers{L}.Wgrad;
             net.layers{L}.bsum = net.layers{L}.bsum + net.layers{L}.bgrad;      
                
          end
          if strcmp(net.layers{L}.type, 'full')  
              
             net.layers{L}.Wsum = net.layers{L}.Wsum + net.layers{L}.Wgrad;
             net.layers{L}.bsum = net.layers{L}.bsum + net.layers{L}.bgrad;      
                
          end       
        
    end  
    
    %%  
   if (minib == opts.minisize)
    for L = fliplr ( 2 : n-1 )  %  for each layer                    
          if strcmp(net.layers{L}.type, 'conv')  
%              net.layers{L}.Wsum = net.layers{L}.Wsum + net.layers{L}.Wgrad;
%              net.layers{L}.bsum = net.layers{L}.bsum + net.layers{L}.bgrad;                   
             net.layers{L}.Wsum = net.layers{L}.Wsum / opts.minisize;
             net.layers{L}.bsum = net.layers{L}.bsum / opts.minisize;         
             net.layers{L}.Wgrad = net.layers{L}.Wsum;
             net.layers{L}.bgrad = net.layers{L}.bsum;                
          end
          if strcmp(net.layers{L}.type, 'full')  
%              net.layers{L}.Wsum = net.layers{L}.Wsum + net.layers{L}.Wgrad;
%              net.layers{L}.bsum = net.layers{L}.bsum + net.layers{L}.bgrad;                   
             net.layers{L}.Wsum = net.layers{L}.Wsum / opts.minisize;
             net.layers{L}.bsum = net.layers{L}.bsum / opts.minisize;        
             net.layers{L}.Wgrad = net.layers{L}.Wsum;
             net.layers{L}.bgrad = net.layers{L}.bsum;                    
          end       
        
    end
   end  
end