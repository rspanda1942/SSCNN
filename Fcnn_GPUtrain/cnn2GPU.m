function net = cnn2GPU(net)

    % numel(net.layers)  表示有多少层 
    for L = 1 : numel(net.layers)      
        %% 
         if strcmp(net.layers{L}.type, 'conv')                          
             net.layers{L}.k = gpuArray( single(net.layers{L}.k) );                  
             net.layers{L}.b = gpuArray( single(net.layers{L}.b) );            
         end
         %% 
         if strcmp(net.layers{L}.type, 'full')              
             net.layers{L}.k = gpuArray( single(net.layers{L}.k) );                  
             net.layers{L}.b = gpuArray( single(net.layers{L}.b) );                    
         end                        
    end
end