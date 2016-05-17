function net = delenet(net)

n = numel(net.layers); 


    for L = 3 : n   %  for each layer       
          
         net.layers{L} = rmfield(net.layers{L},'delta');                             
         net.layers{L-1} = rmfield(net.layers{L-1},'a');                  
    end