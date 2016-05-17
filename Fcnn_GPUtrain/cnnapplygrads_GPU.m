function net = cnnapplygrads_GPU(net, opts)

% opts.base_lr =  opts.base_lr  * (0.993 ^ opts.iter);
Cbase_lr = double( opts.base_lr ) * ( (1 + opts.gamma * opts.iter)^(-opts.power) );
% Cbase_lr =  opts.base_lr ;
n = numel(net.layers); 
%%  set momentum

if (opts.iter == 1)
    for L = 1 : n   %  for each layer      
        
          if strcmp(net.layers{L}.type, 'full')                        
             net.layers{L}.Wmom = gpuArray.zeros( size(net.layers{L}.Wgrad),'single');
             net.layers{L}.bmom = gpuArray.zeros( size(net.layers{L}.bgrad),'single');            
          end         
        
          if strcmp(net.layers{L}.type, 'conv')                        
             net.layers{L}.Wmom = gpuArray.zeros( size(net.layers{L}.Wgrad),'single');
             net.layers{L}.bmom = gpuArray.zeros( size(net.layers{L}.bgrad),'single');            
          end            
                             
    end
end
%% 

    for L = 1 : n   %  for each layer      
        
          if strcmp(net.layers{L}.type, 'full')              
              
             net.Cost = gather(opts.weight_decay * sum((net.layers{L}.k(:).^2)) )+ net.Cost;
              
             net.layers{L}.Wmom = 0.9 * net.layers{L}.Wmom - opts.weight_decay * net.layers{L}.weight_multiplter * Cbase_lr * net.layers{L}.k' ...
                 - net.layers{L}.weight_multiplter * Cbase_lr * net.layers{L}.Wgrad;
             net.layers{L}.k = net.layers{L}.k + net.layers{L}.Wmom';

             net.layers{L}.bmom = 0.9 * net.layers{L}.bmom - opts.bias_decay * net.layers{L}.bias_multiplter * Cbase_lr * net.layers{L}.b' ...
                 - net.layers{L}.bias_multiplter * Cbase_lr * net.layers{L}.bgrad;       
             net.layers{L}.b = net.layers{L}.b + net.layers{L}.bmom';        
                          
         end         
        
          if strcmp(net.layers{L}.type, 'conv')        
              
             net.Cost = gather(opts.weight_decay * sum(sum( net.layers{L}.k).^2 ) ) + net.Cost;             
              
             net.layers{L}.Wmom = 0.9 * net.layers{L}.Wmom - opts.weight_decay * net.layers{L}.weight_multiplter * Cbase_lr * net.layers{L}.k ...
                 - net.layers{L}.weight_multiplter * Cbase_lr * net.layers{L}.Wgrad;
             net.layers{L}.k = net.layers{L}.k + net.layers{L}.Wmom ;
             
             net.layers{L}.bmom = 0.9 * net.layers{L}.bmom - opts.bias_decay * net.layers{L}.bias_multiplter * Cbase_lr * net.layers{L}.b' ...
                 - net.layers{L}.bias_multiplter * Cbase_lr * net.layers{L}.bgrad;
             net.layers{L}.b = net.layers{L}.b + net.layers{L}.bmom';        
             

         end            
                             
    end

end