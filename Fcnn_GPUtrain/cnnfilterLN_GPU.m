function net = cnnfilterLN_GPU(net, value)

n = numel(net.layers);  % number of layer

if value>0
    for L = 1 : n   %  for each layer            
%%        
        if strcmp(net.layers{L}.type, 'conv')  
            
            
%            Cksize = net.layers{L}.kernelsize;   % Current layer kernel size
%            Comap = net.layers{L}.outputmaps;   % Current layer outputmap size
           temp = net.layers{L}.k .* net.layers{L}.k;       
           nsize = size(temp,1);
           temp = sqrt( sum(temp,1)/nsize );
           posi = find( gather(temp) > single(value));
           
           temp = net.layers{L}.k(:,posi) .* net.layers{L}.k(:,posi);    
           temp = sum(temp,1);
           
           if sum(posi)>0
           net.layers{L}.k(:,posi) = bsxfun(@rdivide, net.layers{L}.k(:,posi), sqrt( temp )/ (sqrt(nsize) * value) );
           end
        end        
        
        
    end
end




end