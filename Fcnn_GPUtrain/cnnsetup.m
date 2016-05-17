function net = cnnsetup(net)

    inputsmaps = net.layers{1}.outputmaps;   % input image channel

    for L = 1 : numel(net.layers)
        %% 
        
        % initialize conv layer weight
         if strcmp(net.layers{L}.type, 'conv')  
               
             Cksize = net.layers{L}.kernelsize;   % Current layer kernel size
             Comap = net.layers{L}.outputmaps;   % Current layer outputmap size && kernel number             
             net.layers{L}.height = net.layers{L-1}.height - net.layers{L}.kernelsize + 1;
             net.layers{L}.width = net.layers{L-1}.width - net.layers{L}.kernelsize + 1;            
             
             if strcmp(net.layers{L}.weight, 'gaussian')  
                 WeightV = net.layers{L}.Std;
                 net.layers{L}.k = normrnd(0  ,  WeightV  ,  inputsmaps * Cksize ^ 2  ,  Comap);
                 net.layers{L}.b = zeros(1, Comap) + net.layers{L}.value;                            
             end
              
             inputsmaps = Comap;             
         end
         %% 
         
         % pooling layer reduce the map size
         if strcmp(net.layers{L}.type, 'pool')  
 
             net.layers{L}.height = floor( (net.layers{L-1}.height - net.layers{L}.poolsize)/ net.layers{L}.stride) + 1;       
             net.layers{L}.width = floor( (net.layers{L-1}.width - net.layers{L}.poolsize)/ net.layers{L}.stride) + 1;       
             net.layers{L}.outputmaps = net.layers{L-1}.outputmaps;
         end
         %% 
         
         if strcmp(net.layers{L}.type, 'relu')  
 
             net.layers{L}.height = net.layers{L-1}.height;
             net.layers{L}.width = net.layers{L-1}.width;
             net.layers{L}.outputmaps = net.layers{L-1}.outputmaps;
            
         end        
                  %% 
         
         if strcmp(net.layers{L}.type, 'sigmoid')  
 
             net.layers{L}.height = net.layers{L-1}.height;
             net.layers{L}.width = net.layers{L-1}.width;
             net.layers{L}.outputmaps = net.layers{L-1}.outputmaps;
           
         end     
         %% 
         
          if strcmp(net.layers{L}.type, 'lrn')  
 
             net.layers{L}.height = net.layers{L-1}.height;
             net.layers{L}.width = net.layers{L-1}.width;
             net.layers{L}.outputmaps = net.layers{L-1}.outputmaps;

         end           
         %% 
         
          if strcmp(net.layers{L}.type, 'padding')  
 
             net.layers{L}.height = net.layers{L-1}.height + 2 * net.layers{L}.padsize;
             net.layers{L}.width = net.layers{L-1}.width + 2 * net.layers{L}.padsize;
             net.layers{L}.outputmaps = net.layers{L-1}.outputmaps;

          end             
          %% 
         
           if strcmp(net.layers{L}.type, 'dropout')  
  
             net.layers{L}.height = net.layers{L-1}.height;
             net.layers{L}.width = net.layers{L-1}.width;
             net.layers{L}.outputmaps = net.layers{L-1}.outputmaps;

         end             
          
          
         %% 
         
            % initialize fullconnect layer weight
         if strcmp(net.layers{L}.type, 'full')               
             fan_in = inputsmaps *  net.layers{L-1}.height * net.layers{L-1}.width ; 
             fan_out = net.layers{L}.outputmaps;
             
             if strcmp(net.layers{L}.weight, 'gaussian')  
                 
                 para = net.layers{L}.Std;
                 net.layers{L}.k = normrnd(0  ,  para  ,  fan_in ,  fan_out);
                 net.layers{L}.b = zeros(1, fan_out) + net.layers{L}.value;             
                 
             end
        
             net.layers{L}.height = 1;
             net.layers{L}.width = 1;

             inputsmaps = fan_out;
         end      
         %% 
         
         if strcmp(net.layers{L}.type, 'loss')                             
             net.layers{L}.k = zeros(inputsmaps , 1);
         end          
         
         
    end

end