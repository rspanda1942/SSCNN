function net = cnnbp_GPU(net, y, im2colkernel, maxBkernel ,im2colNkernel , LRNDiffkernel)

n = numel(net.layers); 
numImages = size(y ,1);  % data size
groundTruth = gpuArray.zeros(net.layers{n}.classnum, numImages,'single');

for nnimage = 1 : numImages
groundTruth( y(nnimage), nnimage) = 1;
end

    for L = fliplr ( 3 : n )  %  for each layer
        %% 
    
         if strcmp(net.layers{L}.type, 'loss')  
             
             %  groundTruth =  n * m ,  n = output or class number, m = data size 
             %  net.layers{L}.delta is same as...     
             
           net.layers{L}.delta = net.layers{L}.p -  groundTruth;
           temp11 = log( net.layers{L}.p ) ;
           temp11 = temp11(groundTruth==1) ;
           net.Cost = gather( - 1./numImages * sum(temp11(:)) );   % Cost  did not include W , the weight
           net.lost = net.Cost; 
           
         end
         
         
                 %% 
    
         if strcmp(net.layers{L}.type, 'mseloss')  
             
             %  groundTruth =  n * m ,  n = output or class number, m = data size 
             %  net.layers{L}.delta is same as...     
             
           net.layers{L}.delta = net.layers{L}.p -  groundTruth;
           temp11 = 1./numImages * 0.5 * (net.layers{L}.p -  groundTruth).* (net.layers{L}.p -  groundTruth);
%            temp11 = temp11(groundTruth==1) ;
           net.Cost = sum(temp11(:));   % Cost  did not include W , the weight
           net.lost = net.Cost; 
           
         end
        %% 
         if strcmp(net.layers{L}.type, 'full')  
                 
                  % net.layers{L}.k = weight  n * m  n = last layer input,
                  %                                  m = output
                  % net.layers{L}.delta = delta  n * m  n = last layer input,
                  %                                     m = data size                 
             net.layers{L}.delta = net.layers{L}.k * net.layers{L+1}.delta ;        
         
         end
        %%         
         if strcmp(net.layers{L}.type, 'dropout')  
                 
                  % net.layers{L}.k = weight  n * m  n = last layer input,
                  %                                  m = output
                  % net.layers{L}.delta = delta  n * m  n = last layer input,
                  %                                     m = data size                 
             net.layers{L}.delta = net.layers{L+1}.delta .* reshape(net.layers{L}.dropOutMask, size(net.layers{L+1}.delta));        
%              net.layers{L+1} = rmfield(net.layers{L+1},'delta');
         end
         %% 
         if strcmp(net.layers{L}.type, 'relu')          
                  % net.layers{L}.delta = delta  n * m  n = last layer input,
                  %                                     m = data size     

             if numel( size( net.layers{L}.a ) ) ~= numel( size( net.layers{L+1}.delta ) )                 
                 temp = reshape( net.layers{L}.a, size( net.layers{L+1}.delta) );  
                 net.layers{L}.delta = net.layers{L+1}.delta .* (temp>0);
             else            
                 net.layers{L}.delta = net.layers{L+1}.delta .* (net.layers{L}.a>0);    
             end                            
        %%%%     
%          net.layers{L+1} = rmfield(net.layers{L+1},'delta');
             
         end         
          %% 
         if strcmp(net.layers{L}.type, 'lrn')          
                  % net.layers{L}.delta = delta  n * m  n = last layer input,
                  %                                     m = data size     

             if numel( size( net.layers{L}.a ) ) ~= numel( size( net.layers{L+1}.delta ) )                 
                 temp = reshape( net.layers{L+1}.delta , size( net.layers{L}.a) );  
                 
                net.layers{L}.delta = LRNComputeDiff_GPU(temp, net.layers{L}.a, ...
                   net.layers{L-1}.a, net.layers{L}.scale , ...
                   net.layers{L}.local_size, net.layers{L}.lrn_alpha, net.layers{L}.lrn_beta, LRNDiffkernel) ;            
                 
%                  net.layers{L}.delta = net.layers{L+1}.delta .* (temp>0);
             else            
                net.layers{L}.delta = LRNComputeDiff_GPU(net.layers{L+1}.delta, net.layers{L}.a, ...
                   net.layers{L-1}.a, net.layers{L}.scale , ...
                   net.layers{L}.local_size, net.layers{L}.lrn_alpha, net.layers{L}.lrn_beta, LRNDiffkernel) ;     
             end       
         %%%%    
%              net.layers{L+1} = rmfield(net.layers{L+1},'delta');
             
         end          
         
         
         %% 
        if strcmp(net.layers{L}.type, 'padding')             

         strats = net.layers{L}.padsize + 1;
         endstrats = net.layers{L}.height;
         padsize = net.layers{L}.padsize;
         net.layers{L}.delta = net.layers{L+1}.delta(strats:endstrats-padsize,strats:endstrats-padsize,:,:);
%          net.layers{L+1} = rmfield(net.layers{L+1},'delta');

        end              
                  %% 
         if strcmp(net.layers{L}.type, 'sigmoid')          
                  % net.layers{L}.delta = delta  n * m  n = last layer input,
                  %                                     m = data size     

             if numel( size( net.layers{L}.a ) ) ~= numel( size( net.layers{L+1}.delta ) )
                 
                 temp = reshape( net.layers{L}.a, size( net.layers{L+1}.delta) );  
                 net.layers{L}.delta = net.layers{L+1}.delta .* temp .* (1 - temp);
             else            
                 net.layers{L}.delta = net.layers{L+1}.delta .* net.layers{L}.a .* (1 - net.layers{L}.a );    
             end                                                        
         end   
        %% 
         if strcmp(net.layers{L}.type, 'pool')  
             
            height = net.layers{L-1}.height; % ori image size
            width  = net.layers{L-1}.width;
            psize = net.layers{L}.poolsize;   % pool size
            pstrike = net.layers{L}.stride;   % pool size
            channels = net.layers{L}.outputmaps;  % image band
            imagenum = numImages;    % image number

            poolheight = net.layers{L}.height;
            poolwidth = net.layers{L}.width;

            temp = reshape(net.layers{L+1}.delta, net.layers{L}.height, net.layers{L}.width,[], numImages );       
            net.layers{L}.delta = ReverseMaxpool_GPU(temp, net.layers{L}.indice, maxBkernel,...
                imagenum, channels, height, width, poolheight, poolwidth, psize, pstrike);
            
         %%%%   
%             net.layers{L+1} = rmfield(net.layers{L+1},'delta');
            
         
         end   
         %% 
          if strcmp(net.layers{L}.type, 'conv')  
              
           Cksize = net.layers{L}.kernelsize;   % Current layer kernel size
           Comap = net.layers{L}.outputmaps;   % Current layer outputmap size

           LComap = net.layers{L-1}.outputmaps;   % 前一层 layer outputmap size
           FFsize = repmat(Cksize^2, LComap, 1);
           tempK = mat2cell( gather(net.layers{L}.k), FFsize);           
           tempK = cell2mat(tempK');       
           tempK = reshape( flipud(tempK), Cksize * Cksize * Comap ,[] );
           tempK = gpuArray( single(tempK) );
                    
           temp = padarray(net.layers{L+1}.delta, [Cksize-1 Cksize-1]);
             
           net.layers{L}.delta = gpuArray.zeros(net.layers{L-1}.height, net.layers{L-1}.width, LComap, numImages,'single');
              
                for numbatch = 1:numImages
                   coldata = im2col_GPU(squeeze(temp(:, :, :, numbatch)), im2colkernel, ...
                       net.layers{L}.height+Cksize-1, net.layers{L}.width+Cksize-1, Cksize, Comap);                                                      
                   temp1 = coldata * tempK;                 
                   net.layers{L}.delta(:,:,:,numbatch) = reshape(temp1, net.layers{L-1}.height, net.layers{L-1}.width, LComap);
                end                           
          end   
          
    end
 
    %%   cal gradiant
    for L = fliplr ( 2 : n-1 )  %  for each layer       
        %% 
    
         if strcmp(net.layers{L}.type, 'full')  
            
             if numel( size( net.layers{L-1}.a ))>2
                 net.layers{L}.Wgrad = (1./numImages) * net.layers{L+1}.delta * reshape( net.layers{L-1}.a, [], size(net.layers{L-1}.a,4))';             
             else            
                 net.layers{L}.Wgrad = (1./numImages) * net.layers{L+1}.delta * net.layers{L-1}.a';
             end
              
             net.layers{L}.bgrad = (1./numImages) * sum( net.layers{L+1}.delta ,2) ;     
             
         end              
         %% 

          if strcmp(net.layers{L}.type, 'conv')  
        
           Cksize = net.layers{L}.kernelsize;   % Current layer kernel size
           Comap = net.layers{L}.outputmaps;   % Current layer outputmap size             
           LComap = net.layers{L-1}.outputmaps;   % 前一层 layer outputmap size             
              
%                net.layers{L}.Wgrad = zeros( LComap * Cksize^2 , Comap);
%                tempall= zeros( LComap * Cksize^2 , Comap);
%                tempA = gather(net.layers{L-1}.a);
%                tempB = gather(net.layers{L+1}.delta);
               deLsize = size(net.layers{L+1}.delta,1);

               net.layers{L}.Wgrad = gpuArray.zeros( LComap * Cksize^2 , Comap,'single');
%                tempall= gpuArray( single(zeros( LComap * Cksize^2 , Comap)));
%                tempA = net.layers{L-1}.a;
%                tempB = net.layers{L+1}.delta;               
%                    tic;               
               for numbatch = 1:numImages

               coldata = im2colN_GPU(net.layers{L-1}.a(:, :, :, numbatch), im2colNkernel, size(net.layers{L-1}.a,1) -deLsize+1 , ...
                         size(net.layers{L-1}.a,1) - deLsize+1, deLsize, size(net.layers{L-1}.a,3)) ;
               tempall = coldata * reshape(net.layers{L+1}.delta(:,:,:,numbatch),[],size(net.layers{L+1}.delta,3));

%                    for fer = 1:Comap
%                        
%                        temp= convn(squeeze(tempA(:, :, :, numbatch)), rot90(squeeze(tempB(:,:,fer,numbatch)),2),'valid');
%                        tempall(: , fer) = temp(:);
%                    end

                   net.layers{L}.Wgrad = net.layers{L}.Wgrad + tempall;
                   
               end            
         
                net.layers{L}.Wgrad = gpuArray( single(net.layers{L}.Wgrad));                    
                net.layers{L}.Wgrad = (1./numImages) * net.layers{L}.Wgrad;
                
                net.layers{L}.bgrad = squeeze(sum( net.layers{L+1}.delta ,1)) ; 
                net.layers{L}.bgrad = squeeze(sum( net.layers{L}.bgrad ,1)) ;   
                net.layers{L}.bgrad = squeeze(sum( net.layers{L}.bgrad ,2)) ;                 
                net.layers{L}.bgrad = (1./numImages) * net.layers{L}.bgrad;
                 
          end
       
    
    end

end