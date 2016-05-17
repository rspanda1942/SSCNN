function [bolb ,bolbLabel]= dataSample(x, y, opts)


m = size(x, 4);  % number of all training data

  switch opts.dataAug
%%      
    case 'normal'
       bolbLabel = zeros(opts.batchsize,1);  
       bolb = zeros(size(x,1), size(x,1), size(x,3), opts.batchsize);
       for ii = 1 : opts.batchsize
       inum = randi(m);
       bolbLabel(ii) = y(inum);
       bolb(:,:,:,ii) = x(: ,: , :, inum);       
       end

%%                                     
     case 'crop'
       bolbLabel = zeros(opts.batchsize,1);  
       bolb = zeros(opts.cropsize, opts.cropsize, size(x,3), opts.batchsize);
       for ii = 1 : opts.batchsize
           imagesize = size(x,1);
           cropsize = opts.cropsize;
           inum = randi(m);
           temp = x(:, :, :, inum);
           if strcmp(opts.jetting, 'on')  
           jetnum = 1 + randi(opts.jetting_para*1000)*0.001*(-1)^randi(300);
           txx = size(temp,1);
           tyy = size(temp,2);
           temp = imresize(temp , [fix(txx * jetnum) fix(tyy * jetnum)] ,'bilinear');   
           end
           
           txx = size(temp,1); 
           if txx < imagesize
               
               padyy = fix( (-txx + imagesize)/2);
               temp = padarray(temp, [padyy padyy]);
               
           end
           
           txx = size(temp,1); 
           imagesize = txx;
           
           
           posix = randi(imagesize - cropsize + 1);
           posiy = randi(imagesize - cropsize + 1);     
           bolbLabel(ii) = y(inum);
           bolb(:,:,:,ii) = temp(posix:posix+cropsize-1, posiy:posiy+cropsize-1, :);
           
           probR = rand(1);
           if probR<0.5
               for ij = 1:size(bolb,3)
                   bolb(:,:,ij,ii) = rot90(bolb(:,:,ij,ii),randi(3));
               end
           end                       
       end
       
  end

end