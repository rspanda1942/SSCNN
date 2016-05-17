function bolb = colormul(bolb)

datasize = size(bolb);

    for i = 1 : datasize(4)
    
        temp = rgb2hsv(bolb(:,:,:,i));
        temp(:,:,1) = temp(:,:,1) * ( 1 + randi(300)*0.001*(-1)^randi(300) );        
        temp(:,:,2) = temp(:,:,2) * ( 1 + randi(300)*0.001*(-1)^randi(300) );             
        temp(:,:,3) = temp(:,:,3) * ( 1 + randi(300)*0.001*(-1)^randi(300) );     
        bolb(:,:,:,i) = hsv2rgb(temp);
    
    end

end