function bolb = colormul_AIR(bolb)

datasize = size(bolb);

    for i = 1 : datasize(4)
    
        temp = bolb(:,:,:,i);
        temp(:,:,1) = temp(:,:,1) * ( 1 + randi(150)*0.001*(-1)^randi(200) );        
        temp(:,:,2) = temp(:,:,2) * ( 1 + randi(150)*0.001*(-1)^randi(200) );             
        temp(:,:,3) = temp(:,:,3) * ( 1 + randi(150)*0.001*(-1)^randi(200) );     
        bolb(:,:,:,i) = temp;
    
    end

end