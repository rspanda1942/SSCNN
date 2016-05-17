function [colformat]=im2colmat(i2cmask,imagedata,channel)

[idn,idm]=size(i2cmask);
colformat=zeros(idn*channel,idm);

for ci=1:channel
    im=imagedata(:,:,ci);
    x=im(i2cmask);
    colformat(idn*(ci-1)+1:idn*ci,:)=x;
end

end