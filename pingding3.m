function [w,OA,KAPPA] = pingding3(label,predict)  


class=max(label);
w=ones(class+1,class+2);

for i=1:class
     for j=1:class
       w(i,j)=sum(label==i&predict==j);
     end    
end

w(1:class,class+1)=sum(w(1:class,1:class),2);
w(class+1,1:class+1)=sum(w(1:class,1:class+1));
aa=0;
for i=1:class
      w(i,class+2)=w(i,i)/w(i,class+1)*10000;
      aa=w(i,i)+aa;
end
w(class+1,class+2)=aa/w(class+1,class+1)*10000;
OA=aa/w(class+1,class+1);
QI=w(class+1,1:class)*w(1:class,class+1)/(w(class+1,class+1)*w(class+1,class+1));
KAPPA=(OA-QI)/(1-QI);
w=uint32(w)

KAPPA
