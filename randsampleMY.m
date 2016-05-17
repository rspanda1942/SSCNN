function [trainS  , textS ]= randsampleMY(label , percent)

% allT=find(label~=0);
% te1=size(allT,1);

clan=max(label);

flab=[];  %train
% fsam=[];

flab2=[];  %text
% fsam1=[];


for i=1:clan
    
labT=find(label==i);
news=randperm(size(labT,1))';
te=fix(size(news,1)*percent);

if percent > 1
    te = percent;
end

if percent > size(labT,1)
te=15;
end

  labT2=labT(news(1:te));
  labT3=labT(news(te+1:end));




flab=[flab;labT2];
flab2=[flab2;labT3];

end

trainS = flab;
textS = flab2;

end