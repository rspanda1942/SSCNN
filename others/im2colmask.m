function [indexmatrix]=im2colmask(patchsize,imgsize,step)
%% 
% convert image format to col patch format for fast conv

imrow = imgsize(1) ;   % image row size
imcol = imgsize(2) ;   % image column size

Ypatrow = patchsize(1) ;  % patch row size
Xpatcol = patchsize(2) ;  % patch column size

nstepx=step(2);
nstepy=step(1);

xn=(imcol-Xpatcol)/nstepx+1;  % patches number in col  x - axis
yn=(imrow-Ypatrow)/nstepy+1;  % patches number in row  y - axis

xn = floor(xn);
yn = floor(yn);
xn
yn
%% 
mark = 1 : imrow * imcol ;    % mark image to compute cordinati
mark = reshape (mark, imrow , imcol );

indexmatrix = zeros(Ypatrow * Xpatcol , xn * yn);  % cordinate result

%%   row-major
% 
% for yi = 1 : yn
%     
%     Ystar = 1 + (yi-1) * nstepy;
%     
%     for xi = 1 : xn
% %         Xstar = xi + (xi-1) * nstepx;
%         Xstar = 1 + (xi-1) * nstepx;
%         
%         temp = mark(Ystar:Ystar + Ypatrow - 1 , Xstar:Xstar + Xpatcol - 1 );
%         temp = reshape (temp ,Ypatrow * Xpatcol ,1);
%         indexmatrix( : , (yi-1) * xn + xi) = temp ;
%         
%         
%     end
% end
%%  col-major

for xi = 1: xn
    
    Xstar = 1 + (xi-1) * nstepx;
    
    for yi = 1:yn
%         Xstar = xi + (xi-1) * nstepx;
        Ystar = 1 + (yi-1) * nstepy;
        
        temp = mark(Ystar:Ystar + Ypatrow -1 , Xstar:Xstar + Xpatcol -1 );
        temp = reshape (temp ,Ypatrow * Xpatcol ,1);
        indexmatrix( : , (xi-1) * yn + yi) = temp ;
        
        
    end
end


end