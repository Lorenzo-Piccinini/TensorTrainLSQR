function [QX1,QX2,RX]=trunc_diag4(A,B,r,tol,type_trunc)
% function [QX1,QX2,RX]=trunc_diag4(A,B,r,tol,type_trunc)
% 
% function used to compute low rank factors X1*X2' that approximates A*B'.
% INPUT:
% - A,B: factors we want to approximate.
% - r: maximum rank after truncation.
% - tol: truncation tolerance.
% - type_trunc: how truncation is done.


[Q1,R1]=qr(A,0);
[Q2,R2]=qr(B,0);
[U,S,V]=svd(full(R1*R2'),0);
SS = diag(S);SS=SS(:);


if type_trunc==1
% t=find(cumsum(diag(S))./sum(diag(S))>1-tol,1);
  t=find(cumsum(SS)./sum(SS)>1-tol,1);
else
  t=find(sqrt(cumsum(diag(S).^2))./norm(diag(S))>1-tol,1);
end
t=max([t,1]);
k = round(min([r t]));
%X1=Q1*(U(:,1:k)*sqrt(S(1:k,1:k)));
%X2=Q2*(V(:,1:k)*sqrt(S(1:k,1:k)));

QX1 = Q1*U(:,1:k); QX2 = Q2*V(:,1:k);
%RX = sqrt(SS(1:k));
RX=SS(1:k);

% We store RX1 as a diagonal matrix.


end
