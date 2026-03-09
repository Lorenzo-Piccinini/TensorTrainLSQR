function [QX1,QX2,RX]=trunc_diag3(QA1,RA1,QB1,RB1,QA2,RA2,QB2,RB2,r,tol,type_trunc)


% Gram-Schmidt-like for X1 
%W1 = QB1; 
% Gram-Schmidt-like for X1 - modified GS
%nw = size(W1,2); na = size(QA1,2);
%for k=1:na
%    R12(k,1:nw)=QA1(:,k)'*W1;
%    W1=W1-QA1(:,k)*R12(k,1:nw);
%end
global sigmadrop

R12 = QA1'*QB1;
W1 = QB1 - QA1*R12;
t=  QA1'*W1;
R12 = R12+ t;
W1 = W1 - QA1*t;
[Q2,R22] = qr(W1,0);
%{
[Q2,ss,vv] = svd(W1,0);
ss(end,end)
k=sum(diag(ss)>1e-12*ss(1,1));
R22=ss(1:k,1:k)*vv(:,1:k)';
Q2=Q2(:,1:k);
%}
Q1 = [QA1,Q2];
%R2 = [eye(size(QA1,2)), R12; zeros(size(Q2,2),size(QA1,2)), R22];
%R1 = R2*[RA1, zeros(size(RA1,1),size(RB1,2)); zeros(size(RB1,1),size(RA1,2)), RB1];

% [I, R12] * [ RA1 0]
% [0  R22]   [ 0 RB1];
%O= zeros(size(R22,1),size(RA1,2));
%R1 = [RA1, R12*RB1; O, R22*RB1];

% Gram-Schmidt-like for X2
%U1 = QB2;
%nw = size(U1,2); na = size(QA2,2);
%for k=1:na
%    P12(k,1:nw)=QA2(:,k)'*U1;
%    U1=U1-QA2(:,k)*P12(k,1:nw);
%end
P12 = QA2'*QB2;
U1 = QB2-QA2*P12;
t = QA2'*U1;
P12 = P12 + t;
U1 = U1-QA2*t;
[QQ2,P22] = qr(U1,0);
%{
[QQ2,ss,vv] = svd(U1,0);
ss(end,end)
k=sum(diag(ss)>1e-12*ss(1,1));
P22=ss(1:k,1:k)*vv(:,1:k)';
QQ2=QQ2(:,1:k);
%}
%[QQ2,P22,idx] = rrqrx(U1,0);
%P22=P22(:,idx);
%uu1=QQ2*P22; norm(U1(:,idx)-uu1,1)
Q2 = [QA2, QQ2]; 
%P2 = [eye(size(QA2,2)), P12; zeros(size(QQ2,2),size(QA2,2)), P22];
%R2 = P2*[RA2, zeros(size(RA2,1),size(RB2,2)); zeros(size(RB2,1),size(RA2,2)), RB2];

%R1
% [I, R12] * [ RA1 0]
% [0  R22]   [ 0 RB1];
%R2
% [I, P12] * [ RA2 0]
% [0  P22]   [ 0 RB2];
%O= zeros(size(P22,1),size(RA2,2));
%R2 = [RA2, P12*RB2; O, P22*RB2];


%R = [RA1*RA2'+(R12*RB1)*(P12*RB2)', (R12*RB1)*(P22*RB2)';
%    (R22*RB1)*(P12*RB2)', (R22*RB1)*(P22*RB2)'];
wrk1=RB1*RB2';
wrk2=wrk1*P22';
wrk3=wrk1*P12';
%R = [RA1*RA2'+(R12*RB1)*(P12*RB2)', (R12*RB1)*(P22*RB2)';
%    (R22*RB1)*(P12*RB2)', (R22*RB1)*(P22*RB2)'];
R = [RA1*RA2'+R12*wrk3, R12*wrk2;
     R22*wrk3, R22*wrk2];

%[U,S,V]=svd(full(R1*R2'),0);
[U,S,V]=svd(full(R),0);
SS = diag(S);SS=SS(:);

if type_trunc==1
  t=find(cumsum(SS)./sum(SS)>1-tol,1);
else
  t=find(sqrt(cumsum(diag(S).^2))./norm(diag(S))>1-tol,1);
end
t=max([t,1]);
k = t;
k = round(min([r t]));
%if k<length(SS), SS(k:k+1),end

QX1 = Q1*U(:,1:k);  QX2 = Q2*V(:,1:k);
%RX = sqrt(SS(1:k));
RX=SS(1:k);
if k+1<length(SS)
sigmadrop=SS(k+1);
else
sigmadrop=0;
end
%if sigmadrop>1e-4,t,k,sigmadrop,pause,end

end
