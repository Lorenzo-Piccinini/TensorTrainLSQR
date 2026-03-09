function[X_1,X_2,r_res,a_res,rks,DD,totres]=lsqr_matrix_multi(coeff,C1,C2,tol,imax,tol_tr,r)
%
% function[X,D,r_res,a_res,rks]=lsqr_gen_trunc_V4(A,B,C1,C2,F,E,tol,imax,tol_tr,r)
%
% Truncated version of the LSQR algorithm (from the paper of Page and Saunders), in the sense that it uses low
% rank approximations instead of full matrices. This implementation solves
% the Generalized Sylvester leasts squares problem 
% 
%                   A*X*F+E*X*B+J*X*K=C1*C2^T,
%
% but it can be
% generalized also for other version of the Sylvester equation.
%
% INPUT: 
% - A,B,F,E: coefficient matrices.
% - C1,C2: low rank rhs.
% - tol: tolerance chosen for the stopping criteria.
% - imax: maximum number of iterations allowed.
% - tol_tr: truncation tolerance.
% - r: maximum rank allowed when doing the truncation.
% 
% OUTPUT:
% - X_1,X_2: approximated solution (it will be computed as X=X1*X2').
% - r_res: vector of relative residuals.
% - a_res: vector of absolute residuals.
% - rks: vector of ranks of the approximated solution ar each iteration.
% - DD: array containing all the direction matrices D^(i).
%
% To use this code cit
% L. Piccinini, and V. Simoncini, "Truncated LSQR for matrix least squares
% problem", Computational Optimization and Application (COAP), v.91 (2025),
% pp.905-932. DOI: 10.1007/s10589-024-00629-w
% More codes available at 
% https://github.com/Lorenzo-Piccinini/Truncated-LSQR
global sigmadrop

% Computing the norm of the right-hand side
beta = sqrt(trace((C1'*C1)*(C2'*C2)));
sq_beta = sqrt(beta);
l = length(coeff);

CC1 = []; CC2 = [];
for i = 1:l
    CC1 = [CC1, coeff{i}{1}' * C1];
    CC2 = [CC2, coeff{i}{2} * C2];
end


AtC=CC1*CC2';

res0_2 = sqrt(trace((CC1'*CC1)*(CC2'*CC2)));

% Updatingss
U_1 = C1/sq_beta;             U_2 = C2/sq_beta;
res = beta;
totres=res;
truenormres=res;
r2=r;
maximum_rank=r2;
[QU_1,RU_1] = qr(U_1,0);      
[QU_2,RU_2] = qr(U_2,0);

[uu,S,vv] = svd(full(RU_1*RU_2'));
SS = diag(S);
RU = (SS);
QU_1=QU_1*uu;                 
QU_2=QU_2*vv;   

flag = 1;
maxrank=0;

[V_1, V_2] = L_T(coeff,U_1,U_2,diag(RU));
[QV_1,QV_2,RV] = trunc_diag4(V_1,V_2,r2,tol_tr,flag);

maxrank=max([maxrank,length(RV)]);
totrank(1)=maxrank;
alfa = norm(RV);

RV=RV/alfa;

% Initializing parameters.
phi = beta;
rho = alfa;
i = 0;
rks = [];
X_1 = zeros(size(QV_1));       
X_2 = zeros(size(QV_2));
QX_1 = eye(size(X_1));         
QX_2 = eye(size(X_2));
RX = zeros(size(RV,2),1); 

rks = [rks;size(X_1,2)];
QD_1 = QV_1;                   
QD_2 = QV_2;
RD = RV;
DD=0;
res0 = beta;
res_old = res0;
r_res.rel_res = [1];
r_res.nrml_res = [1];
a_res = [];
a_res = [a_res;res0];
res_2=res0_2;
sigmatot=[];
sigmatot1=[];


while ( i<imax && res_2/res0_2>tol )
    i = i+1;

    uold=QU_1*diag(RU)*QU_2';uold=uold(:);

    [Ut_1, Ut_2] = L(coeff,QV_1,QV_2,(RV));
    Iu=eye(length(RU));     Iu1=eye(size(Ut_1,2));
    [QU_1,QU_2,RU] = trunc_diag3(QU_1,-alfa*diag(RU),Ut_1,Iu1,QU_2,Iu,Ut_2,Iu1,r2,tol_tr,flag);
    sigmatot=[sigmatot,sigmadrop];
    
    maxrank=max([maxrank,size(QU_2,2)]);

    beta = norm(RU);
    RU = RU/beta; 

    unew=QU_1*diag(RU)*QU_2';unew=unew(:);

    vold=QV_1*diag(RV)*QV_2';vold=vold(:);

    [Vt_1, Vt_2] = L_T(coeff,QU_1,QU_2,RU);
    [QV_1,QV_2,RV] = trunc_diag3(QV_1,-beta*diag(RV),Vt_1,eye(size(Vt_1,2)),QV_2,eye(length(RV)),Vt_2,eye(size(Vt_2,2)),r2,tol_tr,flag);
    sigmatot1=[sigmatot1,sigmadrop];
    maxrank=max([maxrank,length(RV)]);

    alfa=norm(RV);
    RV = RV/alfa;

    vnew=QV_1*diag(RV)*QV_2';vnew=vnew(:);

    rho1=(rho^2+beta^2)^(0.5);

    c=rho/rho1;
    s=beta/rho1;

    theta=s*alfa;
    rho=-c*alfa;
    phi1=c*phi;
    phi=s*phi;

    coef=phi1/rho1;
    [QX_1,QX_2,RX] = trunc_diag3(QX_1,diag(RX),QD_1,coef*diag(RD),QX_2,eye(length(RX)),QD_2,eye(length(RD)),r2,tol_tr,flag);
    maxrank=max([maxrank,length(RX)]);

    rks = [rks;length(RX)];

    [wrk1, wrk2] = L(coeff,QX_1,QX_2,RX);
    res_old=truenormres;
    truenormres=norm(C1*C2'-wrk1*wrk2','fro');

    [Qwrk1,Qwrk2,Rwrk] = trunc_diag4(wrk1,wrk2,r2,tol_tr,flag);
    [wrk1, wrk2] = L_T(coeff,Qwrk1,Qwrk2,Rwrk);
    truenormres2 = norm(AtC-wrk1*wrk2','fro');
    totres=[totres,truenormres];

    coef=theta/rho1;
    [QD_1,QD_2,RD] = trunc_diag3(QV_1,diag(RV),QD_1,-coef*diag(RD),QV_2,eye(length(RV)),QD_2,eye(length(RD)),r2,tol_tr,flag);
    maxrank=max([maxrank,length(RD)]);

    res = phi;
    res_2 = phi*alfa*abs(c);
    r_res.rel_res = [r_res.rel_res; truenormres/res0];
    r_res.nrml_res = [r_res.nrml_res; truenormres2/res0_2];
    a_res = [a_res; res/res0];
    disp([i, truenormres, truenormres2, res_2/res0_2])
    if truenormres/res0 <= tol || abs(truenormres-res_old)/truenormres<tol/10
        break, end

totrank=[totrank,maxrank];
end
X_1 = QX_1*diag(sqrt(RX)); X_2 = QX_2*diag(sqrt(RX));
disp([i, imax, tol, res_2/res0_2, res/res0])


figure(202)
semilogy(totres/totres(1),'d-','linewidth',4)
%semilogy(r_res,'linewidth',4)
%pause
%hold on
%semilogy(abs(uorth),'o','linewidth',4)
%semilogy(abs(vorth),'x','linewidth',4)
%%semilogy(totrank/totrank(end),'x','linewidth',4)
%rf=find(totrank/totrank(end)==1,1,'first');
%semilogy([rf,rf],[1e-15,10],'linewidth',4)
%hold off
%legend('true res normal eqn','orth U','orth V','max rank')
%xlabel('number of iterations')
%ylabel('loss of optimality')
% axis([0,150,1e-16,10]);
%hold off
%figure(203)
%semilogy(sigmatot,'o','linewidth',4)
%hold on
%semilogy(sigmatot1,'x','linewidth',4)
%semilogy([rf,rf],[1e-15,10],'linewidth',4)
%hold off 
% axis([0,150,1e-16,10]);
%legend('U','V')
%xlabel('number of iterations')
%ylabel('magnitude of dropped sing.values')


end

function[Y1,Y2] = L(coeff,QX1,QX2,RX)
% Operator L(X1*X2^T) := A*X1*X2^T*D+E*X1*X2^T*B = Y1*Y2^T.
% Y1 = [A*X1,E*X1];
% Y2 = [D'*X2, B'*X2];

RX = sqrt(RX);
RX = diag(RX);
l = length(coeff);
QX1=QX1*RX;
QX2=QX2*RX;
Y1 = []; Y2 = [];
for i = 1:l
    Y1 = [Y1, coeff{i}{1} * QX1];
    Y2 = [Y2, coeff{i}{2}' * QX2];
end

end



function[Y1, Y2] = L_T(coeff,QZ1,QZ2,RZ)
% % Operator L^T(Z1*Z2^T) := A^T*Z1*Z2^T*D^T+E^T*Z1*Z2^T*B^T = Y1*Y2^T.
% % 1 = [A'*Z1, E'*Z1];
% % Y2 = [D*Z2, B*Z2];
RZ = (sqrt(RZ));
RZ = diag(RZ);
l = length(coeff);
QZ1=QZ1*RZ;
QZ2=QZ2*RZ;

Y1 = []; Y2 = [];
for i = 1:l
    Y1 = [Y1, coeff{i}{1}' * QZ1];
    Y2 = [Y2, coeff{i}{2} * QZ2];
end
end

