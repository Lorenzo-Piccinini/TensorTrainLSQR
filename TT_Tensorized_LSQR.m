function[X, Res] = TT_Tensorized_LSQR(values, Rhs, Params, X)

% function[X, Res] = TT_Tensorized_LSQR(values, Rhs, Params, X)
%
% Function solving Tensor equation
%
%   \sum_{i=1}^{nterms} X x_1 A_i^(1) x_2 A_i^(2) x_3 A_i^(3) = Rhs
%
% in Tensor-Train (TT) format.
% INPUTS:
%   - values: structure storing the matrices that define the tensor
%             operator
%   - Rhs: right-hand side in TT
%   - Params: Params.tol = 1e-8;
%             Params.imax = 500;
%             Params.X = X;
%             delta=1e-9; 
%             Params.tol_tr = delta;
%             Params.rank_tr =51;
%   - X: starting solution given in TT
%
% OUTPUTS:
%   - X: solution in TT
%   - Res: structure containing absolute and relative residual history
%
% To use this code cite 
% L. Piccinini, and V. Simoncini "TT-LSQR for tensor least squares problems
% and application in data mining". Numerical Algorithms, 2025.
%
% DOI: 10.1007/s11075-025-02204-8



beta = norm(Rhs);
D = Rhs;

U = D; 
U = U/beta;
V = OpL_T(values, U);
V = round(V, Params.tol_tr,Params.rank_tr);

alfa = norm(V);
V = V/alfa;
W = V;

phi_bar = beta;
rho_bar = alfa;

res0 = beta;
Res.real_abs = [res0];
Res.real_rel = [1];

res_old = res0;
totres = res0;
totresnormal = alfa;

n_terms = length(values);
d = Rhs.d;

Ftot = OpL_T(values, Rhs);
res0_ne = norm(Ftot);

i = 0;
fprintf('iteation  res_ne_true   res_true\n')

while ( i < Params.imax )
    i = i+1;
    
    wrk1 = OpL(values, V);
    wrk1 = round(wrk1, Params.tol_tr,Params.rank_tr);

    U = wrk1 - alfa*U;
    U = round(U, Params.tol_tr,Params.rank_tr);
    clear wrk1

    beta = norm(U);
    U = U/beta;

    wrk2 = OpL_T(values, U);
    wrk2 = round(wrk2, Params.tol_tr,Params.rank_tr);

    V = wrk2 - beta*V;
    V = round(V, Params.tol_tr,Params.rank_tr);
    clear wrk2

    alfa = norm(V);
    V = V/alfa;

    rho = sqrt(rho_bar^2 + beta^2);
    c = rho_bar/rho;
    s = beta/rho;
    theta = s*alfa;
    rho_bar = -c*alfa;
    phi = c*phi_bar;
    phi_bar = s*phi_bar;

    res_est = phi_bar;
    res_ne_est = phi_bar*alfa*abs(c);

    X = X + (phi/rho)*W;
    X = round(X, Params.tol_tr,Params.rank_tr);

    LX = OpL(values, X);
    LTLX = OpL_T(values, LX);
    res_true = norm(Rhs- LX);
    clear LX
    
    res_truenormal = norm(Ftot- LTLX);
    totres = [totres; res_true];
    totresnormal = [totresnormal; res_truenormal];
    clear LTLX

    if res_truenormal/res0_ne <= Params.tol
        fprintf('  %d  %.4e %.4e %.4e\n', [i,res_true/res0_ne, res_true, res_truenormal])
        break, end
    
    res_old = res_true;

    Res.real_abs = [Res.real_abs; res_true];
    Res.real_rel = [Res.real_rel; res_true/res0];

    W = V - (theta/rho)*W;
    W = round(W, Params.tol_tr,Params.rank_tr);

    fprintf('  %d  %.4e %.4e\n', [i,res_true/res0, res_truenormal/res0_ne])

end
end

%---------------------------------------------------------

function[Y] = OpL(values, X)
% Function to apply the operator L
d = X.d;
n_terms = length(values);
for k = 1:n_terms
    wrk = X;
    for j = 1:d
        wrk = ttm(wrk, j, values{k}{j}');
    end
    if k==1, Y=wrk; else, Y = Y + wrk;end
end
end


function[Y] = OpL_T(values, X)
% Function to apply the operator L^T
d = X.d;
n_terms = length(values);
for k = 1:n_terms
    wrk = X;
    for j = 1:d
        wrk = ttm(wrk, j, values{k}{j});
    end
    if k==1, Y=wrk; else, Y = Y + wrk;end
end
end
