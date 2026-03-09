% Nonsymmetric problem stemming from 3d PDE
% Requires oseledets_TT-Toolbox
% https://github.com/oseledets/TT-Toolbox

addpath('./oseledets_TT-Toolbox');

format shorte
format compact

% Dimensions of the problem
N = []; M = [];
d = 3;
% ntot=30;
ntot=50;
n1 = ntot; N = [N, n1]; m1 = ntot; M = [M, m1];
n2 = ntot; N = [N, n2]; m2 = ntot; M = [M, m2];
n3 = ntot; N = [N, n3]; m3 = ntot; M = [M, m3];

h1 = 1/n1;
h2 = 1/n2;
h3 = 1/n3;

x_nodes = linspace(0,1,n1+1)';
y_nodes = linspace(0,1,n2+1)';
z_nodes = linspace(0,1,n3+1)';



% Building right-hand side
a = ones(n1+1,1); a = a/norm(a);
b = ones(n2+1,1); b = b/norm(b);
c = ones(n3+1,1); c = c/norm(c);

% Building the coefficient matrices
aa=exp(-linspace(0,1,2*(m1)+4))';
e = ones(m1+1,1); D= spdiags([-e,e],-1:0,m1+2,m1+1);
A1=-m1^2*D'*spdiags(aa(2:2:2*m1+4),0:0,m1+2,m1+2)*D;
A2=A1; A3=A1;
A4 = m3/2*spdiags([-e 0*e e],-1:1,m3+1,m3+1); 
x = x_nodes;
Phi1 = sparse(diag(exp(-x)));
Ni2 = sparse(diag(exp(-x)));
Psi3 = sparse(diag(exp(-x)));
Psi1 = sparse(diag(ones(n1+1,1)));
Ni1 = sparse(diag(ones(n1+1,1)));
Phi2 =sparse(diag(ones(n1+1,1)));
Psi2 =sparse(diag(ones(n1+1,1)));
Phi3 =sparse(diag(ones(n1+1,1)));
Ni3 =sparse(diag(ones(n1+1,1)));
Psi4 =sparse(diag(ones(n1+1,1)));
n1=n1+1;

% Structure containing the tensor operator
values1{1} = Phi1 * A1; values1{2} = Ni1; values1{3} = Psi1;
values2{1} = Phi2; values2{2} = Ni2 * A2; values2{3} = Psi2;
values3{1} = Phi3; values3{2} = Ni3; values3{3} = Psi3 * A3+(Psi4*A4)';
m=n1;

preconditioning = 1;
if preconditioning
    for i = 1:3
        [~, R1{i}] = qr(values1{i},0); 
        [~, R2{i}] = qr(values2{i},0); 
        [~, R3{i}] = qr(values3{i},0);
    
        [~, ind] = min([condest(R1{i}), condest(R2{i}), condest(R3{i})]);
    
        rr = eval(['R',num2str(ind)]);
    
        new_values1{i} = values1{i}/rr{i};    
        new_values2{i} = values2{i}/rr{i};
        new_values3{i} = values3{i}/rr{i};
    end
else
    new_values{1} = values1;
    new_values{2} = values2;
    new_values{3} = values3;
end

% Defining the updated structure containing the tensor operator
new_values{1} = new_values1;
new_values{2} = new_values2;
new_values{3} = new_values3;

% Building the matrix operator for the matrix-oriented LSQR
coeff{1}{1} = kron(new_values1{2}, new_values1{1});
coeff{1}{2} = new_values1{3};
coeff{2}{1} = kron(new_values2{2}, new_values2{1});
coeff{2}{2} = new_values2{3};
coeff{3}{1} = kron(new_values3{2}, new_values3{1});
coeff{3}{2} = new_values3{3};

% Matrix-oriented right-hand side
C1 = kron(b,a); C2 = c;
clear new_values1 new_values2 new_values3 new_values4 R1 R2 R3 R4
clear values1 values2 values3 values4
p =1;

% Creating the TT-format RHS
rhs_vec = {a,b,c};
F = tt_tensor(rhs_vec);
clear K Qa Qb Qc rhs_vec
F = F/norm(F);


% Setting up parameters
X = tt_zeros([m,m,m],3);
Params.tol = 1e-8;
Params.imax = 500;
Params.X = X;
delta=1e-9; 
Params.tol_tr = delta;
Params.rank_tr =51;


fprintf("Running: TT-LSQR\n")
tic;
[Y2,Res] = TT_Tensorized_LSQR(new_values, F, Params, X);
t_tt_prec = toc;
pause


% Setting up parameters for matrix-oriented code
tol = 1e-8;
imax = Params.imax;
tol_tr = delta;
r = 51;
fprintf("Running: Matrix-oriented LSQR\n")
tic;
[X_1,X_2,r_res,a_res,rks,DD,totres]=lsqr_matrix_multi(coeff,C1,C2,tol,imax,tol_tr,r);
t_matrix = toc;

disp_plot = 1;
if displ plot
    semilogy(Res.real_abs/Res.real_abs(1))
    hold on
    semilogy(totres/totres(1))
    legend('TT','MATRIX')
    hold off
end

