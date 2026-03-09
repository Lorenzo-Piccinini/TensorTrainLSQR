% Test with Fashion_MNIST Dataset

format shorte
format compact

% Upload Fashion_MNIST Dataset
addpath(genpath('./Data'))
addpath(genpath('./oseledets_TT-Toolbox'))

mnist_extract
tr_i_1 = find(labelstrain == 5); % Indeces of sandals
tr_i_2 = find(labelstrain == 7); % Indeces of sneakers
tr_i_3 = find(labelstrain == 9); % Indeces of boots

ts_i_1 = find(labelstest == 5); % Indeces of sandals
ts_i_2 = find(labelstest == 7); % Indeces of sneakers
ts_i_3 = find(labelstest == 9); % Indeces of boots

% Number of modes of the tensor 
d = 3;
% Number of terms 
l = 6;

% Generate matrices
m = 60;
mm = m/l;
A_1 = imagestrain(:, tr_i_1(1:m));
A_2 = imagestrain(:, tr_i_2(1:m));
A_3 = imagestrain(:, tr_i_3(1:m));
A_1 = A_1/norm(A_1',1);
A_2 = A_2/norm(A_2',1);
A_3 = A_3/norm(A_3',1);

[u1,~,~] = svds(A_1,10); 
[u2,~,~] = svds(A_2,10); 
[u3,~,~] = svds(A_3,10); 

n = size(A_1,1);
AA = [A_1, A_2, A_3];

for k = 1:l
    terms{k}{1} = A_1(:, (k-1)*mm+1:k*mm);
    terms{k}{2} = A_2(:, (k-1)*mm+1:k*mm);
    terms{k}{3} = A_3(:, (k-1)*mm+1:k*mm);
end

for i = 1:d
    if l == 3
        [~, R1{i}] = qr(terms{1}{i},0);
        [~, R2{i}] = qr(terms{2}{i},0);
        [~, R3{i}] = qr(terms{3}{i},0);
        [m, ind] = min([cond(R1{i}), cond(R2{i}), cond(R3{i})]);
        rr = eval(['R',num2str(ind)]);
        new_terms{1}{i} = terms{1}{i}/rr{i};
        new_terms{2}{i} = terms{2}{i}/rr{i};
        new_terms{3}{i} = terms{3}{i}/rr{i};
    elseif l == 6
        [~, R1{i}] = qr(terms{1}{i},0);
        [~, R2{i}] = qr(terms{2}{i},0);
        [~, R3{i}] = qr(terms{3}{i},0);
        [~, R4{i}] = qr(terms{4}{i},0);
        [~, R5{i}] = qr(terms{5}{i},0);
        [~, R6{i}] = qr(terms{6}{i},0);
        [m, ind] = min([cond(R1{i}), cond(R2{i}), cond(R3{i}),...
            cond(R4{i}), cond(R5{i}), cond(R6{i})]);
        rr = eval(['R',num2str(ind)]);
        new_terms{1}{i} = terms{1}{i}/rr{i};
        new_terms{2}{i} = terms{2}{i}/rr{i};
        new_terms{3}{i} = terms{3}{i}/rr{i};
        new_terms{4}{i} = terms{4}{i}/rr{i};
        new_terms{5}{i} = terms{5}{i}/rr{i};
        new_terms{6}{i} = terms{6}{i}/rr{i};
    elseif l == 10
        [~, R1{i}] = qr(terms{1}{i},0);
        [~, R2{i}] = qr(terms{2}{i},0);
        [~, R3{i}] = qr(terms{3}{i},0);
        [~, R4{i}] = qr(terms{4}{i},0);
        [~, R5{i}] = qr(terms{5}{i},0);
        [~, R6{i}] = qr(terms{6}{i},0);
        [~, R7{i}] = qr(terms{7}{i},0);
        [~, R8{i}] = qr(terms{8}{i},0);
        [~, R9{i}] = qr(terms{9}{i},0);
        [~, R10{i}] = qr(terms{10}{i},0);
        [m, ind] = min([cond(R1{i}), cond(R2{i}), cond(R3{i}),...
            cond(R4{i}), cond(R5{i}), cond(R6{i}), cond(R7{i}), ...
            cond(R8{i}), cond(R9{i}), cond(R10{i})]);
        rr = eval(['R',num2str(ind)]);
        new_terms{1}{i} = terms{1}{i}/rr{i};
        new_terms{2}{i} = terms{2}{i}/rr{i};
        new_terms{3}{i} = terms{3}{i}/rr{i};
        new_terms{4}{i} = terms{4}{i}/rr{i};
        new_terms{5}{i} = terms{5}{i}/rr{i};
        new_terms{6}{i} = terms{6}{i}/rr{i};
        new_terms{7}{i} = terms{7}{i}/rr{i};
        new_terms{8}{i} = terms{8}{i}/rr{i};
        new_terms{9}{i} = terms{9}{i}/rr{i};
        new_terms{10}{i} = terms{10}{i}/rr{i};
    end
end
        

result_1 = [0, 0, 0];
result_2 = [0, 0, 0];
result_naive = [0, 0, 0];
result_orthodox = [0, 0, 0];

Params.tol = 1e-7;
Params.imax = 100;
Params.tol_tr = 1e-6;

% Choose the object ro recognize
% obj = 'san'; % Sandal
% obj = 'sne'; % Sneaker
 obj = 'boo'; % Boots
iter_tot = [];
 res_tot = [];
for es = 1:10

if obj == 'san'
    l_1 = length(ts_i_1);
    f = imagestest(:, ts_i_1(randi(l_1)));
elseif obj == 'sne'
    l_2 = length(ts_i_2);
    f = imagestest(:, ts_i_2(randi(l_2)));
elseif obj == 'boo'
    l_3 = length(ts_i_3);
    f = imagestest(:, ts_i_3(randi(l_3)));
end
f = f/norm(f);

F = tt_tensor({f, f, f});
X = tt_zeros([mm, mm, mm], 3);
    
tic;
[X, Res, iter] = TT_Tensorized_LSQR(new_terms, F, Params, X);
t_tt = toc;
t_tot = t_tot + t_tt;

for j = 1:X.d
    X = ttm(X, j, inv(rr{j})');
end
lx = OpL(terms,X);
ltlx = OpL_T(terms, lx);
ltf = OpL_T(terms, F);
res_tot = [res_tot; norm(ltf-ltlx)/norm(ltf)];
iter_tot = [iter_tot; iter];

Y = OpL(terms, X);
q1 = ttm(ttm(Y,2,ones(n,1)), 3, ones(n,1));
w1 = norm(ttm(q1,1,f));
q2 = ttm(ttm(Y,1,ones(n,1)), 3, ones(n,1));
w2 = norm(ttm(q2,2,f));
q3 = ttm(ttm(Y,1,ones(n,1)), 2, ones(n,1));
w3 = norm(ttm(q3,3,f));

% OPTION 1
vec = [w1, w2, w3];
[val1, idx1] = max(vec);
result_1(idx1) = result_1(idx1) + 1;

% OPTION 2
% cx = X.core;
% rks = X.r;
% d = X.d;
% pos = X.ps;
% mx = X.n;
% G1 = cx(pos(1):pos(2)-1);
% G2 = cx(pos(2):pos(3)-1);
% G3 = cx(pos(3):pos(4)-1);
% G1 = reshape(G1,mx(1),rks(2));
% G2 = reshape(G2,rks(2),mx(2),rks(3));
% G3 = reshape(G3,rks(3),mx(3));
% [u3, s3, v3] = svd(G3');
% [u1,s1,v1] = svd(G1);
% tx2 = tensor(G2);
% U2 = nvecs(tx2,2,1);
% U1 = u1(:,1); U3 = u3(:,1);
% UU1 = U1/norm(U1);
% UU2 = U2/norm(U2);
% UU3 = U3/norm(U3);
% UU = tt_tensor({UU1,UU2,UU3});
lu = OpL(terms, X);
cores_lu = lu.core;
rks_lu = lu.r;
d_lu = lu.d;
pos_lu = lu.ps;
P1 = cores_lu(pos_lu(1):pos_lu(2)-1);
P2 = cores_lu(pos_lu(2):pos_lu(3)-1);
P3 = cores_lu(pos_lu(3):pos_lu(4)-1);
P1 = reshape(P1,784,rks_lu(2));
P2 = reshape(P2, rks_lu(2), 784, rks_lu(3));
P3 = reshape(P3, rks_lu(3), 784);
[uu1,~,~] = svd(P1,0); UU1 = uu1;
[uu3,~,~] = svd(P3',0); UU3 = uu3;
UU2 = nvecs(tensor(P2),2,size(UU1,2));
%UU1 = UU1/norm(UU1);UU2 = UU2/norm(UU2);UU3 = UU3/norm(UU3);
% dist_0 = svd(f' * orth(UU1));
% dist_1 = svd(f' * orth(UU2)); 
% dist_2 = svd(f' * orth(UU3));
dist_0 = norm(f' * UU1);
dist_1 = norm(f' * UU2); 
dist_2 = norm(f' * UU3);
[val2, idx2] = max([dist_0, dist_1, dist_2]);

result_2(idx2) = result_2(idx2) + 1;


% Naive classification
% xx = AA\f;
% xr = reshape(xx, m, 3);
% [vx, ix] = max(sqrt(sum(xr.*xr)));
% result_naive(ix) = result_naive(ix) + 1;

% Orthodox 
% w1 = norm(f-u1*(u1'*f)); w2 = norm(f-u2*(u2'*f)); w3 = norm(f-u3*(u3'*f));
% [~,oridx]=min([w1,w2,w3]);
% result_orthodox(oridx) = result_orthodox(oridx) + 1;

% result_1,
% result_2,
% result_naive,
% result_orthodox,

end
% result_1,
% result_2,
% result_naive,
% result_orthodox,


