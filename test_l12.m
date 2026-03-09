% Test for classification with Cranfield or Medline dataset

m=60; ivalf=1;
format short e

addpath(genpath('./oseledets_TT-Toolbox'));

% Choose the dataset
% load A_cran.mat; A = A_cran; load Acran_idx; fprintf('Acran\n')
load A_med.mat; A = A_med; load Amed_idx; fprintf('Amed\n')

[n,~]=size(A);
ng = 3;

% Preprocessing data
g1 = find(idx == 1);    s1 = sum(A(:,g1),2);
g2 = find(idx == 2);    s2 = sum(A(:,g2),2);
g3 = find(idx == 3);    s3 = sum(A(:,g3),2);
g4 = 0;
g5 = 0;
g6 = 0;

[m1, i1] = sort(s1, 'descend');
[m2, i2] = sort(s2, 'descend');
[m3, i3] = sort(s3, 'descend');

[vord,iord] = sort([length(g1),length(g2),length(g3),length(g4),length(g5),length(g6)],'descend');

gx1 = eval(['g', num2str(iord(1))]);
gx2 = eval(['g', num2str(iord(2))]);
gx3 = eval(['g', num2str(iord(3))]);

% Constructing the matrices for the tensor operator
A1 = A(:, gx1(1:m)); 
A2 = A(:, gx2(1:m)); 
A3 = A(:, gx3(1:m)); 
A1=A1/norm(A1',1);
A2=A2/norm(A2',1);
A3=A3/norm(A3',1);
AA=[A1 A2 A3 ];
[u1,~,~]=svds(A1,10); 
[u2,~,~]=svds(A2,10); 
[u3,~,~]=svds(A3,10); 

% Number of terms is 12
l = 12;
mm = m/l;

fprintf('num terms: %d,  num modes: %d, matrix dim per mode: %d x %d\n', l,3,n,mm)

values1{1} = A1(:, 1:mm); values1{2} = A2(:, 1:mm); values1{3} = A3(:, 1:mm); 
values2{1} = A1(:, mm+1:2*mm); values2{2} = A2(:, mm+1:2*mm); values2{3} = A3(:, mm+1:2*mm);
values3{1} = A1(:, 2*mm+1:3*mm); values3{2} = A2(:, 2*mm+1:3*mm); values3{3} = A3(:, 2*mm+1:3*mm);
values4{1} = A1(:, 3*mm+1:4*mm); values4{2} = A2(:, 3*mm+1:4*mm); values4{3} = A3(:, 3*mm+1:4*mm);
values5{1} = A1(:, 4*mm+1:5*mm); values5{2} = A2(:, 4*mm+1:5*mm); values5{3} = A3(:, 4*mm+1:5*mm); 
values6{1} = A1(:, 5*mm+1:6*mm); values6{2} = A2(:, 5*mm+1:6*mm); values6{3} = A3(:, 5*mm+1:6*mm); 
values7{1} = A1(:, 6*mm+1:7*mm); values7{2} = A2(:, 6*mm+1:7*mm); values7{3} = A3(:, 6*mm+1:7*mm); 
values8{1} = A1(:, 7*mm+1:8*mm); values8{2} = A2(:, 7*mm+1:8*mm); values8{3} = A3(:, 7*mm+1:8*mm);
values9{1} = A1(:, 8*mm+1:9*mm); values9{2} = A2(:, 8*mm+1:9*mm); values9{3} = A3(:, 8*mm+1:9*mm);
values10{1} = A1(:, 9*mm+1:10*mm); values10{2} = A2(:, 9*mm+1:10*mm); values10{3} = A3(:, 9*mm+1:10*mm); 
values11{1} = A1(:, 10*mm+1:11*mm); values11{2} = A2(:, 10*mm+1:11*mm); values11{3} = A3(:, 10*mm+1:11*mm); 
values12{1} = A1(:, 11*mm+1:12*mm); values12{2} = A2(:, 11*mm+1:12*mm); values12{3} = A3(:, 11*mm+1:12*mm); 

% Final tensor operator
values{1} = values1;
values{2} = values2;
values{3} = values3;
values{4} = values4;
values{5} = values5;
values{6} = values6;
values{7} = values7;
values{8} = values8;
values{9} = values9;
values{10} = values10;
values{11} = values11;
values{12} = values12;

ix_tot=0;
ival_tot=0;
ivalw_tot=0;
oridx_tot=0;
ivalten_tot=0;

% Running n_runs times the algorithm to compute the average accuracy
n_runs = 20;
for kk=1:n_runs;  
  kk
  switch ivalf

 % Choosing the right-hand side (Image to recognize)
 case 1
    f = A(:, gx1(m+10+kk)); itest=1;
 case 2
    f = A(:, gx2(m+10+kk)); itest=2;
 case 3
    f = A(:, gx3(m+10+kk)); itest=3;
 otherwise
    break
  end
  f=f/norm(f);

  fprintf('test class is %d\n', itest)
  F = tt_tensor({f,f,f});

  X0 = tt_zeros([mm,mm,mm], 3);
  Params.tol = 1e-4;
  Params.imax = 200;
  Params.rank_tr = 200;
  Params.tol_tr = 1e-4;
  Params.r = 1000;
  
  tic;
  [X, Res] = TT_Tensorized_LSQR(values, F, Params, X0);
  t_tt = toc
  
  y = opl_val1(values, X);    
  %Criterion 1
  [mival,ival]=max([norm(ttm(y,1,f)), norm(ttm(y,2,f)), norm(ttm(y,3,f))]);
  fprintf('tensorized: test is guessed mode %d\n',ival)
  if ival==itest, ival_tot=ival_tot+1;end


  %Criterion 2
  q1= ttm(ttm(y,2,ones(n,1)),3,ones(n,1)); 
  w1=norm(ttm( q1,1,f));
  q1= ttm(ttm(y,1,ones(n,1)),3,ones(n,1)); 
  w2=norm(ttm(q1,2,f));
  q1= ttm(ttm(y,1,ones(n,1)),2,ones(n,1)); 
  w3=norm(ttm(q1,3,f));
  [~,ivalw]=max([w1,w2,w3]);
  if ivalw==itest, ivalw_tot=ivalw_tot+1;end
  fprintf('tensorized check2: test is guessed mode %d\n',ivalw)
clear q1 X Res


lu = y;
cores_lu = lu.core;
rks_lu = lu.r;
d_lu = lu.d;
pos_lu = lu.ps;
P1 = cores_lu(pos_lu(1):pos_lu(2)-1);
P2 = cores_lu(pos_lu(2):pos_lu(3)-1);
P3 = cores_lu(pos_lu(3):pos_lu(4)-1);
P1 = reshape(P1,n,rks_lu(2));
P2 = reshape(P2, rks_lu(2), n, rks_lu(3));
P3 = reshape(P3, rks_lu(3), n);
[UU1,~,~] = svd(P1,0); 
[UU3,~,~] = svd(P3',0); 
UU2 = nvecs(tensor(P2),2,size(UU1,2));
dist_0 = norm(f' * UU1);
dist_1 = norm(f' * UU2);
dist_2 = norm(f' * UU3);
[val2, idx2] = max([dist_0, dist_1, dist_2]);


  if idx2==itest, ivalten_tot=ivalten_tot+1;end
  fprintf('tensorized check2plus: test is guessed mode %d\n',idx2)
clear lu P1 P2 P3 UU1 UU2 UU3


  
% standard query matching
  AA=[A1 A2 A3 ];
  x = AA\f;
  xr=reshape(x,m,ng); 
  [vx,ix]=max(sqrt(sum(xr.*xr)));
  fprintf('matrix: test is guessed mode %d\n',ix)
  if ix==itest, ix_tot=ix_tot+1;end
clear x y F


% orthodox query matching
  w1=norm(f-u1*(u1'*f)); w2=norm(f-u2*(u2'*f)); w3=norm(f-u3*(u3'*f));
  [~,oridx]=min([w1,w2,w3]);
  if oridx==itest, oridx_tot=oridx_tot+1;end
fprintf('orthodox comparison, guessed mode %d \n',oridx)


end


fprintf('percentage of correct classification: tensorized1 %d, tensorized2 %d, tensorized2plus %d, matrix %d orth %d \n', ival_tot/kk*100,ivalw_tot/kk*100,...
ivalten_tot/kk*100,ix_tot/kk*100,oridx_tot/kk*100);
