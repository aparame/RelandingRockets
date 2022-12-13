clc
clear all

A = [1.5 0 0;0 1.2 0; 0 0 0.5];

B=[0.0835;0.1332;0.1734]; % Previously Chosen B - Matrix
Q=rand(3);
out = Q*Q.';
R=0.001;
Pf=Q;  % Riccati matrix at N, could be diferent from Q.
% iterate N times.
n = 20;
for i = 1:n
    K=-inv(B'*Pf*B+R)*B'*Pf*A;
    Pf=Q+A'*Pf*A-A'*Pf*B*inv(B'*Pf*B+R)*B'*Pf*A
end

sol_1 = eig(A+B*K)
[~,~,L] = idare(A,B,Q,R)