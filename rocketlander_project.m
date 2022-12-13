clc
clear all
close all

%% Define Parameters %%
% Linear and nonlinear control of a system.
mpc = import_mpctools();

%Nonlinear ODE
% x(1) = x
% x(2) = xdot
% x(3) = z
% x(4) = zdot
% x(5) = theta
% x(6) = thetadot

% Random variable standard deviations.
sig_v = 0.001; % Measurement noise.
sig_w = 3; % State noise.
sig_p = 30; % Prior.


% Constants.
Delta = 0.05;
Nx = 6;
Nu = 3;
Ny = 6;
Nw = Nx;
Nv = Ny;
Nsim = 100;
Nt = 25;
x0 = [10;0; 30;0; 0;0];
u0 = [0;0;0];
xtarget = zeros(Nx,1);
xtarget(1) = 0 ; xtarget(3) = 0; xtarget(5) = 0;
% utarget = zeros(Nu,1);

% Simulator function.
ode = @(x, u, w) odefunc(x, u, w);
model = mpc.getCasadiIntegrator(ode, Delta, [Nx, Nu, Nw], ...
    {'x','u', 'w'});
F = mpc.getCasadiFunc(ode, [Nx, Nu, Nw], {'x','u', 'w'}, {'f'}, ...
    'rk4', true(), 'Delta', Delta, 'M', 4);

% Define stage cost and terminal weight.
g = mpc.getCasadiFunc(@stagecost, [Nx,Nx,Nu], {'x','xtarget','u'}, {'g'});
H = mpc.getCasadiFunc(@termcost, [Nx,Nx], {'x','xtarget'}, {'H'});

for ii=1:10
    % First, simulate the system to get data.
    w1 = 20 + sig_w*randn(Nw, Nsim); %Accl. 12 m/s2 when wind velocity is 25 kmph
    w = zeros(Nw,Nsim);
    w(2,:) = w1(2,:);
    v = sig_v*randn(Nv, Nsim + 1);

    % Make controllers.
    
    par = struct('xtarget',xtarget,'w', zeros(Nw,1));
    Nlin = struct('x', Nx, 'u', Nu, 't', Nt);
    Nnonlin = Nlin;
    % Nnonlin.c = 2; % Use collocation for nonlinear MPC.

    guess = struct('x', repmat(x0, 1, Nt + 1), 'u', repmat(u0, 1, Nt));
    commonargs = struct('l', g, 'Vf', H, ...
        'lb', struct('x',[-2;-100000;-0.1;-100000;-1000000;-1000000], 'u', repmat([0;-130;-15], 1, Nt)), ...
        'ub', struct('x',[1000000;1000000;1000000;1000000;1000000;1000000], 'u', repmat([6486;130;15], 1, Nt)));

    tic()
    % NON LINEAR MPC SOLVER INITIALIZATION:

    solvers.NMPC = mpc.nmpc('f', F, 'N', Nnonlin, 'Delta', Delta, ...
        '**', commonargs, 'par',par, 'guess', guess);

    % Distubance cost
    meas = @(x) measfunc(x);  % measurment function
    h = mpc.getCasadiFunc(meas, [Nx], {'x'}, 'funcname', {'H'});

    % SIMULATING THE FIRST TIME HORIZON FOR INITAL INPUTS
    xsim = NaN(Nx, Nsim + 1);
    xsim(:,1) = x0;
    usim = zeros(Nu, Nsim + 1);
    ysim = NaN(Ny, Nsim + 1);
    yclean = NaN(Ny, Nsim + 1);
    P1 = sig_p.^2*eye(Nx);

    for t = 1:(Nsim + 1)
        yclean(:,t) = full(h(xsim(:,t)));
        ysim(:,t) = yclean(:,t) + v(:,t);

        if t <= Nsim
            xsim(:,t + 1) = full(model(xsim(:,t), usim(:,t), w(:,t)));
            if xsim(3,t+1) < 1e-1
                xsim(3,t+1) = 0;
                xsim(4,t+1) = 0;
            end
        end
    end

    xhat0 = x0;

    % Pick stage costs.
    lfunc = @(w, v) (w'*w) + v'*v/0.001;

    % lfunc is noise cost.
    l = mpc.getCasadiFunc(lfunc, [Nw, Nv], {'w', 'v'}, {'l'});
    guess = struct('x', repmat(x0, 1, Nt + 1));
    N = struct('x', Nx, 'u', Nu, 'w', Nw, 'y', Ny, 't', Nt);
    par = struct('Pinv', mpctools.spdinv(P1), 'x0bar', xhat0);

    % DISTURBANCE ESTIMATION SOLVER INITIALIZATION

    buildsolvertime = tic();
    solver = mpc.nmhe('f', F, 'l', l, 'h', h,'u', usim(:,1:Nt + 1), 'y', ysim(:,1:Nt + 1), 'N', N, 'guess', guess, ...
        'lb', struct('x',[-2;-100000;0;-100000;-1000000;-1000000]), ...
        'ub', struct('x',[1000000;1000000;1000000;1000000;1000000;1000000]), 'par', par, 'Nhistory', Nsim, ...
        'priorupdate', 'filtering');
    buildsolvertime = toc(buildsolvertime);
    fprintf('Building solver took %g s.\n', buildsolvertime);

    fprintf('Building controller took %.5g s.\n',toc());

    xsim = NaN(Nx, Nsim + 1);
    xsim(:,1) = x0;
    ysim = NaN(Ny, Nsim + 1);
    u = NaN(Nu, Nsim);
    ztarget = x0(3)*exp(-0.05*(1:Nsim))-1;
    zdottarget = linspace(0,2,Nsim);

    xhat = NaN(Nx, Nsim);
    xplot = NaN(Nx, Nsim + 1);
    yhat = NaN(Ny, Nt + 1, Nsim + 1);
    yplot = NaN(Ny, Nsim + 1);
    w_hat = NaN(Nw, Nsim);
    w_hat(:,1) = 0;
    uprev = zeros(Nu,1);
    xprior = repmat(zeros(Nx,1), 1, Nt);
    wprior = zeros(Nw, Nt);

    xfinal = 1;

    for t = 1:Nsim

        yclean(:,t) = full(h(xsim(:,t)));
        ysim(:,t) = yclean(:,t) ;

        % Get new measurement or extend horizon.
        if t > Nt + 1
            solver.newmeasurement(ysim(:,t),uprev);
        else
            solver.truncatehorizon(t - 1);
        end

        % Solve MHE problem and save state estimate.
        solver.solve();
        fprintf('Step %d: %s\n', t, solver.status);
        if ~isequal(solver.status, 'Solve_Succeeded')
            warning('Solver failure at time %d!', t);
            break
        end
        solver.saveestimate(); % Stores current estimate to struct.
        
        if t > 1 && t <= Nt + 1
            w_hat(:,t) = solver.var.w(:,t-1);
        elseif t > Nt + 1
            w_hat(:,t) = solver.var.w(:,end);
        end
        solver.saveguess(); % Stores current estimate to struct.

        %xtarget = [xsim(1,t) + (xtarget(1) - xsim(1,t))/(1+exp(-t)); 0; ztarget(t);0;0;0];
        %solvers.NMPC.par.xtarget = xtarget;
        % Set initial condition and solve.
        solvers.NMPC.fixvar('x', 1, xsim(:,t));
        solvers.par.w = w_hat(:,t);
        xguess = solvers.NMPC.guess.x;
        uguess = solvers.NMPC.guess.u;
        for k = 1:Nt
            xguess(:,k + 1) = full(F(xguess(:,k), uguess(:,k), w_hat(:,t)));
        end
        solvers.NMPC.guess.x = xguess;
        solvers.NMPC.solve();

        % Print status.
%         fprintf('%5s %d: %s\n', t, solvers.NMPC.status);
%         if ~isequal(solvers.NMPC.status, 'Solve_Succeeded')
%             warning('%s failed at time %d!', t);
%             break
%         end
        solvers.NMPC.saveguess();
        u(:,t) = solvers.NMPC.var.u(:,1);
%         xsim(:,t + 1) = full(model(xsim(:,t), u(:,t),w_hat(:,t)));
        uprev = u(:,t);
        % Stop early if the system is near the origin.
        %sqrt((x.(c)(1,t + 1))^2 + (x.(c)(3,t + 1))^2)
        %if sqrt((x.(c)(1,t + 1))^2 + (x.(c)(3,t + 1))^2) < 2e-1

        xsim(:,t + 1) = full(model(xsim(:,t), u(:,t),w(:,t)));
%         xsim(3,t+1)

        if (xsim(3,t+1)) < 0.003
            fprintf('%s at origin after %d iterations\n', t);
            %x.(c)(:,(t + 2):end) = 0;
            %x.(c)(1,(t + 2):end) = x.(c)(1,(t + 1));
            u(:,(t + 1):end) = 0;
            break
        end
    end


    % Make timeseries and phase plots.
    % figure();
    % colors = struct('NMPC', 'r');
    % for i = 1:1
    %     mpc.mpcplot(xsim, u,'fig', gcf());
    % end
    hold('on');
    plot(xsim(1,:), xsim(3,:),'-r');
    xlabel('x');
    ylabel('z');
    xlim([-1,15])
    %legend(controllers{:}, 'Location', 'SouthEast');
end

%% Subfunction %%

function dxdt = odefunc(x, u, w)
%Nonlinear ODE
% x(1) = x
% x(2) = xdot
% x(3) = z
% x(4) = zdot
% x(5) = theta
% x(6) = thetadot

%m = 25.22;
m = 35;
g = 9.81;
J = 482.2956;
L1 = 3.8677;
L2 = 3.7;
LN = 0.1892;

dxdt = [x(2); ((u(1) * sin(x(5) + u(3)) + u(2)*cos(x(5))) / m )+ w(2); ...
    x(4);((u(1)*cos(x(5) + u(3)) - u(2)*sin(x(5)) - m * g) / m); ...
    x(6);((-u(1)*sin(u(3))*(L1 + LN*cos(u(3))) + L2*u(2))/J)];

end%function

function g = stagecost(x,xtarget,u)
I_6 = eye(6);
I_6(1,1) = 100;
I_6(4,4) = 0;
I_6(2,2) = 0 ; %I_6(4,4) = 0; I_6(6,6) = 0; I_6(3,3) = 2;
Q = 10000*I_6;  % State Cost
R = 1*eye(3);  % Input Cost
%R(3,3) = 3;
g = (x-xtarget)'*Q*(x-xtarget) + (u)'*R*(u);
end

function H = termcost(x,xtarget)
P = 10000*eye(6);  % Terminal Cost
H = (x-xtarget)'*P*(x-xtarget);
end

function y = measfunc(x)
% Measurement function.
I_6 = eye(6);
%I_6(2,2) = 0 ;I_6(4,4) = 0; I_6(6,6) = 0;
y = I_6*[x(1); x(2); x(3); x(4); x(5); x(6)];
end%function

function cost = priorcost(x, d, xhat, w_hat, Pinv)
z = [x; d];
zhat = [xhat; w_hat];
dz = z - zhat;
cost = dz'*Pinv*dz;
end%function