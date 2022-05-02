clear all
close all
clc

load SV_Pos                         % position of satellites
load SV_Rho                         % pseudorange of satellites  

T = 1; % positioning interval
N = 25;% total number of steps
% State vector is as [x Vx y Vy z Vz b d].', i.e. the coordinate (x,y,z),
% the clock bias b, and their derivatives.

% Set f, see [1]
f = @(X) ConstantVelocity(X, T);

% Set Q, see [1]
Sf = 36;Sg = 0.01;sigma=5;         %state transition variance
Qb = [Sf*T+Sg*T*T*T/3 Sg*T*T/2;
	  Sg*T*T/2 Sg*T];
Qxyz = sigma^2 * [T^3/3 T^2/2;
                  T^2/2 T];
Q = blkdiag(Qxyz,Qxyz,Qxyz,Qb);

% Set initial values of X and P     
X = zeros(8,1);
X([1 3 5]) = [-2.168816181271560e+006 
                    4.386648549091666e+006 
                        4.077161596428751e+006];                 %Initial position
X([2 4 6]) = [0 0 0];                                            %Initial velocity
X(7,1) = 3.575261153706439e+006;                                 %Initial clock bias
X(8,1) = 4.549246345845814e+001;                                 %Initial clock drift
P = eye(8)*10;

fprintf('GPS positioning using EKF started\n') 
tic

for ii = 1:N
    % Set g
    g = @(X) PseudorangeEquation(X, SV_Pos{ii});                 % pseudorange equations for each satellites                

    % Set R
    Rhoerror = 36;                                               % variance of measurement error(pseudorange error)
    R = eye(size(SV_Pos{ii}, 1)) * Rhoerror; 

    % Set Z
    Z = SV_Rho{ii}.';                                            % measurement value

    [X,P] = Extended_KF(f,g,Q,R,Z,X,P);
    Pos_KF(:,ii) = X([1 3 5]).';                                 % positioning using Kalman Filter
    Pos_LS(:,ii) = Rcv_Pos_Compute(SV_Pos{ii}, SV_Rho{ii});      % positioning using Least Square as a contrast
    
    fprintf('KF time point %d in %d  ',ii,N)
    time = toc;
    remaintime = time * N / ii - time;
    fprintf('Time elapsed: %f seconds, Time remaining: %f seconds\n',time,remaintime)
end

% Plot the results. Relative error is used (i.e. just subtract the mean)

for ii = 1:3
    subplot(3,1,ii)
    plot(1:N, Pos_KF(ii,:) - mean(Pos_KF(ii,:)),'-r')
    hold on;grid on;
    plot(1:N, Pos_LS(ii,:) - mean(Pos_KF(ii,:)))
    legend('EKF','ILS')
    xlabel('Sampling index')
    ylabel('Error(meters)')
end
ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
text(0.5, 1,'\bf Relative positioning error in x,y and z directions','HorizontalAlignment','center','VerticalAlignment', 'top');