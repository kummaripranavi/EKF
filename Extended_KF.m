
% State Equation:
%     X(n+1) = f(X(n)) + w(n)
%     where the state X has the dimension N-by-1
% Observation Equation:
%     Z(n) = g(X(n)) + v(n)
%     where the observation y has the dimension M-by-1
%     w ~ N(0,Q) is gaussian noise with covariance Q
%     v ~ N(0,R) is gaussian noise with covariance R  


%Algorithm for Extended Kalman Filter:
% Linearize input functions f and g to get fy(state transition matrix)
% and H(observation matrix) for an ordinary Kalman Filter:


% State Equation:
%     X(n+1) = fy * X(n) + w(n)
% Observation Equation:

%     Z(n) = H * X(n) + v(n)
%
% 1. Xp = f(Xi)                     : One step projection, also provides 
%                                     linearization point
% 
% 2. 
%          d f    |
% fy = -----------|                 : Linearize state equation, fy is the
%          d X    |X=Xp               Jacobian of the process model
%       
% 
% 3.
%          d g    |
% H  = -----------|                 : Linearize observation equation, H is
%          d X    |X=Xp               the Jacobian of the measurement model
%             
%       
% 4. Pp = fy * Pi * fy' + Q         : Covariance of Xp
% 
% 5. K = Pp * H' * inv(H * Pp * H' + R): Kalman Gain
% 
% 6. Xo = Xp + K * (Z - g(Xp))      : Output state
% 
% 7. Po = [I - K * H] * Pp          : Covariance of Xo

function [Xo,Po] = Extended_KF(f,g,Q,R,Z,Xi,Pi)
N_state = size(Xi, 1);    

[Xp, ~] = f(Xi);%1

[~, fy] = f(Xp);%2

[gXp, H] = g(Xp);%3

Pp = fy * Pi * fy.' + Q;%4

K = Pp * H' / (H * Pp * H.' + R);%5
    
Xo = Xp + K * (Z - gXp);%6

I = eye(N_state, N_state);
Po = (I - K * H) * Pp;%7
    
 