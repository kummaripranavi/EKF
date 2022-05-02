                        
function [Xo,Po] = Extended_KF(f,g,Q,R,Z,X,P,Xstate)
N_state = length(Xstate);
N_obs = length(Z);
    
Xp = subs(f,Xstate,X);%1

fy = subs(jacobian(f,Xstate),Xstate,Xp);%2

H = subs(jacobian(g,Xstate),Xstate,Xp);%3

Pp = fy * P * fy.' + Q;%4

K = Pp * H' * inv(H * Pp * H.' + R);%5
    
Xo = Xp + K * (Z - subs(g,Xstate,Xp));%6

I = eye(N_state,N_state);
Po = [I - K * H] * Pp;%7
    
 