function out = cubic_fold
%CUBIC_FOLD MatCont system for xdot = r + x - x^3/3.

out{1} = @init;
out{2} = @fun_eval;
out{3} = @jacobian;
out{4} = @jacobianp;
out{5} = @hessians;
out{6} = @hessiansp;
out{7} = @der3;
out{8} = [];
out{9} = [];
end

function dydt = fun_eval(~, state, r)
x = state(1);
dydt = r + x - x^3 / 3;
end

function [tspan, y0, options] = init
y0 = -2;
handles = feval(@cubic_fold);
options = odeset('Jacobian', handles(3), 'JacobianP', handles(4), ...
    'Hessians', handles(5), 'HessiansP', handles(6));
tspan = [0 10];
end

function jac = jacobian(~, state, ~)
x = state(1);
jac = 1 - x^2;
end

function jacp = jacobianp(~, ~, ~)
jacp = 1;
end

function hess = hessians(~, state, ~)
x = state(1);
hess = -2 * x;
end

function hessp = hessiansp(~, ~, ~)
hessp = 0;
end

function tens3 = der3(~, ~, ~)
tens3 = -2;
end

