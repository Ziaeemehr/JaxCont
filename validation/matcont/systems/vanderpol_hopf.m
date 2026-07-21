function out = vanderpol_hopf
%VANDERPOL_HOPF MatCont system for the Van der Pol equilibrium branch.

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

function dydt = fun_eval(~, state, mu)
x = state(1);
y = state(2);
dydt = [y; mu * (1 - x^2) * y - x];
end

function [tspan, y0, options] = init
y0 = [0; 0];
handles = feval(@vanderpol_hopf);
options = odeset('Jacobian', handles(3), 'JacobianP', handles(4), ...
    'Hessians', handles(5), 'HessiansP', handles(6));
tspan = [0 10];
end

function jac = jacobian(~, state, mu)
x = state(1);
y = state(2);
jac = [0, 1; -1 - 2 * mu * x * y, mu * (1 - x^2)];
end

function jacp = jacobianp(~, state, ~)
x = state(1);
y = state(2);
jacp = [0; (1 - x^2) * y];
end

function hess = hessians(~, state, mu)
x = state(1);
y = state(2);
hess = zeros(2, 2, 2);
hess(:, :, 2) = [-2 * mu * y, -2 * mu * x; -2 * mu * x, 0];
end

function hessp = hessiansp(~, state, ~)
x = state(1);
y = state(2);
hessp = zeros(2, 2, 1);
hessp(:, :, 1) = [0, 0; -2 * x * y, 1 - x^2];
end

function tens3 = der3(~, ~, mu)
tens3 = zeros(2, 2, 2, 2);
tens3(1, 1, 2, 2) = -2 * mu;
tens3(1, 2, 1, 2) = -2 * mu;
tens3(2, 1, 1, 2) = -2 * mu;
end

