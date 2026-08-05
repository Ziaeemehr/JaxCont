function out = radial_cycle_system
%RADIAL_CYCLE Exact cycle r'=r(rho-r^2), theta'=1.
out{1} = @init;
out{2} = @fun_eval;
out{3} = @jacobian;
out{4} = @jacobianp;
out{5} = @hessians;
out{6} = @hessiansp;
out{7} = [];
out{8} = [];
out{9} = [];
end

function dydt = fun_eval(~, state, rho)
x = state(1); y = state(2); r2 = x^2 + y^2;
dydt = [(rho - r2) * x - y; (rho - r2) * y + x];
end

function [tspan, y0, options] = init
y0 = [0; 0];
handles = feval(@radial_cycle_system);
options = odeset('Jacobian', handles{3}, 'JacobianP', handles{4}, ...
    'Hessians', handles{5}, 'HessiansP', handles{6});
tspan = [0 10];
end

function jac = jacobian(~, state, rho)
x = state(1); y = state(2);
jac = [rho - 3*x^2 - y^2, -2*x*y - 1; ...
    -2*x*y + 1, rho - x^2 - 3*y^2];
end

function jacp = jacobianp(~, state, ~)
jacp = state(:);
end

function hess = hessians(~, state, ~)
x = state(1); y = state(2);
hess(:, :, 1) = [-6*x, -2*y; -2*y, -2*x];
hess(:, :, 2) = [-2*y, -2*x; -2*x, -6*y];
end

function hessp = hessiansp(~, ~, ~)
hessp(:, :, 1) = eye(2);
end
