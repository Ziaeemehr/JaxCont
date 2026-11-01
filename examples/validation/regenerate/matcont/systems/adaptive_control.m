function out = adaptive_control
%ADAPTIVE_CONTROL MatCont adaptx system (alpha, beta).
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

function dydt = fun_eval(~, state, alpha, beta)
dydt = [state(2); state(3); ...
    -alpha * state(3) - beta * state(2) - state(1) + state(1)^2];
end

function [tspan, y0, options] = init
y0 = [0; 0; 0];
handles = feval(@adaptive_control);
options = odeset('Jacobian', handles{3}, 'JacobianP', handles{4}, ...
    'Hessians', handles{5}, 'HessiansP', handles{6});
tspan = [0 10];
end

function jac = jacobian(~, state, alpha, beta)
jac = [0, 1, 0; 0, 0, 1; -1 + 2 * state(1), -beta, -alpha];
end

function jacp = jacobianp(~, state, ~, ~)
jacp = [0, 0; 0, 0; -state(3), -state(2)];
end

function hess = hessians(~, ~, ~, ~)
hess = zeros(3, 3, 3);
hess(1, 1, 1) = 0;
hess(3, 1, 1) = 2;
end

function hessp = hessiansp(~, ~, ~, ~)
hessp = zeros(3, 3, 2);
hessp(3, 3, 1) = -1;
hessp(3, 2, 2) = -1;
end
