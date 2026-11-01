function out = vanderpol_hopf
%VANDERPOL_HOPF MatCont system for the Van der Pol equilibrium branch.
out{1} = @init;
out{2} = @fun_eval;
out{3} = @jacobian;
out{4} = @jacobianp;
out{5} = [];
out{6} = [];
out{7} = [];
out{8} = [];
out{9} = [];
end

function dydt = fun_eval(~, state, mu)
x = state(1); y = state(2);
dydt = [y; mu * (1 - x^2) * y - x];
end

function [tspan, y0, options] = init
y0 = [0; 0];
handles = feval(@vanderpol_hopf);
options = odeset('Jacobian', handles{3}, 'JacobianP', handles{4});
tspan = [0 10];
end

function jac = jacobian(~, state, mu)
x = state(1); y = state(2);
jac = [0, 1; -1 - 2 * mu * x * y, mu * (1 - x^2)];
end

function jacp = jacobianp(~, state, ~)
x = state(1); y = state(2);
jacp = [0; (1 - x^2) * y];
end
