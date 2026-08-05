function run_two_parameter_hopf_curve()
%UNSUPPORTED_BY_JAXCONT Continue MatCont's catalytic-oscillator Hopf curve.
setup_matcont();
p = [2.5; 2.204678; 10; 0.0675; 1; 0.1; 0.4];
[x0, ~] = init_EP_EP(@cataloscill, [0.001137; 0.891483; 0.062345], p, 2);
opt = contset;
opt = contset(opt, 'MaxStepsize', 0.025);
opt = contset(opt, 'MaxNumPoints', 78);
opt = contset(opt, 'Singularities', 1);
[x, ~, s] = cont(@equilibrium, x0, [], opt);
labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
hopf = find(strcmp(labels, 'H'), 1);
assert(~isempty(hopf), 'Catalytic oscillator Hopf seed not found.');
p(2) = x(end, s(hopf).index);
[seed, tangent] = init_H_H(@cataloscill, x(1:3, s(hopf).index), p, [2, 7]);
opt = contset(opt, 'MaxNumPoints', 25);
[curve, ~] = cont(@hopf, seed, tangent, opt);
assert(size(curve, 2) > 1);
fprintf('UNSUPPORTED_BY_JAXCONT two-parameter Hopf curve: %d points\n', size(curve, 2));
end
