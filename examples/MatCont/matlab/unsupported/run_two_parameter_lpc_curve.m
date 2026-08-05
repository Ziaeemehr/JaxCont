function run_two_parameter_lpc_curve()
%UNSUPPORTED_BY_JAXCONT Continue MatCont's Morris-Lecar LPC curve.
setup_matcont();
p = [0.11047; 0.1];
[x0, ~] = init_EP_EP(@MLfast, [0.047222; 0.32564], p, 1);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'MaxNumPoints', 65);
opt = contset(opt, 'MinStepsize', 1e-5);
opt = contset(opt, 'MaxStepsize', 0.01);
opt = contset(opt, 'Backward', 1);
[x, ~, s] = cont(@equilibrium, x0, [], opt);
labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
hopf = find(strcmp(labels, 'H'), 1);
assert(~isempty(hopf), 'Morris-Lecar Hopf seed not found.');
p(1) = x(end, s(hopf).index);
[seed, tangent] = init_H_LC(@MLfast, x(1:2, s(hopf).index), p, 1, 1e-4, 30, 4);
opt = contset;
opt = contset(opt, 'IgnoreSingularity', 1);
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'MaxNumPoints', 50);
opt = contset(opt, 'FunTolerance', 1e-7);
opt = contset(opt, 'VarTolerance', 1e-7);
[cycles, ~, cycle_events] = cont(@limitcycle, seed, tangent, opt);
labels = cellfun(@strtrim, {cycle_events.label}, 'UniformOutput', false);
lpc = find(strcmp(labels, 'LPC'), 1);
assert(~isempty(lpc), 'Morris-Lecar LPC seed not found.');
[seed, tangent] = init_LPC_LPC(@MLfast, cycles, cycle_events(lpc), [1, 2], 30, 4);
opt = contset(opt, 'MaxNumPoints', 20);
[curve, ~] = cont(@limitpointcycle, seed, tangent, opt);
assert(size(curve, 2) > 1);
fprintf('UNSUPPORTED_BY_JAXCONT two-parameter LPC curve: %d points\n', size(curve, 2));
end
