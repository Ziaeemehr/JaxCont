function run_two_parameter_pd_curve()
%UNSUPPORTED_BY_JAXCONT Continue the MatCont adaptive-control PD curve.
setup_matcont();
[x0, ~] = init_EP_EP(@adaptx, [0; 0; 0], [-10; 1], 1);
opt = contset;
opt = contset(opt, 'Singularities', 1);
[x, ~, s] = cont(@equilibrium, x0, [], opt);
labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
hopf = find(strcmp(labels, 'H'), 1);
assert(~isempty(hopf), 'Adaptive-control Hopf seed not found.');
p = [x(end, s(hopf).index); 1];
[seed, tangent] = init_H_LC(@adaptx, x(1:3, s(hopf).index), p, 1, 1e-6, 20, 4);
opt = contset(opt, 'MaxNumPoints', 200);
opt = contset(opt, 'Multipliers', 1);
opt = contset(opt, 'Adapt', 1);
[cycles, ~, cycle_events] = cont(@limitcycle, seed, tangent, opt);
labels = cellfun(@strtrim, {cycle_events.label}, 'UniformOutput', false);
pd = find(strcmp(labels, 'PD'), 1);
assert(~isempty(pd), 'Adaptive-control PD seed not found.');
[seed, tangent] = init_PD_PD(@adaptx, cycles, cycle_events(pd), [1, 2], 20, 4);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'MaxNumPoints', 25);
[curve, ~] = cont(@perioddoubling, seed, tangent, opt);
assert(size(curve, 2) > 1);
fprintf('UNSUPPORTED_BY_JAXCONT two-parameter PD curve: %d points\n', size(curve, 2));
end
