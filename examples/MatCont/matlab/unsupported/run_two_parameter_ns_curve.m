function run_two_parameter_ns_curve()
%UNSUPPORTED_BY_JAXCONT Continue the MatCont torBPC1 NS curve.
setup_matcont();
p = [0.5; -0.6; 0.6; 0.32858; 0.93358; -0.9; 0.001];
[x0, ~] = init_EP_EP(@torBPC, [0.00125; -0.001; 0.00052502], p, 6);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'MaxNumPoints', 10);
[x, ~, s] = cont(@equilibrium, x0, [], opt);
labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
hopf = find(strcmp(labels, 'H'), 1);
assert(~isempty(hopf), 'torBPC Hopf seed not found.');
p(6) = x(end, s(hopf).index);
[seed, tangent] = init_H_LC(@torBPC, x(1:3, s(hopf).index), p, 6, 1e-4, 25, 4);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'Multipliers', 1);
opt = contset(opt, 'MaxNumPoints', 50);
[cycles, ~, cycle_events] = cont(@limitcycle, seed, tangent, opt);
labels = cellfun(@strtrim, {cycle_events.label}, 'UniformOutput', false);
ns = find(strcmp(labels, 'NS'), 1);
assert(~isempty(ns), 'torBPC NS seed not found.');
[seed, tangent] = init_NS_NS(@torBPC, cycles, cycle_events(ns), [6, 7], 25, 4);
opt = contset;
opt = contset(opt, 'VarTolerance', 1e-4);
opt = contset(opt, 'FunTolerance', 1e-4);
opt = contset(opt, 'Backward', 1);
opt = contset(opt, 'MaxNumPoints', 16);
[curve, ~] = cont(@neimarksacker, seed, tangent, opt);
assert(size(curve, 2) > 1);
fprintf('UNSUPPORTED_BY_JAXCONT two-parameter NS curve: %d points\n', size(curve, 2));
end
