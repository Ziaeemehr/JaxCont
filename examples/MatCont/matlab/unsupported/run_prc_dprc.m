function run_prc_dprc()
%UNSUPPORTED_BY_JAXCONT Compute MatCont's adaptive-control PRC and dPRC.
setup_matcont();
[x0, ~] = init_EP_EP(@adaptx, [0; 0; 0], [-10; 1], 1);
opt = contset;
opt = contset(opt, 'Singularities', 1);
[x, ~, s] = cont(@equilibrium, x0, [], opt);
labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
hopf = find(strcmp(labels, 'H'), 1);
p = [x(end, s(hopf).index); 1];
[seed, tangent] = init_H_LC(@adaptx, x(1:3, s(hopf).index), p, 1, 1e-6, 20, 4);
opt = contset(opt, 'MaxNumPoints', 10);
opt = contset(opt, 'Multipliers', 1);
opt = contset(opt, 'Adapt', 1);
[cycles, tangents, cycle_events] = cont(@limitcycle, seed, tangent, opt);
parameters = cycle_events(end).data.parametervalues;
[seed, tangent] = init_LC_LC(@adaptx, cycles, tangents, cycle_events(end), ...
    parameters, 1, 20, 4);
opt = contset(opt, 'PRC', 1);
opt = contset(opt, 'dPRC', 1);
opt = contset(opt, 'Input', 1);
opt = contset(opt, 'MaxNumPoints', 20);
[~, ~, ~, ~, processor_data] = cont(@limitcycle, seed, tangent, opt);
assert(~isempty(processor_data) && all(isfinite(processor_data(:, end))));
fprintf('UNSUPPORTED_BY_JAXCONT PRC/dPRC: %d processor values\n', ...
    size(processor_data, 1));
end
