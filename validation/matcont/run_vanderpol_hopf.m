%RUN_VANDERPOL_HOPF Generate MatCont reference data for V01-EQ-003.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir, fullfile(script_dir, 'systems'));
setup_matcont();

[x0, v0] = init_EP_EP(@vanderpol_hopf, [0; 0], -2, 1);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'Eigenvalues', 1);
opt = contset(opt, 'MaxNumPoints', 300);
opt = contset(opt, 'InitStepsize', 0.02);
opt = contset(opt, 'MaxStepsize', 0.05);
[x, ~, s, ~, ~] = cont(@equilibrium, x0, v0, opt);

rhs = @(state, mu) [state(2); mu * (1 - state(1)^2) * state(2) - state(1)];
output_dir = fullfile(script_dir, '..', 'reference', 'generated');
export_equilibrium_run('V01-EQ-003', x, s, rhs, output_dir);

labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
hopf_parameters = x(end, [s(strcmp(labels, 'H')).index]);
assert(numel(hopf_parameters) == 1, 'Expected exactly one H event.');
assert(abs(hopf_parameters(1)) < 5e-4, ...
    'Hopf parameter does not match the analytic value mu=0.');
