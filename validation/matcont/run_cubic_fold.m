%RUN_CUBIC_FOLD Generate MatCont reference data for V01-EQ-002.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir, fullfile(script_dir, 'systems'));
setup_matcont();

[x0, v0] = init_EP_EP(@cubic_fold, -2, -1, 1);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'Eigenvalues', 1);
opt = contset(opt, 'MaxNumPoints', 500);
opt = contset(opt, 'InitStepsize', 0.01);
opt = contset(opt, 'MaxStepsize', 0.02);
[x, ~, s, ~, ~] = cont(@equilibrium, x0, v0, opt);

rhs = @(state, r) r + state(1) - state(1)^3 / 3;
output_dir = fullfile(script_dir, '..', 'reference', 'generated');
export_equilibrium_run('V01-EQ-002', x, s, rhs, output_dir);

labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
fold_parameters = x(end, [s(strcmp(labels, 'LP')).index]);
assert(numel(fold_parameters) == 2, 'Expected exactly two LP events.');
assert(max(abs(sort(fold_parameters) - [-2/3, 2/3])) < 5e-4, ...
    'Fold parameters do not match the analytic values.');
