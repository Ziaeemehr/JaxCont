function run_cubic_fold(output_dir)
%RUN_CUBIC_FOLD Produce MC-EQ-001 with CL_MatCont.
script_dir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(script_dir, '..', 'generated');
end
addpath(script_dir, fullfile(script_dir, 'systems'));
setup_matcont();
clear global cds eds lds
[x0, v0] = init_EP_EP(@cubic_fold, -2.1038034027355366, -1, 1);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'Eigenvalues', 1);
opt = contset(opt, 'MaxNumPoints', 500);
opt = contset(opt, 'InitStepsize', 0.01);
opt = contset(opt, 'MaxStepsize', 0.02);
[x, ~, s, ~, ~] = cont(@equilibrium, x0, v0, opt);
rhs = @(state, r) r + state(1) - state(1)^3 / 3;
metadata = struct('producer', mfilename, 'source', 'analytic cubic fold', ...
    'precision', 'double', 'solver_settings', opt);
export_equilibrium_run('MC-EQ-001', x, s, rhs, output_dir, metadata);
end
