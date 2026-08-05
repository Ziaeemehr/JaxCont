function run_adaptive_control(output_dir)
%RUN_ADAPTIVE_CONTROL Produce MC-EQ-003 from MatCont's adaptx model.
script_dir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(script_dir, '..', 'generated');
end
addpath(script_dir, fullfile(script_dir, 'systems'));
setup_matcont();
clear global cds eds lds
[x0, v0] = init_EP_EP(@adaptive_control, [0; 0; 0], [-2; 1], 1);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'Eigenvalues', 1);
opt = contset(opt, 'MaxNumPoints', 300);
opt = contset(opt, 'InitStepsize', 0.02);
opt = contset(opt, 'MaxStepsize', 0.05);
[x, ~, s, ~, processor_data] = cont(@equilibrium, x0, v0, opt);
rhs = @(state, alpha) [state(2); state(3); ...
    -alpha * state(3) - state(2) - state(1) + state(1)^2];
metadata = struct('producer', mfilename, ...
    'source', 'MatCont 7.6 Testruns/testadapt.m', ...
    'precision', 'double', 'solver_settings', opt);
export_equilibrium_run('MC-EQ-003', x, s, processor_data, rhs, output_dir, metadata);
end
