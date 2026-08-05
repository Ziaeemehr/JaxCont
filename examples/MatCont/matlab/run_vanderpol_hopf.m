function run_vanderpol_hopf(output_dir)
%RUN_VANDERPOL_HOPF Produce MC-EQ-002 with CL_MatCont.
script_dir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(script_dir, '..', 'generated');
end
addpath(script_dir, fullfile(script_dir, 'systems'));
setup_matcont();
clear global cds eds lds
[x0, v0] = init_EP_EP(@vanderpol_hopf, [0; 0], -2, 1);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'Eigenvalues', 1);
opt = contset(opt, 'MaxNumPoints', 300);
opt = contset(opt, 'InitStepsize', 0.02);
opt = contset(opt, 'MaxStepsize', 0.05);
[x, ~, s, ~, processor_data] = cont(@equilibrium, x0, v0, opt);
rhs = @(state, mu) [state(2); mu * (1 - state(1)^2) * state(2) - state(1)];
metadata = struct('producer', mfilename, 'source', 'analytic Van der Pol Hopf', ...
    'precision', 'double', 'solver_settings', opt);
export_equilibrium_run('MC-EQ-002', x, s, processor_data, rhs, output_dir, metadata);
end
