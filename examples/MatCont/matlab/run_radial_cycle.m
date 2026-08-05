function run_radial_cycle(output_dir)
%RUN_RADIAL_CYCLE Produce MC-LC-001 from the analytic Hopf family.
script_dir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(script_dir, '..', 'generated');
end
addpath(script_dir, fullfile(script_dir, 'systems'));
setup_matcont();
clear global cds eds lds
rho = 1;
ntst = 25;
ncol = 4;
tau = linspace(0, 1, ntst * ncol + 1);
orbit = [cos(2*pi*tau); sin(2*pi*tau)];
cycle = [orbit(:); 2*pi; rho];
seed.index = 1;
seed.data.ntst = ntst;
seed.data.ncol = ncol;
seed.data.timemesh = linspace(0, 1, ntst + 1);
seed.data.T = 2*pi;
[x0, v0] = init_LC_LC(@radial_cycle_system, cycle, [], seed, rho, 1, ntst, ncol);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'Multipliers', 1);
opt = contset(opt, 'Adapt', 0);
opt = contset(opt, 'MaxNumPoints', 80);
opt = contset(opt, 'InitStepsize', 0.01);
opt = contset(opt, 'MaxStepsize', 0.03);
[x, ~, s, ~, processor_data] = cont(@limitcycle, x0, v0, opt);
metadata = struct('producer', mfilename, ...
    'source', 'analytic radial normal form', ...
    'precision', 'double', 'solver_settings', opt);
export_cycle_run('MC-LC-001', x, s, processor_data, output_dir, metadata);
end
