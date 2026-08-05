function run_general_bvp_continuation()
%UNSUPPORTED_BY_JAXCONT Exercise MatCont's collocation BVP continuation path.
% This radial cycle is a compact standalone seed for adapting the wrapper to
% a user-supplied general boundary-value problem.
setup_matcont();
script_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(script_dir, 'systems'));
clear global cds eds lds
rho = 1;
ntst = 10;
ncol = 4;
tau = linspace(0, 1, ntst * ncol + 1);
orbit = [cos(2*pi*tau); sin(2*pi*tau)];
cycle = [orbit(:); 2*pi; rho];
seed_data.index = 1;
seed_data.data.ntst = ntst;
seed_data.data.ncol = ncol;
seed_data.data.timemesh = linspace(0, 1, ntst + 1);
seed_data.data.T = 2*pi;
[seed, tangent] = init_LC_LC(@radial_cycle_system, cycle, [], seed_data, rho, 1, ntst, ncol);
opt = contset;
opt = contset(opt, 'MaxNumPoints', 10);
opt = contset(opt, 'Adapt', 0);
[curve, ~] = cont(@limitcycle, seed, tangent, opt);
assert(size(curve, 2) > 1);
fprintf('UNSUPPORTED_BY_JAXCONT general BVP template: %d collocation points\n', ...
    size(curve, 2));
end
