function run_torbpc_cycle(output_dir)
%RUN_TORBPC_CYCLE Reproduce MatCont 7.6 Testruns/testtorBPC1.m.
script_dir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(script_dir, '..', 'generated');
end
addpath(script_dir, fullfile(script_dir, 'systems'));
setup_matcont();
clear global cds eds lds
p = [0.5; -0.6; 0.6; 0.32858; 0.93358; -0.9; 0.001];
state = [0.00125; -0.001; 0.00052502];
[x0, ~] = init_EP_EP(@torBPC, state, p, 6);
eqopt = contset;
eqopt = contset(eqopt, 'Singularities', 1);
eqopt = contset(eqopt, 'MaxNumPoints', 10);
[xeq, ~, seq, ~, ~] = cont(@equilibrium, x0, [], eqopt);
labels = cellfun(@strtrim, {seq.label}, 'UniformOutput', false);
hopf = seq(strcmp(labels, 'H'));
assert(numel(hopf) == 1, 'Expected one equilibrium Hopf seed.');
state = xeq(1:3, hopf.index);
p(6) = xeq(end, hopf.index);
[x0, v0] = init_H_LC(@torBPC, state, p, 6, 0.0001, 25, 4);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'Multipliers', 1);
opt = contset(opt, 'Adapt', 0);
opt = contset(opt, 'MaxNumPoints', 80);
[x, ~, s, ~, processor_data] = cont(@limitcycle, x0, v0, opt);
metadata = struct('producer', mfilename, ...
    'source', 'MatCont 7.6 Testruns/testtorBPC1.m', ...
    'precision', 'double', 'fixed_parameters', p([1:5, 7])', ...
    'solver_settings', opt);
export_cycle_run('MC-LC-002', x, s, processor_data, output_dir, metadata);
end
