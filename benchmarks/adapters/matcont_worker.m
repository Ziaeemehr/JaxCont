function matcont_worker(mode, repetitions, warmups, controls_json)
%MATCONT_WORKER Benchmark worker for cubic.eq-fold (READY, then JSON lines).
%   Startup (MATLAB launch, MatCont path setup) ends at the READY line and is
%   timed by the Python supervisor. Each measurement prints one JSON object
%   with the wall time of cont() and the raw results the Python side needs to
%   compute accuracy; MatCont never reports its own accuracy.
here = fileparts(mfilename('fullpath'));
validation = fullfile(here, '..', '..', 'examples', 'validation', 'regenerate', 'matcont');
addpath(validation, fullfile(validation, 'systems'));
setup_matcont();
c = jsondecode(controls_json);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'Eigenvalues', double(c.stability_every_step));
opt = contset(opt, 'MaxNumPoints', c.max_steps);
opt = contset(opt, 'InitStepsize', c.ds);
opt = contset(opt, 'MinStepsize', c.ds_min);
opt = contset(opt, 'MaxStepsize', c.ds_max);
opt = contset(opt, 'FunTolerance', c.newton_atol);
opt = contset(opt, 'VarTolerance', c.newton_atol);
opt = contset(opt, 'MaxCorrIters', c.newton_max_iter);
opt = contset(opt, 'TestTolerance', c.event_atol);
opt = contset(opt, 'Backward', c.direction < 0);
fprintf('READY\n');
if strcmp(mode, 'first_call')
    emit('first_call', opt, true);
    return
end
for k = 1:warmups
    run_once(opt);
end
for k = 1:repetitions
    emit('warm', opt, false);
end
end

function emit(stage, opt, with_results)
[seconds, x, s] = run_once(opt);
row = struct('stage', stage, 'seconds', seconds, 'points', size(x, 2));
if with_results
    folds = [];
    fold_states = [];
    for k = 1:numel(s)
        if strcmp(strtrim(s(k).label), 'LP')
            folds(end + 1) = x(end, s(k).index); %#ok<AGROW>
            fold_states(end + 1) = x(1, s(k).index); %#ok<AGROW>
        end
    end
    row.folds = folds;
    row.fold_states = fold_states;
    [~, matcont] = fileparts(getenv('MATCONT_ROOT'));  % folder name only, no local path
    row.versions = struct('matlab', version(), 'matcont', matcont);
    row.states = x(1, :);
    row.params = x(end, :);
end
fprintf('%s\n', jsonencode(row));
end

function [seconds, x, s] = run_once(opt)
clear global cds eds lds
[x0, v0] = init_EP_EP(@cubic_fold, -2.1038034027355366, -1, 1);
started = tic;
[x, ~, s] = cont(@equilibrium, x0, v0, opt);
seconds = toc(started);
end
