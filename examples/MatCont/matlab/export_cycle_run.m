function export_cycle_run(case_id, x, s, processor_data, output_dir, metadata)
%EXPORT_CYCLE_RUN Write normalized periodic branch/event/multiplier artifacts.

global lds
if ~isfolder(output_dir)
    mkdir(output_dir);
end
npoint = size(x, 2);
nphase = lds.nphase;
ncoords = size(x, 1) - 2;
parameter = x(end, :)';
period = x(end-1, :)';
residual = zeros(npoint, 1);
state_min = zeros(npoint, nphase);
state_max = zeros(npoint, nphase);
curve = feval(@limitcycle);
saved_mesh = lds.msh;
saved_dt = lds.dt;
for k = 1:npoint
    % The final BVP equation is MatCont's moving integral phase condition;
    % its reference orbit is updated during continuation and only represents
    % the latest point after ``cont`` returns.  The collocation defects and
    % continuity equations remain portable for every stored branch point.
    if size(processor_data, 1) >= lds.ntst + 1
        lds.msh = real(processor_data(1:lds.ntst+1, k))';
        lds.dt = lds.msh(lds.tsts+1) - lds.msh(lds.tsts);
    end
    value = feval(curve{1}, x(:, k));
    residual(k) = norm(value(1:end-1), inf);
    orbit = reshape(x(1:ncoords, k), nphase, []);
    state_min(k, :) = min(orbit, [], 2)';
    state_max(k, :) = max(orbit, [], 2)';
end
lds.msh = saved_mesh;
lds.dt = saved_dt;

point = (0:npoint-1)';
branch = table(repmat({case_id}, npoint, 1), point, parameter, period, residual, ...
    'VariableNames', {'case_id', 'point', 'parameter', 'period', 'residual_norm'});
for j = 1:nphase
    branch.(sprintf('state_%d_min', j - 1)) = state_min(:, j);
    branch.(sprintf('state_%d_max', j - 1)) = state_max(:, j);
end
writetable(branch, fullfile(output_dir, [case_id '_branch.csv']));

labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
keep = ~ismember(labels, {'00', '99'});
event_struct = s(keep);
labels = labels(keep);
nevent = numel(event_struct);
event_index = (0:nevent-1)';
event_type = labels(:);
event_point = zeros(nevent, 1);
event_parameter = zeros(nevent, 1);
event_period = zeros(nevent, 1);
multiplier_case = {};
multiplier_point = [];
multiplier_event = [];
multiplier_type = {};
multiplier_index = [];
multiplier_real = [];
multiplier_imag = [];
if size(processor_data, 1) >= lds.ntst + 1 + nphase
    for k = 1:npoint
        values = processor_data(lds.ntst+2:lds.ntst+1+nphase, k);
        for j = 1:numel(values)
            multiplier_case{end+1, 1} = case_id; %#ok<AGROW>
            multiplier_point(end+1, 1) = k - 1; %#ok<AGROW>
            multiplier_event(end+1, 1) = -1; %#ok<AGROW>
            multiplier_type{end+1, 1} = 'BRANCH'; %#ok<AGROW>
            multiplier_index(end+1, 1) = j - 1; %#ok<AGROW>
            multiplier_real(end+1, 1) = real(values(j)); %#ok<AGROW>
            multiplier_imag(end+1, 1) = imag(values(j)); %#ok<AGROW>
        end
    end
end
for k = 1:nevent
    idx = event_struct(k).index;
    event_point(k) = idx - 1;
    event_parameter(k) = parameter(idx);
    event_period(k) = period(idx);
    if isfield(event_struct(k), 'data') && ...
            isfield(event_struct(k).data, 'multipliers')
        values = event_struct(k).data.multipliers(:);
        for j = 1:numel(values)
            multiplier_case{end+1, 1} = case_id; %#ok<AGROW>
            multiplier_point(end+1, 1) = idx - 1; %#ok<AGROW>
            multiplier_event(end+1, 1) = k - 1; %#ok<AGROW>
            multiplier_type{end+1, 1} = labels{k}; %#ok<AGROW>
            multiplier_index(end+1, 1) = j - 1; %#ok<AGROW>
            multiplier_real(end+1, 1) = real(values(j)); %#ok<AGROW>
            multiplier_imag(end+1, 1) = imag(values(j)); %#ok<AGROW>
        end
    end
end
events = table(repmat({case_id}, nevent, 1), event_index, event_type, ...
    event_point, event_parameter, event_period, 'VariableNames', ...
    {'case_id', 'event_index', 'event_type', 'point', 'parameter', 'period'});
writetable(events, fullfile(output_dir, [case_id '_events.csv']));
multipliers = table(multiplier_case, multiplier_point, multiplier_event, multiplier_type, ...
    multiplier_index, multiplier_real, multiplier_imag, 'VariableNames', ...
    {'case_id', 'point', 'event_index', 'event_type', 'multiplier_index', 'real', 'imag'});
writetable(multipliers, fullfile(output_dir, [case_id '_multipliers.csv']));

metadata.case_id = case_id;
metadata.matlab_version = version;
metadata.matcont_version = '7.6';
metadata.mesh = struct('ntst', lds.ntst, 'ncol', lds.ncol);
metadata.generated_utc = datestr(datetime('now', 'TimeZone', 'UTC'), ...
    'yyyy-mm-ddTHH:MM:SSZ');
fid = fopen(fullfile(output_dir, [case_id '_metadata.json']), 'w');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(metadata));
end
