function export_equilibrium_run(case_id, x, s, rhs, output_dir, metadata)
%EXPORT_EQUILIBRIUM_RUN Write normalized equilibrium CSV/JSON artifacts.

if ~isfolder(output_dir)
    mkdir(output_dir);
end
nstate = size(x, 1) - 1;
npoint = size(x, 2);
parameter = x(end, :)';
states = x(1:nstate, :)';
residual = zeros(npoint, 1);
for k = 1:npoint
    residual(k) = norm(rhs(states(k, :)', parameter(k)), inf);
end

point = (0:npoint-1)';
branch = table(repmat({case_id}, npoint, 1), point, parameter, residual, ...
    'VariableNames', {'case_id', 'point', 'parameter', 'residual_norm'});
for j = 1:nstate
    branch.(sprintf('state_%d', j - 1)) = states(:, j);
end
writetable(branch, fullfile(output_dir, [case_id '_branch.csv']));

[event_struct, labels] = mathematical_events(s);
nevent = numel(event_struct);
event_index = (0:nevent-1)';
event_type = labels(:);
event_point = zeros(nevent, 1);
event_parameter = zeros(nevent, 1);
for k = 1:nevent
    event_point(k) = event_struct(k).index - 1;
    event_parameter(k) = x(end, event_struct(k).index);
end
events = table(repmat({case_id}, nevent, 1), event_index, event_type, ...
    event_point, event_parameter, 'VariableNames', ...
    {'case_id', 'event_index', 'event_type', 'point', 'parameter'});
writetable(events, fullfile(output_dir, [case_id '_events.csv']));

empty_multipliers = table(cell(0, 1), zeros(0, 1), cell(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), 'VariableNames', ...
    {'case_id', 'event_index', 'event_type', 'multiplier_index', 'real', 'imag'});
writetable(empty_multipliers, fullfile(output_dir, [case_id '_multipliers.csv']));
write_metadata(case_id, output_dir, metadata);
end

function [selected, labels] = mathematical_events(s)
labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
keep = ~ismember(labels, {'00', '99'});
selected = s(keep);
labels = labels(keep);
end

function write_metadata(case_id, output_dir, metadata)
metadata.case_id = case_id;
metadata.matlab_version = version;
metadata.matcont_version = '7.6';
metadata.generated_utc = datestr(datetime('now', 'TimeZone', 'UTC'), ...
    'yyyy-mm-ddTHH:MM:SSZ');
fid = fopen(fullfile(output_dir, [case_id '_metadata.json']), 'w');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(metadata));
end
