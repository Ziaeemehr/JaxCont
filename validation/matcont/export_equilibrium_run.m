function export_equilibrium_run(case_id, x, s, rhs, output_dir)
%EXPORT_EQUILIBRIUM_RUN Write transparent branch/event CSV files.

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
writetable(branch, fullfile(output_dir, [case_id '_matcont_branch.csv']));

labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
% MatCont stores the initial/final continuation markers as 00/99 in the same
% structure as mathematical special points. They are branch bookkeeping, not
% bifurcation events, so keep them out of the portable event artifact.
s = s(~ismember(labels, {'00', '99'}));
nevent = numel(s);
event_index = (0:nevent-1)';
event_type = cell(nevent, 1);
event_parameter = zeros(nevent, 1);
event_point = zeros(nevent, 1);
for k = 1:nevent
    event_type{k} = strtrim(s(k).label);
    event_point(k) = s(k).index - 1;
    event_parameter(k) = x(end, s(k).index);
end
events = table(repmat({case_id}, nevent, 1), event_index, event_type, ...
    event_point, event_parameter, 'VariableNames', ...
    {'case_id', 'event_index', 'event_type', 'point', 'parameter'});
writetable(events, fullfile(output_dir, [case_id '_matcont_events.csv']));
end
