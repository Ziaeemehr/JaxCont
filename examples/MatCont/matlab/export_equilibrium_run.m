function export_equilibrium_run(case_id, x, s, processor_data, rhs, output_dir, metadata)
%EXPORT_EQUILIBRIUM_RUN Write normalized equilibrium CSV/JSON artifacts.

if ~isfolder(output_dir)
    mkdir(output_dir);
end
nstate = size(x, 1) - 1;
npoint = size(x, 2);
parameter = x(end, :)';
states = x(1:nstate, :)';
residual = zeros(npoint, 1);
unstable_dimension = zeros(npoint, 1);
stable = false(npoint, 1);
for k = 1:npoint
    residual(k) = norm(rhs(states(k, :)', parameter(k)), inf);
    eigenvalues = processor_data(:, k);
    unstable_dimension(k) = sum(real(eigenvalues) > 0);
    stable(k) = unstable_dimension(k) == 0;
end

point = (0:npoint-1)';
branch = table(repmat({case_id}, npoint, 1), point, parameter, residual, ...
    stable, unstable_dimension, 'VariableNames', ...
    {'case_id', 'point', 'parameter', 'residual_norm', 'stable', 'unstable_dimension'});
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
frequency = nan(nevent, 1);
fold_coefficient = nan(nevent, 1);
first_lyapunov = nan(nevent, 1);
for k = 1:nevent
    event_point(k) = event_struct(k).index - 1;
    event_parameter(k) = x(end, event_struct(k).index);
    if isfield(event_struct(k), 'data') && isfield(event_struct(k).data, 'eval')
        event_eigenvalues = event_struct(k).data.eval(:);
        positive_frequencies = imag(event_eigenvalues(imag(event_eigenvalues) > 0));
        if ~isempty(positive_frequencies)
            frequency(k) = max(positive_frequencies);
        end
    end
    if strcmp(labels{k}, 'LP') && isfield(event_struct(k).data, 'a')
        fold_coefficient(k) = event_struct(k).data.a;
    end
    if strcmp(labels{k}, 'H') && isfield(event_struct(k).data, 'lyapunov') && ...
            isnumeric(event_struct(k).data.lyapunov)
        first_lyapunov(k) = event_struct(k).data.lyapunov;
    end
end
events = table(repmat({case_id}, nevent, 1), event_index, event_type, ...
    event_point, event_parameter, frequency, fold_coefficient, first_lyapunov, ...
    'VariableNames', {'case_id', 'event_index', 'event_type', 'point', ...
    'parameter', 'frequency', 'fold_coefficient', 'first_lyapunov'});
writetable(events, fullfile(output_dir, [case_id '_events.csv']));

spectrum_case = {};
spectrum_point = [];
spectrum_event = [];
spectrum_type = {};
spectrum_kind = {};
spectrum_index = [];
spectrum_real = [];
spectrum_imag = [];
for k = 1:npoint
    values = processor_data(:, k);
    for j = 1:numel(values)
        spectrum_case{end+1, 1} = case_id; %#ok<AGROW>
        spectrum_point(end+1, 1) = k - 1; %#ok<AGROW>
        spectrum_event(end+1, 1) = -1; %#ok<AGROW>
        spectrum_type{end+1, 1} = 'BRANCH'; %#ok<AGROW>
        spectrum_kind{end+1, 1} = 'EIGENVALUE'; %#ok<AGROW>
        spectrum_index(end+1, 1) = j - 1; %#ok<AGROW>
        spectrum_real(end+1, 1) = real(values(j)); %#ok<AGROW>
        spectrum_imag(end+1, 1) = imag(values(j)); %#ok<AGROW>
    end
end
for k = 1:nevent
    if isfield(event_struct(k), 'data') && isfield(event_struct(k).data, 'eval')
        values = event_struct(k).data.eval(:);
        for j = 1:numel(values)
            spectrum_case{end+1, 1} = case_id; %#ok<AGROW>
            spectrum_point(end+1, 1) = event_point(k); %#ok<AGROW>
            spectrum_event(end+1, 1) = k - 1; %#ok<AGROW>
            spectrum_type{end+1, 1} = labels{k}; %#ok<AGROW>
            spectrum_kind{end+1, 1} = 'EIGENVALUE'; %#ok<AGROW>
            spectrum_index(end+1, 1) = j - 1; %#ok<AGROW>
            spectrum_real(end+1, 1) = real(values(j)); %#ok<AGROW>
            spectrum_imag(end+1, 1) = imag(values(j)); %#ok<AGROW>
        end
    end
end
spectra = table(spectrum_case, spectrum_point, spectrum_event, spectrum_type, ...
    spectrum_kind, spectrum_index, spectrum_real, spectrum_imag, ...
    'VariableNames', {'case_id', 'point', 'event_index', 'event_type', ...
    'spectrum_kind', 'multiplier_index', 'real', 'imag'});
writetable(spectra, fullfile(output_dir, [case_id '_multipliers.csv']));
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
