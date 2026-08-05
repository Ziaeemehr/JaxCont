function export_prc_run(case_id, processor_data, output_dir, metadata)
%EXPORT_PRC_RUN Write normalized PRC/dPRC artifacts from one continuation
%point's processor_data column (MatCont's PRC/dPRC/Input option layout:
%rows 1:(ntst+1) = mesh, next 81 rows = PRC, final 81 rows = dPRC -- see
%Testruns/testadaptPRC.m).
global lds
if ~isfolder(output_dir)
    mkdir(output_dir);
end
ntst = lds.ntst;
mesh_rows = ntst + 1;
% PRC/dPRC each occupy ntst*ncol+1 rows (see Testruns/testadaptPRC.m's
% fvector(22:102)/fvector(103:183) slicing for ntst=20, ncol=4). Trailing
% rows beyond mesh+PRC+dPRC (e.g. Floquet multipliers, appended when the
% 'Multipliers' option is left on from an earlier continuation) are
% intentionally ignored rather than folded into this split.
half = ntst * lds.ncol + 1;
assert(size(processor_data, 1) >= mesh_rows + 2 * half, ...
    'processor_data does not contain the expected mesh + PRC + dPRC rows');

fvector = processor_data(:, end);
mesh_points = real(fvector(1:mesh_rows));
prc_values = real(fvector(mesh_rows + 1 : mesh_rows + half));
dprc_values = real(fvector(mesh_rows + half + 1 : mesh_rows + 2 * half));

n_points = numel(prc_values);
point = (0:n_points-1)';
phase_fraction = point / n_points;
prc_table = table(repmat({case_id}, n_points, 1), point, phase_fraction, prc_values, dprc_values, ...
    'VariableNames', {'case_id', 'point', 'phase_fraction', 'prc', 'dprc'});
writetable(prc_table, fullfile(output_dir, [case_id '_prc.csv']));

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
