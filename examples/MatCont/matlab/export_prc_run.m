function export_prc_run(case_id, x2, processor_data, output_dir, metadata)
%EXPORT_PRC_RUN Write normalized PRC/dPRC artifacts from one continuation
%point's processor_data column (MatCont's PRC/dPRC/Input option layout:
%rows 1:(ntst+1) = mesh, next 81 rows = PRC, final 81 rows = dPRC -- see
%Testruns/testadaptPRC.m). ``x2`` is the matching state/parameter output
%of the same ``cont`` call (rows: [cycle coordinates; period; parameter],
%same convention export_cycle_run.m uses): its last column's final two
%rows give the period/alpha the exported processor_data column (also its
%last column) actually converged to.
global lds
if ~isfolder(output_dir)
    mkdir(output_dir);
end
metadata.converged_alpha = x2(end, end);
metadata.converged_period = x2(end-1, end);
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
% Do not label phase as a naive uniform point/n_points -- the 'Adapt', 1
% option (set for this run) redistributes lds.msh non-uniformly (measured
% spacings ranged 0.043-0.059 for MC-PRC-001, not the uniform 1/ntst =
% 0.05 a linear point/n_points label would imply), and PRC/dPRC's ncol
% sub-points per mesh interval are genuinely evaluated at the resulting
% non-uniform locations, not at uniform fractions. mesh_points (extracted
% above from processor_data's own mesh block, i.e. MatCont's lds.msh for
% this exact point) is normalized-time in [0, 1], so reconstruct the same
% ntst*ncol+1 "finemsh" MatCont itself builds internally (see
% LimitCycle/adapt_mesh.m's lds.finemsh assignment) by uniformly
% subdividing each of the ntst (possibly unequal-width) mesh intervals
% into ncol pieces, rather than assuming the whole [0, 1] range is
% uniformly subdivided across all ntst*ncol intervals at once.
ncol = lds.ncol;
lower = mesh_points(1:end-1);
upper = mesh_points(2:end);
fractions = (1:ncol) / ncol;
subdivided = lower(:) + (upper(:) - lower(:)) * fractions;  % ntst x ncol
phase_fraction = [0; reshape(subdivided', [], 1)];  % interval-major, ntst*ncol+1 rows
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
