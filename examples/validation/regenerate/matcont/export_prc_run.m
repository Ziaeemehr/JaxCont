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

% calcPRC.m circularly rotates its PRC/dPRC VALUE output so the exported
% curve starts at the trajectory's own x-maximum ("make the PRC and
% collocation mesh start at the spike-top"; see LimitCycle/calcPRC.m's
% [val,ind] = max(tmpcycle(1,:)) block and the PRC(end+1:end+ind-1) =
% PRC(1:ind-1); PRC = PRC(ind:end) rotation applied to PRC, dPRC, and its
% own inpmesh). prc_values/dprc_values above are exactly that already-
% rotated calcPRC.m output (they flow through lds.PRCdata/lds.dPRCdata,
% see limitcycle.m). phase_fraction, in contrast, was just built from
% mesh_points = lds.msh, which calcPRC.m never rotates -- it is already
% the correct, natural (unrotated) 0..1 labeling, with the periodic
% closure (phase 0 == phase 1, same physical point) sitting cleanly at
% the two endpoints.
%
% The bug is pairing this natural, unrotated phase_fraction with the
% already-rotated prc_values/dprc_values. Rotating phase_fraction to
% match would only relabel the rotation's own internal wrap-around seam
% (where the periodic-closure duplicate lands adjacent, INTERIOR to the
% array, at position half-rotation_index+1/+2) -- it would not move that
% duplicate back to the endpoints, since a circular rotation's own
% closure seam is invariant to how the array is subsequently labeled.
% The only way to make the exported (phase, value) pairs show the
% periodic closure at the endpoints (phase 0 <-> phase 1, matching
% values) is to undo calcPRC.m's rotation on the VALUES, recovering
% their natural pre-rotation order, and pair that with phase_fraction
% as already (correctly) built above.
%
% Derive the same rotation index calcPRC.m computes (from the
% trajectory's own x-maximum, using x2's cycle coordinates -- the same
% values calcPRC.m was called with), then invert calcPRC.m's circular
% shift: if rotated(i) = natural(mod(rotation_index+i-2, half)+1), then
% natural(j) = rotated(mod(j-rotation_index, half)+1).
nphase = lds.nphase;
cycle_coords = real(x2(1:nphase * half, end));
tmpcycle = reshape(cycle_coords, nphase, half);
[~, rotation_index] = max(tmpcycle(1, :));
unrotate_index = mod((1:half)' - rotation_index, half) + 1;
prc_values = prc_values(unrotate_index);
dprc_values = dprc_values(unrotate_index);

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
