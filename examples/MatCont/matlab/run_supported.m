function run_supported(output_dir)
%RUN_SUPPORTED Regenerate every supported MatCont producer.

script_dir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(script_dir, '..', 'generated');
end
addpath(script_dir, fullfile(script_dir, 'systems'));
setup_matcont();
run_cubic_fold(output_dir);
run_vanderpol_hopf(output_dir);
run_adaptive_control(output_dir);
run_radial_cycle(output_dir);
run_torbpc_cycle(output_dir);
end
