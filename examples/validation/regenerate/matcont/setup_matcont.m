function root = setup_matcont(root)
%SETUP_MATCONT Add a CL_MatCont 7.6 installation to the MATLAB path.

if nargin < 1 || isempty(root)
    root = getenv('MATCONT_ROOT');
end
if isempty(root)
    error('JaxContValidation:MissingMatCont', 'set MATCONT_ROOT');
end
if ~isfolder(root)
    error('JaxContValidation:MissingMatCont', ...
        'MatCont directory not found: %s. Set MATCONT_ROOT.', root);
end

addpath(genpath(root));
end
