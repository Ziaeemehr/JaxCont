function root = setup_matcont()
%SETUP_MATCONT Add a configured CL_MatCont installation to the MATLAB path.

root = getenv('MATCONT_ROOT');
if isempty(root)
    root = '/home/ziaee/prog/MatCont/MatCont7p6';
end

if ~isfolder(root)
    error('JaxContValidation:MissingMatCont', ...
        'MatCont directory not found: %s. Set MATCONT_ROOT.', root);
end

addpath(genpath(root));
end

