function run_heteroclinic_continuation()
%UNSUPPORTED_BY_JAXCONT Check the MatCont heteroclinic continuation workflow.
% Heteroclinic seeds require two problem-specific equilibria; this standalone
% wrapper verifies the installed continuation curve and names the required
% initializer without fabricating a model-specific seed.
setup_matcont();
assert(exist('heteroclinic', 'file') == 2, 'MatCont heteroclinic curve is unavailable.');
assert(exist('init_HTHet_Het', 'file') == 2, ...
    'MatCont heteroclinic homotopy initializer is unavailable.');
fprintf(['UNSUPPORTED_BY_JAXCONT heteroclinic continuation: use init_HTHet_Het ' ...
    'with problem-specific endpoint equilibria, then cont(@heteroclinic, ...).\n']);
end
