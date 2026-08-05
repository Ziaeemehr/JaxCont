function run_heteroclinic_continuation(seed)
%UNSUPPORTED_BY_JAXCONT NON_EXECUTABLE_TEMPLATE for heteroclinic continuation.
%
% A genuine MatCont heteroclinic seed is model-specific and needs two endpoint
% equilibria, their stable/unstable eigenspaces, truncation times, mesh, active
% parameters, and homotopy stages. The concrete MatCont workflow is:
%   1. assemble endpoints/eigenspaces for the chosen system;
%   2. initialize homotopy with init_HTHet_HTHet;
%   3. continue the homotopy stages with cont(@homotopyhet, ...);
%   4. convert using init_HTHet_Het;
%   5. continue with cont(@heteroclinic, ...).
%
% This file checks that the installed MatCont has those entry points, but it
% refuses to report execution without a problem-specific seed structure.
setup_matcont();
assert(exist('heteroclinic', 'file') == 2, 'MatCont heteroclinic curve is unavailable.');
assert(exist('init_HTHet_Het', 'file') == 2, ...
    'MatCont heteroclinic homotopy initializer is unavailable.');
if nargin == 1
    assert(isstruct(seed), 'Heteroclinic seed must be a problem-specific structure.');
end
error('JaxContValidation:NonExecutableTemplate', ...
    ['NON_EXECUTABLE_TEMPLATE: endpoint equilibria and homotopy seed are ' ...
     'problem-specific and are not fabricated by this suite.']);
end
