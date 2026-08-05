function run_general_bvp_continuation(problem)
%UNSUPPORTED_BY_JAXCONT NON_EXECUTABLE_TEMPLATE for a general BVP.
%
% MatCont's standard limit-cycle curve is a specialized collocation BVP, but
% it is not a standalone general-BVP API. A genuine general BVP needs a user
% problem that supplies, at minimum:
%   problem.curve       - MatCont curve function handle
%   problem.x0          - assembled initial BVP unknown
%   problem.v0          - initial tangent (or [])
%   problem.options     - contset options
%
% With those problem-specific pieces the execution step is:
%   [x,v,s,h,f] = cont(problem.curve, problem.x0, problem.v0, problem.options);
%
% This template deliberately refuses to claim a successful validation without
% such a problem. The radial periodic producer is available separately as
% run_radial_cycle and must not be mislabeled as a general BVP example.
setup_matcont();
if nargin == 1
    required = {'curve', 'x0', 'v0', 'options'};
    assert(all(isfield(problem, required)), ...
        'General BVP problem must define curve, x0, v0, and options.');
end
error('JaxContValidation:NonExecutableTemplate', ...
    ['NON_EXECUTABLE_TEMPLATE: supply a documented problem-specific BVP ' ...
     'before calling cont; no standalone MatCont general-BVP seed is bundled.']);
end
