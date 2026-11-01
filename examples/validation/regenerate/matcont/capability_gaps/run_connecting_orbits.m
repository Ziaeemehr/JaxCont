function run_connecting_orbits(kind)
%UNSUPPORTED_BY_JAXCONT Run MatCont's connecting-orbit continuation example.
setup_matcont();
if nargin < 1
    kind = 'homoclinic';
end
if strcmpi(kind, 'heteroclinic')
    assert(exist('heteroclinic', 'file') == 2, 'MatCont heteroclinic curve is unavailable.');
    fprintf(['UNSUPPORTED_BY_JAXCONT heteroclinic continuation is installed; ' ...
        'supply problem-specific endpoint equilibria to init_HTHet_Het.\n']);
    return
end
p = [0.11047; 0.1];
[x0, ~] = init_EP_EP(@MLfast, [0.047222; 0.32564], p, 1);
opt = contset;
opt = contset(opt, 'Singularities', 1);
opt = contset(opt, 'MaxNumPoints', 65);
opt = contset(opt, 'MinStepsize', 1e-5);
opt = contset(opt, 'MaxStepsize', 0.01);
opt = contset(opt, 'Backward', 1);
[x, ~, s] = cont(@equilibrium, x0, [], opt);
labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
hopf = find(strcmp(labels, 'H'), 1);
p(1) = x(end, s(hopf).index);
[seed, tangent] = init_H_LC(@MLfast, x(1:2, s(hopf).index), p, 1, 1e-4, 30, 4);
opt = contset(opt, 'IgnoreSingularity', 1);
opt = contset(opt, 'MaxNumPoints', 200);
opt = contset(opt, 'MaxStepsize', 1);
[cycles, ~, cycle_events] = cont(@limitcycle, seed, tangent, opt);
p(1) = cycles(end, end);
half_period = cycles(end-1, end) / 2;
[seed, tangent] = init_LC_Hom(@MLfast, cycles(:, end), cycle_events(end), p, ...
    [1, 2], 40, 4, [0, 1, 1], half_period, 0.01, 0.01);
opt = contset(opt, 'MaxNumPoints', 15);
[curve, ~] = cont(@homoclinic, seed, tangent, opt);
assert(size(curve, 2) > 1);
fprintf('UNSUPPORTED_BY_JAXCONT homoclinic continuation: %d points\n', size(curve, 2));
end
