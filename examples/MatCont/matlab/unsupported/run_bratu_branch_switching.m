function run_bratu_branch_switching()
%UNSUPPORTED_BY_JAXCONT Continue both branches from MatCont's Bratu BP.
setup_matcont();
clear global cds eds lds
global cds
p = 0;
[x0, ~] = init_EP_EP(@bratu, [0; 0], p, 1);
opt = contset;
opt = contset(opt, 'MaxNumPoints', 50);
opt = contset(opt, 'Singularities', 1);
[x, v, s, h, f] = cont(@equilibrium, x0, [], opt);
[x, v, s, h, f] = cont(x, v, s, h, f, cds);
labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
bp = find(strcmp(labels, 'BP'), 1);
assert(~isempty(bp), 'MatCont Bratu producer did not locate a BP.');
p = x(end, s(bp).index);
[seed, tangent] = init_BP_EP(@bratu, x(1:2, s(bp).index), p, s(bp), 0.01);
opt = contset(opt, 'MaxNumPoints', 25);
[secondary_forward, ~] = cont(@equilibrium, seed, tangent, opt);
opt = contset(opt, 'Backward', 1);
[secondary_backward, ~] = cont(@equilibrium, seed, tangent, opt);
assert(size(secondary_forward, 2) > 1 && size(secondary_backward, 2) > 1);
fprintf('UNSUPPORTED_BY_JAXCONT Bratu branch switching: %d + %d points\n', ...
    size(secondary_forward, 2), size(secondary_backward, 2));
end
