using BifurcationKit, Plots
F(x, p) = @. p[1] + x - x^3 / 3
prob = BifurcationProblem(F, [-2.], [-1.], 1;
    record_from_solution=(x, p; k...) -> x[1])
br = continuation(prob, PALC(), ContinuationPar(p_min=-1., p_max=1.))
plot(br)