using BifurcationKit, LinearAlgebra
const BK = BifurcationKit

# Independent BifurcationKit.jl v0.5.2 reference values for JaxCont's codim-2
# solvers (jaxcont.bifurcations.codim2.bogdanov_takens_point). Run offline:
#   julia examples/BifurcationKit/05_codim2.jl
# Copy the printed values into tests/test_codim2.py.
#
# Route (a): a real applied model neither implementation was tuned for --
# the Lorenz-84 atmospheric circulation model, already used elsewhere in
# this repo (examples/BifurcationKit/02_lorenz84.jl). BifurcationKit.jl's
# OWN test suite (test/lorenz84.jl in the BifurcationKit.jl package tree)
# independently proves this exact model has a genuine Bogdanov-Takens point
# reachable by two-parameter continuation from F, then T -- this script
# reruns that continuation and Newton-refines the BT point to high
# precision so it can be cross-checked against jaxcont's direct solver.
#
# (A first attempt at route (a) used Bazykin's predator-prey model
# continued in (a, b) from (a,b,c) = (1.0, 0.5, 0.4); it only exposed a
# transcritical point, no fold/Hopf/BT/GH, in that parameter window --
# recorded here as a negative result, not repeated.)

function Lor(u, p, t = 0)
    (; α, β, γ, δ, G, F, T) = p
    X, Y, Z, U = u
    [
        -Y^2 - Z^2 - α*X + α*F - γ*U^2,
        X*Y - β*X*Z - Y + G,
        β*X*Y + X*Z - Z,
        -δ*U + γ*U*X + T,
    ]
end

parlor = (α = 1 // 4, β = 1, G = 0.25, δ = 1.04, γ = 0.987, F = 1.7620532879639, T = 0.0001265)

opts_br = ContinuationPar(p_min = -1.5, p_max = 3.0, ds = 0.001, dsmax = 0.025,
    detect_bifurcation = 3, n_inversion = 6, max_bisection_steps = 25,
    nev = 4, max_steps = 252)
@reset opts_br.newton_options.max_iterations = 25

z0 = [2.9787004394953343, -0.03868302503393752, 0.058232737694740085, -0.02105288273117459]

record_from_solution_lor(u::AbstractVector, p; k...) = (X = u[1], Y = u[2], Z = u[3], U = u[4])
record_from_solution_lor(u::BorderedArray, p; k...) = record_from_solution_lor(u.u, p)

prob = BK.BifurcationProblem(Lor, z0, parlor, (@optic _.F);
    record_from_solution = record_from_solution_lor)

# Continue the equilibrium branch in F (mirrors 02_lorenz84.jl).
br = continuation(re_make(prob, params = setproperties(parlor; T = 0.04, F = 3.0)),
    PALC(tangent = Bordered()), opts_br; normC = norminf, bothside = true)

# Follow the detected fold (index 5 on `br`) into the second parameter T
# with codim-2 detection switched on: the fold curve crosses two
# Bogdanov-Takens points (and two zero-Hopf points) as T varies.
sn_codim2 = continuation((@set br.alg.tangent = Secant()), 5, (@optic _.T),
    ContinuationPar(opts_br, p_max = 3.2, p_min = -0.1, detect_bifurcation = 1,
        dsmin = 1e-5, ds = -0.001, dsmax = 0.015, n_inversion = 10,
        save_sol_every_step = 1, max_steps = 30, max_bisection_steps = 55);
    verbosity = 0, normC = norminf, detect_codim2_bifurcation = 1,
    update_minaug_every_step = 1, start_with_eigen = true,
    record_from_solution = record_from_solution_lor, bdlinsolver = MatrixBLS())

println("=== fold branch (continued in T) special points ===")
for (i, sp) in enumerate(sn_codim2.specialpoint)
    println(i, "  ", sp.type)
end

# --- Fold-curve endpoints, for JaxCont's two-parameter continuation test ---
println("=== LP curve endpoints (seed values for tests/test_curves.py) ===")
println("first point x  = ", sn_codim2.sol[1].x)
println("first point p  = ", sn_codim2.sol[1].p)
println("last  point p  = ", sn_codim2.sol[end].p)
println("n curve points = ", length(sn_codim2.sol))

btpt = get_normal_form(sn_codim2, 1; nev = 4, verbose = false)
println("=== BT point (bisection-located, from continuation) ===")
println("x0 = ", btpt.x0)
println("F = ", btpt.params.F, "  T = ", btpt.params.T)
println("nf.a = ", btpt.nf.a, "  nf.b = ", btpt.nf.b)

# Newton-refine the BT point to high precision (normN = norminf is required
# here -- the raw `norm` default does not dispatch on BorderedArray in this
# BifurcationKit version).
solbt = newton(sn_codim2, 1; options = NewtonPar(br.contparams.newton_options;
        verbose = false, tol = 1e-13), start_with_eigen = true,
    jacobian_ma = BK.MinAug(), normN = norminf)

println("=== BT point (Newton-refined -- USE THESE VALUES) ===")
println("converged = ", BK.converged(solbt))
println("residual = ", solbt.residuals[end])
println("x0 = ", solbt.u.x0)
println("F = ", solbt.u.params.F, "  T = ", solbt.u.params.T)

J = BK.jacobian(prob, solbt.u.x0, solbt.u.params)
println("eigenvalues at refined BT point: ", eigvals(J))

println("=== fold branch (continued in T) special point details ===")
for (i, sp) in enumerate(sn_codim2.specialpoint)
    println(i, "  type=", sp.type, "  step=", sp.step, "  param=", sp.param)
end
