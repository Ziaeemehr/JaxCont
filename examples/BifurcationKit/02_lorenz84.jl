using Revise, Plots, LinearAlgebra
using BifurcationKit
const BK = BifurcationKit

# vector field
function Lor!(out, u, p, t = 0)
	(;α,β,γ,δ,G,F,T) = p
	X,Y,Z,U = u
	out[1] = -Y^2 - Z^2 - α*X + α*F - γ*U^2
	out[2] = X*Y - β*X*Z - Y + G
	out[3] = β*X*Y + X*Z - Z
	out[4] = -δ*U + γ*U*X + T
	out
end

parlor = (α = 1//4, β = 1., G = .25, δ = 1.04, γ = 0.987, F = 1.7620532879639, T = .0001265)

z0 = [2.9787004394953343, -0.03868302503393752,  0.058232737694740085, -0.02105288273117459]

recordFromSolutionLor(x, p; k...) = (u = BK.getvec(x);(X = u[1], Y = u[2], Z = u[3], U = u[4]))
prob = BK.BifurcationProblem(Lor!, z0, parlor, (@optic _.F);
	record_from_solution = (x, p; k...) -> (X = x[1], Y = x[2], Z = x[3], U = x[4]),)

opts_br = ContinuationPar(p_min = -1.5, p_max = 3.0, ds = 0.002, dsmax = 0.05, n_inversion = 6, nev = 4, max_steps = 200)
@reset opts_br.newton_options.tol = 1e-12
br = @time continuation(re_make(prob, params = (parlor..., T=0.04, F=3.)),
	 	PALC(), opts_br;
		normC = norminf, bothside = true)

scene = plot(br, plotfold=false, markersize=4, legend=:topleft)