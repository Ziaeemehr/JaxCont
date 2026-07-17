using Revise, Plots
using BifurcationKit
const BK = BifurcationKit

# vector field
function TMvf(z, p)
	(;J, α, E0, τ, τD, τF, U0) = p
	E, x, u = z
	SS0 = J * u * x * E + E0
	SS1 = α * log(1 + exp(SS0 / α))
	[
		(-E + SS1) / τ,
		(1.0 - x) / τD - u * x * E,
		(U0 - u) / τF +  U0 * (1.0 - u) * E
	]
end

# parameter values
par_tm = (α = 1.5, τ = 0.013, J = 3.07, E0 = -2.0, τD = 0.200, U0 = 0.3, τF = 1.5, τS = 0.007)

# initial condition
z0 = [0.238616, 0.982747, 0.367876]

# Bifurcation Problem
prob = BifurcationProblem(TMvf, z0, par_tm, (@optic _.E0);
	record_from_solution = (x, p; k...) -> (E = x[1], x = x[2], u = x[3]),)

# continuation options, we limit the parameter range for E0
opts_br = ContinuationPar(p_min = -4.0, p_max = -0.9)

# continuation of equilibria
br = continuation(prob, PALC(), opts_br;
	# we want to compute both sides of the branch of the initial
	# value of E0 = -2
	bothside = true)

scene = plot(br, legend=:topleft)    