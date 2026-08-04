using BifurcationKit
const BK = BifurcationKit

# Asymmetric Hopf example used to cross-validate
# jaxcont.bifurcations.hopf_normal_form.lyapunov_coefficient (see
# docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md). Deliberately
# non-symmetric (unlike the standard textbook polar-coordinates example) so
# the cubic-form B-based correction terms in Kuznetsov's l1 formula are
# actually exercised, not identically zero.
function Fbp(u, p)
    x, y = u
    mu = p.mu
    return [mu*x - y + x^2 - x^3 - x*y^2,
            x + mu*y + x*y - y^3]
end

par = (mu = -1.0,)
u0 = [0.0, 0.0]
prob = BK.BifurcationProblem(Fbp, u0, par, (@optic _.mu))
opts = ContinuationPar(p_min = -2.0, p_max = 2.0, ds = 0.01, dsmax = 0.05,
                        n_inversion = 8, max_bisection_steps = 25, nev = 2)
br = continuation(prob, PALC(), opts; normC = norminf, bothside = true)

hopfidx = [i for (i, bp) in enumerate(br.specialpoint) if bp.type == :hopf]
for i in hopfidx
    hp = get_normal_form(prob, br, i)
    # BifurcationKit.jl's `b` plays the role of the amplitude-equation cubic
    # coefficient -- its own source (NormalForms.jl) calls it "the Lyapunov
    # coefficient" in an internal warning message. It equals 2*l1 in
    # Kuznetsov's normalization: verified against the exact-known textbook
    # example first (l1=-1 there; BK independently gives b=-2.0 exactly)
    # before trusting this relationship on this asymmetric example.
    println("Hopf at p=", hp.p, " omega0=", hp.ω, "  BK b=", real(hp.nf.b),
            "  l1 = b/2 = ", real(hp.nf.b) / 2)
end
