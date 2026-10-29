# Benchmark worker for cubic.eq-fold (READY, then JSON lines).
#   julia --project=<pinned BK env> bifurcationkit_worker.jl MODE REPS WARMUPS CONTROLS_JSON
# Startup (Julia launch, package load) ends at READY and is timed by the Python
# supervisor; the first call therefore includes Julia's own compilation, the
# analogue of JaxCont's first call. Accuracy is computed in Python from the
# returned branch and fold parameters, never reported here.
using BifurcationKit, JSON, LinearAlgebra
const BK = BifurcationKit

mode, repetitions, warmups = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3])
c = JSON.parse(ARGS[4])
# BK's PALC norm is θ/n |du|^2 + (1-θ) dp^2; with θ = 0.5 it is half the
# contract's 1/n |du|^2 + dp^2, so contract steps scale by sqrt(0.5).
const scale = sqrt(0.5)
F(u, p) = [p.p + u[1] - u[1]^3 / 3]
J(u, p) = reshape([1 - u[1]^2], 1, 1)
low, high = c["bounds"]
# The case seed (x, p) = (-2.1038..., -1); direction sets the sign of ds.
prob = BifurcationProblem(F, [-2.1038034027355366], (p = -1.0,), (@optic _.p); J = J)
opts = ContinuationPar(p_min = low, p_max = high, ds = c["direction"] * c["ds"] * scale,
    dsmin = c["ds_min"] * scale, dsmax = c["ds_max"] * scale, max_steps = c["max_steps"],
    nev = 1, detect_bifurcation = 3, detect_fold = true, save_sol_every_step = 1,
    newton_options = NewtonPar(tol = c["newton_atol"], max_iterations = c["newton_max_iter"]))
run_once() = continuation(prob, PALC(θ = 0.5), opts; verbosity = 0)

function emit(stage, with_results)
    seconds = @elapsed br = run_once()
    row = Dict{String, Any}("stage" => stage, "seconds" => seconds, "points" => length(br.sol))
    if with_results
        # On this scalar equation F_p = 1, so every zero eigenvalue is a fold;
        # BK 0.8.8 labels them :bp. Both labels count, and the native labels are kept.
        folds = [sp for sp in br.specialpoint if sp.type in (:fold, :bp)]
        row["folds"] = [sp.param for sp in folds]
        row["fold_states"] = [sp.x[1] for sp in folds]
        row["versions"] = Dict("julia" => string(VERSION), "bifurcationkit" => string(pkgversion(BifurcationKit)))
        row["native_labels"] = [string(sp.type) for sp in folds]
        row["states"] = [s.x[1] for s in br.sol]
        row["params"] = [s.p for s in br.sol]
    end
    println(JSON.json(row)); flush(stdout)
end

println("READY"); flush(stdout)
if mode == "first_call"
    emit("first_call", true)
else
    for _ in 1:warmups
        run_once()
    end
    for _ in 1:repetitions
        emit("warm", false)
    end
end
