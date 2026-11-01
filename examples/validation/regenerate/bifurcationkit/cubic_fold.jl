include("export.jl")
case,output=cli("cubic.eq-fold")
F(u,p)=[p.p+u[1]-u[1]^3/3]
J(u,p)=reshape([1-u[1]^2],1,1)
seed=[-2.1038034027355366]
prob=BifurcationProblem(F,seed,(p=-1.,),(@optic _.p);J=J,record_from_solution=(u,p;k...)->(x=u[1],))
opts=ContinuationPar(p_min=-1.,p_max=1.,ds=.01,dsmax=.05,max_steps=300,nev=1,detect_bifurcation=3,
    save_sol_every_step=1,n_inversion=8,max_bisection_steps=55,newton_options=NewtonPar(tol=1e-12,max_iterations=25))
br=continuation(prob,PALC(),opts;normC=norminf)
rows=Dict{String,Any}[];events=Dict{String,Any}[];spectra=Dict{String,Any}[]
for (i,s) in enumerate(br.sol)
    p=(p=s.p,); row=equilibrium_row(s.x,p,F,J;point=i-1,coordinate=:p)
    push!(rows,row);append!(spectra,spectrum_rows(row,J,s.x,p))
end
for (i,sp) in enumerate(br.specialpoint)
    normalize_kind(sp.type)=="LP" || continue
    row=equilibrium_row(sp.x,(p=sp.param,),F,J;point=argmin([abs(r["parameter"]-sp.param) for r in rows])-1,coordinate=:p)
    e=event_row(row,"LP","e$i";bracket=sp.interval)
    # BK's native fold normal form x' = a01*dp + b20/2*x^2 + ... with
    # |zeta| = 1 and <zeta*, zeta> = 1, the MatCont normalization of q and p;
    # MatCont's coefficient a = <p, B(q,q)>/2 is therefore b20/2 (analytic
    # cubic: +1 at x=-1, -1 at x=+1). The native value is kept alongside.
    nf=get_normal_form(br,i;verbose=false)
    e["fold_coefficient"]=nf.nf.b20/2;e["native_b20"]=nf.nf.b20
    e["fold_coefficient_convention"]="MatCont a = BK b20/2"
    push!(events,e);append!(spectra,spectrum_rows(row,J,sp.x,(p=sp.param,);event="e$i"))
end
println("Cubic special types: ",[(s.type,s.param) for s in br.specialpoint]); length(events)==2 || error("Full cubic branch must contain two folds; points=$(length(rows))")
export_bundle(case,output,rows,events,spectra;driver=@__FILE__,parameter_order=["p"],controls=Dict("newton_atol"=>1e-12,"ds"=>.01,"ds_max"=>.05,"max_steps"=>300,"bounds"=>[-1.,1.]),seeds=Dict("state"=>seed,"p"=>-1.))
