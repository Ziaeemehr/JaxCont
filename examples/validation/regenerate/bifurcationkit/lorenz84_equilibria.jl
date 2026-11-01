include("export.jl")
# Independent transcription of the reviewed Lorenz-84 contract.
function Lor(u,p)
    X,Y,Z,U=u
    [-Y^2-Z^2-.25X+.25p.F-.987U^2,
     X*Y-X*Z-Y+.25, X*Y+X*Z-Z, -1.04U+.987U*X+p.T]
end
function jac_Lor(u,p)
    X,Y,Z,U=u
    [-.25 -2Y -2Z -1.974U;
     Y-Z X-1 -X 0;
     Y+Z X X-1 0;
     .987U 0 0 -1.04+.987X]
end
const LOR_SEED=[2.9787004394953343,-.03868302503393752,.058232737694740085,-.02105288273117459]
const LOR_PAR=(F=3.,T=.04)
function lorenz_branch()
    prob=BK.BifurcationProblem(Lor,LOR_SEED,LOR_PAR,(@optic _.F); J=jac_Lor,
        record_from_solution=(x,p;k...)->(X=x[1],Y=x[2],Z=x[3],U=x[4]))
    # dsmax .01, not the contract .05: at .05 BK sampled 57 points and its
    # interpolation could not support the state/spectrum budgets near the
    # fold (review round 2, 2026-10-29); 1000 steps keep the full F range.
    opts=ContinuationPar(p_min=-1.5,p_max=3.,ds=.002,dsmax=.01,max_steps=1000,nev=4,
        detect_bifurcation=3,n_inversion=6,max_bisection_steps=55,save_sol_every_step=1,
        newton_options=NewtonPar(tol=1e-12,max_iterations=25))
    corrected=BK.solve(prob,Newton(),opts.newton_options;normN=norminf)
    BK.converged(corrected) || error("Contract seed correction failed")
    prob=BK.re_make(prob;u0=corrected.u)
    br=continuation(prob,PALC(tangent=Bordered()),opts;normC=norminf,bothside=true)
    prob,br,opts
end
function export_lorenz_eq(case,output)
    prob,br,opts=lorenz_branch()
    rows=Dict{String,Any}[]; spectra=Dict{String,Any}[]; events=Dict{String,Any}[]
    for (i,sol) in enumerate(br.sol)
        -1.5<=sol.p<=3. || continue
        p=(F=sol.p,T=.04); u=sol.x
        row=equilibrium_row(u,p,Lor,jac_Lor;point=i-1)
        push!(rows,row); append!(spectra,spectrum_rows(row,jac_Lor,u,p))
    end
    for (i,sp) in enumerate(br.specialpoint)
        kind=normalize_kind(sp.type)
        kind in ("LP","H") || continue
        -1.5<=sp.param<=3. || continue
        p=(F=sp.param,T=.04); u=sp.x
        nearest=argmin([norm(u-[r["state_$j"] for j in 0:3]) for r in rows])
        row=equilibrium_row(u,p,Lor,jac_Lor;point=parse(Int,rows[nearest]["point_id"]))
        e=event_row(row,kind,"e$i";bracket=sp.interval)
        e["native_status"]=string(sp.status);e["native_label"]=string(sp.type)
        if kind=="H"; e["frequency"]=abs(imag(eigvals(jac_Lor(u,p))[argmin(abs.(real.(eigvals(jac_Lor(u,p)))))])); end
        push!(events,e); append!(spectra,spectrum_rows(row,jac_Lor,u,p;event="e$i"))
    end
    for (kind,window) in (("LP",(1.5,1.58)),("H",(1.59,1.65)),("H",(2.4,2.55)),("H",(2.8,2.95)))
        select_special(br,kind;F=window)
    end
    samples=[Dict("state"=>u,"F"=>p.F,"T"=>p.T,"rhs"=>Lor(u,p),"jacobian"=>[collect(r) for r in eachrow(jac_Lor(u,p))]) for (u,p) in (([1.,.2,-.3,.1],(F=2.,T=.04)),(LOR_SEED,LOR_PAR))]
    open(joinpath(output,"samples.json"),"w") do io; JSON.print(io,samples,2); end
    export_bundle(case,output,rows,events,spectra;driver=@__FILE__,controls=Dict("ds"=>.002,"ds_max"=>.01,"max_steps_per_direction"=>1000,"newton_atol"=>1e-12,"bounds"=>[-1.5,3.],"both_directions"=>true),
        seeds=Dict("state"=>LOR_SEED,"F"=>3.,"T"=>.04),conventions=Dict("comparison_domain"=>"clip each sheet to F bounds before matching","time"=>"original","frequency"=>"angular frequency"))
end
if abspath(PROGRAM_FILE)==@__FILE__
    case,output=cli("lorenz84.eq-f")
    export_lorenz_eq(case,output)
end
