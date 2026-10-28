include("lorenz84_equilibria.jl")
case,output=cli("lorenz84.fold-ft")
prob,br,opts=lorenz_branch()
ishopf=case=="lorenz84.hopf-ft"
kind=ishopf ? "H" : "LP"
seedidx=select_special(br,kind;F=ishopf ? (1.59,1.65) : (1.50,1.58),T=(.039,.041))
curveopts=ContinuationPar(opts;p_max=3.2,p_min=-.1,detect_bifurcation=1,dsmin=1e-5,ds=-.001,dsmax=.015,
    n_inversion=10,save_sol_every_step=1,max_steps=30,max_bisection_steps=55)
curve=continuation((@set br.alg.tangent=Secant()),seedidx,(@optic _.T),curveopts;
    verbosity=0,normC=norminf,detect_codim2_bifurcation=1,update_minaug_every_step=1,
    start_with_eigen=true,bdlinsolver=MatrixBLS(),
    record_from_solution=(u,p;k...)->(X=u[1],Y=u[2],Z=u[3],U=u[4]))
rows=Dict{String,Any}[];events=Dict{String,Any}[];spectra=Dict{String,Any}[]
for (i,s) in enumerate(curve.sol)
    u=s.x.x;p=(F=s.x.p1,T=s.p)
    row=equilibrium_row(u,p,Lor,jac_Lor;segment=kind,point=i-1,coordinate=:T)
    push!(rows,row);append!(spectra,spectrum_rows(row,jac_Lor,u,p))
end
for (i,s) in enumerate(curve.specialpoint)
    eventkind=normalize_kind(s.type)
    eventkind in ("BT","ZH","GH","HH") || continue
    u=special_state(s);p=(F=special_F(s),T=s.param)
    row=equilibrium_row(u,p,Lor,jac_Lor;segment=kind,point=argmin([abs(r["T"]-s.param) for r in rows])-1,coordinate=:T)
    event=event_row(row,eventkind,"e$i";bracket=s.interval)
    event["native_status"]=string(s.status);event["status"]=uppercase(string(s.status));event["native_label"]=string(s.type)
    push!(events,event);append!(spectra,spectrum_rows(row,jac_Lor,u,p;event="e$i"))
end
if case=="lorenz84.bt-ft"
    # This selector is immutable. A point outside it is a failed route.
    try
        idx=select_special(curve,"BT";F=(1.5,2.),T=(.01,.03),codim2=true)
        sol=newton(curve,idx;options=NewtonPar(tol=1e-13,max_iterations=25),start_with_eigen=true,jacobian_ma=BK.MinAug(),normN=norminf)
        BK.converged(sol) || error("BT Newton refinement failed")
        u=sol.u.x0;p=(F=sol.u.params.F,T=sol.u.params.T)
        (1.5<=p.F<=2. && .01<=p.T<=.03) || error("Refined BT outside reviewed window")
        row=equilibrium_row(u,p,Lor,jac_Lor;segment="BT",point=0,coordinate=:T)
        e=event_row(row,"BT","bt");e["geometric_nullity"]=count(x->x<1e-6,svdvals(jac_Lor(u,p)))
        rows=[row];events=[e];spectra=vcat(spectrum_rows(row,jac_Lor,u,p),spectrum_rows(row,jac_Lor,u,p;event="bt"))
    catch err
        open(joinpath(output,"failure.json"),"w") do io
            JSON.print(io,Dict("status"=>"FAIL","reason"=>sprint(showerror,err),"detected_curve_events"=>events),2)
        end
        rethrow()
    end
end
chart=ishopf && minimum(r["T"] for r in rows)>-.01 ? "Hopf chart boundary near BT; not a BT detection" : "completed configured steps"
export_bundle(case,output,rows,events,spectra;driver=@__FILE__,equation=joinpath(@__DIR__,"lorenz84_equilibria.jl"),
    controls=Dict("parameter_order"=>["F","T"],"free_parameter"=>"T","ds"=>-.001,"ds_min"=>1e-5,"ds_max"=>.015,"max_steps"=>30,"newton_atol"=>1e-12,"T_bounds"=>[-.1,3.2]),
    seeds=Dict("state"=>LOR_SEED,"F"=>3.,"T"=>.04,"curve_seed"=>kind,"window"=>ishopf ? [1.59,1.65] : [1.5,1.58]),
    conventions=Dict("time"=>"original","chart_boundary"=>"Hopf termination near BT is not a BT detection","parameter_order"=>"F,T"),chart=chart)
