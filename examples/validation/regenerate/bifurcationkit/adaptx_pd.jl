include("periodic_export.jl")
function main()
    case,output=cli("adaptx.cycle-pd-switch");c=PERIODIC_CONTRACT["adaptx"]
    try
        rhs(u,p)=[u[2],u[3],-p.alpha*u[3]-u[2]-u[1]+u[1]^2]
        prob=BifurcationProblem(rhs,zeros(3),(alpha=1.1,),(@optic _.alpha))
        eq=continuation(prob,PALC(),periodic_options(c;bounds=(.9,1.1),ds=-c["ds"]);verbosity=0)
        hi=periodic_special(eq,:hopf,c["hopf_window"])
        opts=periodic_options(c;bounds=(c["base_domain"][1],1.01),ds=-c["ds"])
        br=continuation(eq,hi,opts,Collocation(c["ntst"],c["ncol"];jacobian=BK.DenseAnalyticalInplace());δp=-.01,verbosity=0,plot=false,eigsolver=BK.FloquetColl())
        pd=periodic_special(br,:pd,c["pd_window"])
        br.specialpoint[pd].status==:converged || error("Unrefined PD")
        doubled=continuation(br,pd,ContinuationPar(opts,p_min=c["base_domain"][1]);δp=-.0001,ampfactor=.05,verbosity=0,plot=false,eigsolver=BK.FloquetColl())
        periodic_special(doubled,:pd,c["doubled_pd_window"])
        rows,events,modes,orbits=periodic_tables(br,"base",rhs,:alpha)
        r2,e2,m2,o2=periodic_tables(doubled,"doubled",rhs,:alpha)
        minimum(r["parameter"] for r in rows)<=c["base_domain"][1] || error("Incomplete base domain")
        minimum(r["parameter"] for r in r2)<=c["doubled_domain"][1] || error("Incomplete doubled domain")
        maximum(r["parameter"] for r in r2)>=c["doubled_domain"][2] || error("Incomplete upper doubled domain")
        maximum(r["half_period_difference"] for r in r2)>1e-3 || error("Repeated base traversal on doubled branch")
        ratio=r2[1]["period"]/br.specialpoint[pd].x[end]
        abs(ratio-2)<.005 || error("Period ratio $ratio fails")
        refinement=refinement_record(br,rhs,:alpha,c)
        append!(rows,r2);append!(events,e2);append!(modes,m2);append!(orbits,o2)
        write_table(joinpath(output,"orbits.csv"),orbits,["segment_id","point_id","phase_fraction","time","state_0","state_1","state_2"])
        export_bundle(case,output,rows,events,modes;driver=@__FILE__,equation=@__FILE__,parameter_order=["alpha","beta"],controls=Dict("ds"=>c["ds"],"ds_min"=>c["ds_min"],"ds_max"=>c["ds_max"],"max_steps"=>c["max_steps"],"newton_atol"=>1e-10,"event_bisection_inversions"=>16,"tol_bisection_eigenvalue"=>0),seeds=Dict("equilibrium"=>zeros(3),"alpha"=>1.1,"beta"=>1,"hopf"=>eq.specialpoint[hi].param),mesh=Dict("ntst"=>c["ntst"],"ncol"=>c["ncol"],"doubled_ntst"=>2*c["ntst"]))
        path=joinpath(output,"metadata.json");meta=JSON.parsefile(path)
        merge!(meta,Dict("comparable"=>false,"noncomparability_reasons"=>["Native BK PALC state metric differs from JaxCont collocation layout and MatCont Euclidean coordinates"],"period_ratio"=>ratio,"refinement"=>refinement,"coverage_status"=>"COMPLETE","hopf_alpha"=>eq.specialpoint[hi].param,"hopf_period"=>2pi/maximum(abs.(imag.(eigvals(BK.jacobian(prob,zeros(3),(alpha=eq.specialpoint[hi].param,))))))))
        meta["source_hashes"]["periodic_export.jl"]=hashfile(joinpath(@__DIR__,"periodic_export.jl"))
        meta["source_hashes"]["periodic.json"]=hashfile(joinpath(@__DIR__,"../../cases/systems/periodic.json"))
        open(path,"w") do io;JSON.print(io,meta,2);end
    catch err
        periodic_failure(case,output,err);rethrow()
    end
end
main()
