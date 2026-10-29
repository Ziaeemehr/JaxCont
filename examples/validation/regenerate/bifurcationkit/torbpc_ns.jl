include("periodic_export.jl")
function main()
    case,output=cli("torbpc.cycle-lpc-pd-ns");c=PERIODIC_CONTRACT["torbpc"]
    try
        function rhs(u,p)
            x,y,z=u
            [(-(.5+p.nu)*x+.5*y-.32858*x^3+.93358*(y-x)^3)/.6,
             .5*x-(-.1)*y-z-.93358*(y-x)^3,y+.001]
        end
        prob=BifurcationProblem(rhs,[.00125,-.001,.00052502],(nu=-.9,),(@optic _.nu))
        eq=continuation(prob,PALC(),periodic_options(c;bounds=(-.9,-.57),ds=.01);verbosity=0)
        hi=periodic_special(eq,:hopf,c["hopf_window"])
        opts=periodic_options(c;bounds=c["continuation_bounds"],ds=c["ds"])
        br=continuation(eq,hi,opts,Collocation(c["ntst"],c["ncol"];jacobian=BK.DenseAnalyticalInplace());δp=.00001,verbosity=0,plot=false,eigsolver=BK.FloquetColl())
        # At the double +1, rounding may give BK a tiny-angle :ns pair.
        # Select a converged native event in the independently computed LPC
        # window, then certify the two +1 multipliers and parameter turn below.
        lpcs=findall(s->s.type in (:bp,:ns,:fold) && s.status==:converged && c["lpc_window"][1]<=s.param<=c["lpc_window"][2],br.specialpoint)
        length(lpcs)==1 || error("No unique converged LPC candidate")
        lpc=only(lpcs)
        pd=periodic_special(br,:pd,c["pd_window"])
        ns=periodic_special(br,:ns,c["ns_window"])
        rows,events,modes,orbits=periodic_tables(br,"cycle",rhs,:nu)
        # Probe evidence: with 16 inversions the native event at LPC is :ns
        # with angle 1.39e-5 (rounding of the double +1), followed by :bp
        # status :guessL at T=8.445. The latter is not a refined LPC. Preserve
        # it as native BPC; certify the converged event spectrally instead.
        native_id="cycle-$lpc"
        e=only(filter(e->e["event_id"]==native_id,events))
        lm=filter(m->m["event_id"]==native_id,modes)
        count(m->abs(complex(m["real"],m["imag"])-1)<1e-4,lm)==2 || error("LPC lacks two +1 multipliers")
        e["native_type"]=string(br.specialpoint[lpc].type);e["event_type"]="LPC"
        abs(e["parameter"]-maximum(r["parameter"] for r in rows))<1e-8 || error("LPC has no parameter turn")
        minimum(r["parameter"] for r in rows)<=c["domain"][1] || error("Truncated torBPC domain")
        # Include refined event extrema when measuring parameter coverage.
        maximum(r["parameter"] for r in [rows;events])>=c["domain"][2]-1e-9 || error("Missing LPC endpoint")
        nse=only(filter(e->e["event_id"]=="cycle-$ns",events))
        pair=filter(m->m["event_id"]==nse["event_id"] && abs(m["imag"])>1e-3,modes)
        length(pair)==2 || error("Missing NS complex pair")
        nse["ns_angle"]=abs(pair[1]["phase"])
        .01<nse["ns_angle"]<pi-.01 || error("Resonant NS angle")
        for (kind,key) in [("LPC","lpc_window"),("PD","pd_window"),("NS","ns_window")]
            window=c[key]
            length(filter(e->e["event_type"]==kind && window[1]<=e["parameter"]<=window[2],events))==1 || error("Missing or duplicate $kind in frozen window")
        end
        # Local crossing uses native spectra at points on either side in
        # continuation order; no parameter sorting across the fold.
        sp=br.specialpoint[ns];i=argmin(abs.([s.p for s in br.sol].-sp.param))
        1<i<length(br.sol) || error("NS crossing has no two-sided samples")
        a=BK.eigenvals(br,br.sol[i-1].step);b=BK.eigenvals(br,br.sol[i+1].step)
        ca=maximum(real(z) for z in a if abs(imag(z))>.01);cb=maximum(real(z) for z in b if abs(imag(z))>.01)
        ca*cb<0 || error("NS pair does not cross unit circle")
        nse["crossing_logabs_before"]=ca;nse["crossing_logabs_after"]=cb
        refinement=refinement_record(br,rhs,:nu,c)
        write_table(joinpath(output,"orbits.csv"),orbits,["segment_id","point_id","phase_fraction","time","state_0","state_1","state_2"])
        export_bundle(case,output,rows,events,modes;driver=@__FILE__,parameter_order=["nu"],controls=Dict("ds"=>c["ds"],"ds_min"=>c["ds_min"],"ds_max"=>c["ds_max"],"max_steps"=>c["max_steps"],"bounds"=>c["continuation_bounds"],"newton_atol"=>1e-10,"tol_bisection_eigenvalue"=>0,"event_bisection_inversions"=>16),seeds=Dict("nu"=>-.9,"state"=>[.00125,-.001,.00052502],"hopf_nu"=>eq.specialpoint[hi].param),mesh=Dict("ntst"=>c["ntst"],"ncol"=>c["ncol"]))
        path=joinpath(output,"metadata.json");meta=JSON.parsefile(path)
        merge!(meta,Dict("comparable"=>false,"noncomparability_reasons"=>["Native BK PALC metric differs; JC uses ntst=10, MC and BK use ntst=25"],"refinement"=>refinement,"coverage_status"=>"COMPLETE"))
        for (name,file) in [("periodic_export.jl",joinpath(@__DIR__,"periodic_export.jl")),("periodic.json",joinpath(@__DIR__,"../../cases/systems/periodic.json"))];meta["source_hashes"][name]=hashfile(file);end
        open(path,"w") do io;JSON.print(io,meta,2);end
    catch err
        periodic_failure(case,output,err);rethrow()
    end
end
main()
