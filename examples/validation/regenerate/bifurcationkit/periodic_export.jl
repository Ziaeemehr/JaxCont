# Native periodic tools for pinned BK 0.8.8. No optional ODE packages needed.
include("export.jl")
const PERIODIC_CONTRACT=JSON.parsefile(joinpath(@__DIR__,"../../cases/systems/periodic.json"))

function periodic_options(c; bounds, ds)
    ContinuationPar(p_min=bounds[1],p_max=bounds[2],ds=ds,dsmin=c["ds_min"],dsmax=c["ds_max"],max_steps=c["max_steps"],nev=3,
        detect_bifurcation=3,n_inversion=16,max_bisection_steps=80,
        # A trivial flow exponent is zero at EVERY periodic point. A positive
        # eigenvalue stopping tolerance prematurely terminates BK bisection.
        tol_bisection_eigenvalue=0.,dsmin_bisection=1e-12,tol_stability=1e-8,
        save_sol_every_step=1,newton_options=NewtonPar(tol=1e-10,max_iterations=25))
end
function periodic_special(br,kind,window)
    ids=findall(s->s.type==kind && window[1]<=s.param<=window[2],br.specialpoint)
    length(ids)==1 || error("Expected one native $kind in computed window $window; found $(length(ids))")
    only(ids)
end
rk4(f,u,h)=begin
    a=f(u);b=f(u+h/2*a);c=f(u+h/2*b);d=f(u+h*c)
    u+h/6*(a+2b+2c+d)
end
function independent_periodic_defect(coll,x,p,rhs;substeps=32)
    T=x[end];ntst=length(BK.getmesh(coll))-1
    h=T/ntst/substeps;worst=0.;closure=0.
    for i in 0:ntst-1
        u=BK.__interpolate_posolution(coll,i*T/ntst,x,T)
        for j in 1:substeps
            u=rk4(v->rhs(v,p),u,h)
            sample=BK.__interpolate_posolution(coll,(i+j/substeps)*T/ntst,x,T)
            worst=max(worst,norm(u-sample,Inf))
        end
        closure=max(closure,norm(u-BK.__interpolate_posolution(coll,mod(i+1,ntst)*T/ntst,x,T),Inf))
    end
    Dict("independent_dense_defect"=>worst,"independent_interval_closure"=>closure)
end
function periodic_tables(br,segment,rhs,parameter_name; special_kinds=Dict(:pd=>"PD",:ns=>"NS",:bp=>"BPC",:fold=>"LPC"))
    coll=BK.get_discretization(BK.getprob(br))
    rows=Dict{String,Any}[];events=Dict{String,Any}[];modes=Dict{String,Any}[];orbits=Dict{String,Any}[]
    function addpoint(x,a,id,order;step=nothing,event="",kind="",native_status="")
        p=NamedTuple{(parameter_name,)}((a,));orbit=BK.get_periodic_orbit(coll,x,p)
        row=Dict{String,Any}("segment_id"=>segment,"point_id"=>id,"order"=>order,"parameter"=>a,string(parameter_name)=>a,"period"=>x[end],
            "residual_norm"=>norm(BK.po_residual(coll,x,p)[1:end-1],Inf),"unstable_dimension"=>0,"status"=>"CONVERGED")
        for j in 1:3; row["state_$(j-1)_min"]=minimum(orbit.u[j,:]);row["state_$(j-1)_max"]=maximum(orbit.u[j,:]);end
        # Flow alignment uses physical state components of native Floquet vectors.
        # Compute the eigensystem at refined events too, without substituting a
        # nearby branch point's multipliers for the event spectrum.
        J=BK.jacobian(BK.getprob(br),x,p)
        fl=BK.FloquetGEV(DefaultEig(),length(x)-1,3); ev,vecs,_,_=fl(coll,J,3)
        flow=rhs(orbit.u[:,1],p)
        alignment=[abs(dot(vecs[1:3,j],flow))/(norm(vecs[1:3,j])*norm(flow)) for j in eachindex(ev)]
        trivial=argmax(alignment)
        row["unstable_dimension"]=count(j->j!=trivial && real(ev[j])>1e-7,eachindex(ev))
        samples=hcat([BK.__interpolate_posolution(coll,t*x[end],x,x[end]) for t in range(0,1,length=161)[1:end-1]]...)
        centered=samples.-sum(samples,dims=2)/size(samples,2)
        row["half_period_difference"]=maximum(sqrt.(sum(abs2,centered-circshift(centered,(0,80));dims=1)))/maximum(sqrt.(sum(abs2,centered;dims=1)))
        if !isempty(event) || order in (0,length(br.sol)-1)
            merge!(row,independent_periodic_defect(coll,x,p,rhs))
        end
        for (j,z) in enumerate(ev)
            push!(modes,Dict("segment_id"=>segment,"point_id"=>id,"event_id"=>event,"kind"=>"FLOQUET_MULTIPLIER","mode_id"=>string(j-1),"real"=>real(exp(z)),"imag"=>imag(exp(z)),"logabs"=>real(z),"phase"=>imag(z),"trivial_mode_method"=>"native eigenvector flow alignment","flow_alignment"=>alignment[j],"is_trivial"=>Int(j==trivial)))
        end
        for j in eachindex(orbit.t)
            push!(orbits,Dict("segment_id"=>segment,"point_id"=>id,"time"=>orbit.t[j],"phase_fraction"=>orbit.t[j]/x[end],"state_0"=>orbit.u[1,j],"state_1"=>orbit.u[2,j],"state_2"=>orbit.u[3,j]))
        end
        if isempty(event)
            push!(rows,row)
        else
            push!(rows,row) # Refined event has its own linked branch identity.
            e=event_row(row,kind,event);e["native_status"]=native_status;e["native_type"]=string(kind);push!(events,e)
        end
    end
    for (i,s) in enumerate(br.sol);addpoint(s.x,s.p,string(i-1),i-1;step=s.step);end
    for (i,s) in enumerate(br.specialpoint)
        haskey(special_kinds,s.type) || continue
        addpoint(s.x,s.param,"event-$i",s.step-.5;event="$segment-$i",kind=special_kinds[s.type],native_status=string(s.status))
        events[end]["native_type"]=string(s.type)
    end
    sort!(rows;by=r->r["order"])
    rows,events,modes,orbits
end
function refinement_record(br,rhs,parameter_name,c)
    coll=BK.get_discretization(BK.getprob(br));s=br.sol[end];p=NamedTuple{(parameter_name,)}((s.p,));T=s.x[end]
    nn=3*(1+2*c["ntst"]*c["ncol"])
    refined=BK.Collocation(2*c["ntst"],c["ncol"];prob_vf=coll.prob_vf,N=3,ϕ=zeros(nn),xπ=zeros(nn),jacobian=BK.AutoDiffDense())
    seed=BK.generate_solution(refined,t->BK.__interpolate_posolution(coll,t,s.x,T),T)
    BK.updatesection!(refined,seed,p)
    refined=@set refined.prob_vf.params=p
    sol=newton(refined,seed,NewtonPar(tol=1e-10,max_iterations=25))
    BK.converged(sol) || error("Mesh-refinement Newton failed")
    Dict("parameter"=>s.p,"coarse_ntst"=>c["ntst"],"fine_ntst"=>2*c["ntst"],"coarse"=>independent_periodic_defect(coll,s.x,p,rhs),"fine"=>independent_periodic_defect(refined,sol.u,p,rhs),"period_difference"=>abs(sol.u[end]-T))
end
function periodic_failure(case,output,err)
    rm(joinpath(output,"metadata.json");force=true)
    open(joinpath(output,"failure.json"),"w") do io
        JSON.print(io,Dict("canonical_id"=>case,"status"=>"FAIL","error"=>sprint(showerror,err),"manifest_hash"=>hashfile(joinpath(@__DIR__,"Manifest.toml"))),2)
    end
end
