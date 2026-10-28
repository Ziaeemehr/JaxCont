include("export.jl")
case,output=cli("radial.cycle-floquet")
function radial(u,p)
    x,y=u;r2=x*x+y*y
    [(p.rho-r2)*x-y,(p.rho-r2)*y+x]
end
par=(rho=.5,)
prob=BifurcationProblem(radial,[sqrt(.5),0.],par,(@optic _.rho))
ntst=10;ncol=4;nn=2*(1+ntst*ncol)
coll=BK.Collocation(ntst,ncol;prob_vf=prob,N=2,ϕ=zeros(nn),xπ=zeros(nn),jacobian=BK.AutoDiffDense())
seed=BK.generate_solution(coll,t->sqrt(.5).*[cos(t),sin(t)],2pi)
# Fix the phase section from the sampled seed and its tangent.
BK.updatesection!(coll,seed,nothing)
opts=ContinuationPar(p_min=.5,p_max=1.5,ds=.05,dsmax=.08,max_steps=35,nev=2,detect_bifurcation=1,
    save_sol_every_step=1,newton_options=NewtonPar(tol=1e-12,max_iterations=25))
sol=newton(coll,seed,opts.newton_options)
BK.converged(sol) || error("Radial collocation seed correction failed")
br=continuation(coll,sol.u,PALC(),opts;verbosity=0,plot=false,eigsolver=BK.FloquetGEV(DefaultEig(),nn,2))
rows=Dict{String,Any}[];spectra=Dict{String,Any}[];orbits=Dict{String,Any}[]
for (i,s) in enumerate(br.sol)
    p=(rho=s.p,);orbit=BK.get_periodic_orbit(coll,s.x,p);period=s.x[end]
    row=Dict{String,Any}("segment_id"=>"cycle","point_id"=>string(i-1),"order"=>i-1,"parameter"=>s.p,"rho"=>s.p,"period"=>period,
        "residual_norm"=>norm(BK.po_residual(coll,s.x,p),Inf),"unstable_dimension"=>0,"status"=>"CONVERGED",
        "state_0"=>maximum(orbit.u[1,:]),"state_1"=>minimum(orbit.u[1,:]))
    push!(rows,row)
    for (j,t) in enumerate(orbit.t)
        push!(orbits,Dict("segment_id"=>"cycle","point_id"=>string(i-1),"time"=>t,"state_0"=>orbit.u[1,j],"state_1"=>orbit.u[2,j]))
    end
    # BifurcationKit periodic eigenvalues are log multipliers, not multipliers.
    ev=BK.eigenvals(br,s.step)
    vectors=br.eig[s.step+1].eigenvecs
    flow=radial(orbit.u[:,1],p)
    alignment=[abs(dot(vectors[1:2,j],flow))/(norm(vectors[1:2,j])*norm(flow)) for j in eachindex(ev)]
    trivial=argmax(alignment)
    alignment[trivial]>.99 || error("No Floquet mode aligns with the physical flow")
    for (j,z) in enumerate(ev)
        push!(spectra,Dict("segment_id"=>"cycle","point_id"=>string(i-1),"event_id"=>"","kind"=>"FLOQUET_MULTIPLIER","mode_id"=>string(j-1),
            "real"=>real(exp(z)),"imag"=>imag(exp(z)),"logabs"=>real(z),"phase"=>imag(z),"trivial_mode_method"=>"flow alignment at orbit phase zero", "flow_alignment"=>alignment[j], "is_trivial"=>Int(j==trivial)))
    end
end
write_table(joinpath(output,"orbits.csv"),orbits,["segment_id","point_id","time","state_0","state_1"])
export_bundle(case,output,rows,Dict{String,Any}[],spectra;driver=@__FILE__,parameter_order=["rho"],controls=Dict("newton_atol"=>1e-12,"ds"=>.05,"ds_max"=>.08,"max_steps"=>35,"bounds"=>[.5,1.5]),
    seeds=Dict("rho"=>.5,"orbit"=>"sqrt(rho)*(cos(t),sin(t))","period"=>2pi),mesh=Dict("ntst"=>ntst,"ncol"=>ncol),
    conventions=Dict("time"=>"original","floquet"=>"native FloquetGEV log multipliers exponentiated; trivial mode selected by eigenvector alignment with physical flow at phase zero"))
