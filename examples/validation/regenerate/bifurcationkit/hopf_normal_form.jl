include("export.jl")
case,output=cli("hopf-normal-form.l1")
function Fbp(u,p)
    x,y=u;mu=p.mu
    [mu*x-y+x^2-x^3-x*y^2,x+mu*y+x*y-y^3]
end
Jbp(u,p)=BK.ForwardDiff.jacobian(x->Fbp(x,p),u)
prob=BifurcationProblem(Fbp,[0.,0.],(mu=-1.,),(@optic _.mu))
opts=ContinuationPar(p_min=-2.,p_max=2.,ds=.01,dsmax=.05,max_steps=150,nev=2,detect_bifurcation=3,
    save_sol_every_step=1,n_inversion=8,max_bisection_steps=55,newton_options=NewtonPar(tol=1e-12,max_iterations=25))
br=continuation(prob,PALC(),opts;normC=norminf,bothside=true)
matches=[i for (i,s) in enumerate(br.specialpoint) if s.type==:hopf && -.01<=s.param<=.01]
length(matches)==1 || error("Unique asymmetric Hopf required")
i=only(matches); hp=get_normal_form(br,i;verbose=false)
rows=Dict{String,Any}[];spectra=Dict{String,Any}[]
for (j,s) in enumerate(br.sol)
    p=(mu=s.p,);row=equilibrium_row(s.x,p,Fbp,Jbp;point=j-1,coordinate=:mu)
    push!(rows,row);append!(spectra,spectrum_rows(row,Jbp,s.x,p))
end
sp=br.specialpoint[i];row=equilibrium_row(sp.x,(mu=sp.param,),Fbp,Jbp;point=argmin([abs(r["parameter"]-sp.param) for r in rows])-1,coordinate=:mu)
e=event_row(row,"H","hopf";bracket=sp.interval)
e["frequency"]=abs(hp.ω);e["l1_native_real"]=real(hp.nf.b);e["l1_native_imag"]=imag(hp.nf.b)
append!(spectra,spectrum_rows(row,Jbp,sp.x,(mu=sp.param,);event="hopf"))
export_bundle(case,output,rows,[e],spectra;driver=@__FILE__,parameter_order=["mu"],controls=Dict("newton_atol"=>1e-12,"ds"=>.01,"ds_max"=>.05,"bounds"=>[-2.,2.]),
    seeds=Dict("state"=>[0.,0.],"mu"=>-1.),conventions=Dict("normal_form"=>"BK native complex cubic b; real(b) exported without conversion to another tool convention", "eigenvector"=>"BK get_normal_form default right norm=1 and adjoint pairing=1", "time"=>"original"))
