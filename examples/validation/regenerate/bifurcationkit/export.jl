# Native candidate exporter: no Python numerical data is read.
using BifurcationKit, Accessors, LinearAlgebra, CSV, JSON, SHA, Dates
const BK = BifurcationKit

function cli(default_case)
    case=default_case; output=""
    for i in 1:2:length(ARGS)
        ARGS[i]=="--case" ? (case=ARGS[i+1]) : ARGS[i]=="--output-dir" ? (output=abspath(ARGS[i+1])) : error("Unknown argument $(ARGS[i])")
    end
    isempty(output) && error("--output-dir required")
    reference=realpath(joinpath(@__DIR__,"../../reference"))
    # Reject lexical paths and symlink ancestors into the immutable tree.
    ancestor=output
    while !ispath(ancestor); ancestor=dirname(ancestor); end
    resolved=joinpath(realpath(ancestor),relpath(output,ancestor))
    (startswith(resolved,reference*"/") || resolved==reference || startswith(reference,resolved*"/")) && error("Candidate output overlaps reference")
    mkpath(output)
    rm(joinpath(output,"metadata.json");force=true)
    case,output
end
hashfile(path)="sha256:"*bytes2hex(sha256(read(path)))
function write_table(path, rows, required)
    columns=sort!(collect(union(Set(required), (Set(keys(r)) for r in rows)...)))
    open(path,"w") do io
        println(io,join(columns,","))
        for row in rows
            values=[get(row,k,"") for k in columns]
            println(io,join(("\""*replace(string(v),'"'=>"\"\"")*"\"" for v in values),","))
        end
    end
end
function export_bundle(case,output,branch,events,spectra; driver, equation=driver, controls, seeds, mesh=Dict("kind"=>"equilibrium"), conventions=Dict("time"=>"original"), chart="completed", parameter_order=["F","T"])
    tables=Dict("branch.csv"=>(branch,["segment_id","point_id","order","parameter","residual_norm","unstable_dimension","status"]),
                "events.csv"=>(events,["event_id","event_type","status","segment_id","point_id","parameter","bracket_lower","bracket_upper"]),
                "spectra.csv"=>(spectra,["segment_id","point_id","event_id","kind","mode_id","real","imag","logabs","phase","trivial_mode_method"]))
    for (name,(rows,required)) in tables; write_table(joinpath(output,name),rows,required); end
    metadata=Dict("schema_version"=>2,"canonical_id"=>case,"tool"=>"bifurcationkit","runtime_version"=>string(VERSION),
        "package_versions"=>Dict("BifurcationKit"=>string(pkgversion(BK)),"CSV"=>string(pkgversion(CSV)),"JSON"=>string(pkgversion(JSON))),
        "lock_hash"=>hashfile(joinpath(@__DIR__,"Manifest.toml")),"precision"=>"float64", "equation_hash"=>hashfile(equation),"driver_hash"=>hashfile(driver),
        "platform"=>Dict("os"=>string(Sys.KERNEL),"cpu"=>Sys.CPU_NAME,"blas"=>sprint(show,BLAS.get_config())),
        "parameter_order"=>parameter_order,"scaling"=>Dict("time"=>"original","arclength"=>"BK PALC theta=0.5: state weight 0.5/n; parameter weight 0.5"),
        "seeds"=>seeds,"mesh"=>mesh,"resolved_controls"=>controls,"solver_strategies"=>Dict("method"=>"BifurcationKit PALC", "event_selection"=>"type and parameter window"),
        "command"=>["julia","--project="*@__DIR__,abspath(driver),ARGS...],"generated_utc"=>Dates.format(now(UTC),dateformat"yyyy-mm-ddTHH:MM:SS")*"Z",
        "generator_revision"=>strip(read(`git rev-parse HEAD`,String)),"reviewed"=>false,"reviewer"=>"unknown","review_date"=>"unknown","conversion_chain"=>[],
        "output_hashes"=>Dict(name=>hashfile(joinpath(output,name)) for name in union(collect(keys(tables)),filter(n->isfile(joinpath(output,n)),["orbits.csv","samples.json"]))),"conventions"=>conventions,"chart_status"=>chart)
    metadata["source_hashes"]=Dict(basename(p)=>hashfile(p) for p in [joinpath(@__DIR__,"export.jl"),equation,driver])
    if case in ("lorenz84.fold-ft","lorenz84.hopf-ft")
        domain=[minimum(r["T"] for r in branch),maximum(r["T"] for r in branch)]
        # Same rule as chart_coverage in cases/systems/lorenz84.py: the Hopf
        # chart ends at the BT (T~0.02094, tolerance 1e-3); the fold chart spans [-0.01, 0.04].
        ishopf=case=="lorenz84.hopf-ft"
        expected=ishopf ? [.02094,.04] : [-.01,.04]
        metadata["observed_domain"]=domain;metadata["expected_domain"]=expected
        metadata["coverage_status"]=(domain[1]<=expected[1]+(ishopf ? 1e-3 : 0.) && domain[2]>=expected[2]) ? "COMPLETE" : "PARTIAL"
        metadata["coverage_status"]=="PARTIAL" && (metadata["chart_status"]="Chart boundary or configured step budget reached before expected T domain")
    end
    open(joinpath(output,"metadata.json"),"w") do io; JSON.print(io,metadata,2); end
end
function equilibrium_row(u,p,rhs,jac;segment="equilibrium",point=0,coordinate=:F)
    eigenvalues=eigvals(jac(u,p))
    row=Dict{String,Any}("segment_id"=>segment,"point_id"=>string(point),"order"=>point,"parameter"=>getproperty(p,coordinate),
        "residual_norm"=>norm(rhs(u,p),Inf),"unstable_dimension"=>count(z->real(z)>1e-7,eigenvalues),"status"=>"CONVERGED")
    for name in propertynames(p); row[string(name)]=getproperty(p,name); end
    for (i,x) in enumerate(u); row["state_$(i-1)"]=x; end
    row
end
function spectrum_rows(row,jac,u,p;event="")
    [Dict{String,Any}("segment_id"=>row["segment_id"],"point_id"=>row["point_id"],"event_id"=>event,"kind"=>"EIGENVALUE","mode_id"=>string(i-1),
        "real"=>real(z),"imag"=>imag(z),"logabs"=>log(max(abs(z),floatmin(Float64))),"phase"=>angle(z),"trivial_mode_method"=>"not-applicable") for (i,z) in enumerate(eigvals(jac(u,p)))]
end
# In these smooth equilibrium fixtures BK labels a simple real zero crossing
# :bp. Native labels are also preserved; fold identity is checked by windows.
normalize_kind(type)=get(Dict(:fold=>"LP",:bp=>"LP",:hopf=>"H",:bt=>"BT",:zh=>"ZH",:gh=>"GH",:hh=>"HH"),type,string(type))
special_state(s)=hasproperty(s.x,:p1) ? s.x.x : hasproperty(s.x,:u) ? s.x.u : s.x[1:4]
special_F(s)=hasproperty(s.x,:p1) ? s.x.p1 : hasproperty(s.x,:p) ? (s.x.p isa Number ? s.x.p : s.x.p[1]) : s.x[5]
function select_special(branch,kind;F=(-Inf,Inf),T=(-Inf,Inf),codim2=false)
    matches=Int[]
    for (i,s) in enumerate(branch.specialpoint)
        normalize_kind(s.type)==kind || continue
        fp=codim2 ? special_F(s) : s.param
        tp=codim2 ? s.param : .04
        F[1]<=fp<=F[2] && T[1]<=tp<=T[2] && push!(matches,i)
    end
    length(matches)==1 || error("Expected unique $kind in F=$F,T=$T, found $(length(matches))")
    only(matches)
end
function event_row(row,kind,id; bracket=(row["parameter"],row["parameter"]))
    merge(row,Dict("event_type"=>kind,"kind"=>kind,"event_id"=>id,"bracket_lower"=>minimum(bracket),"bracket_upper"=>maximum(bracket)))
end
