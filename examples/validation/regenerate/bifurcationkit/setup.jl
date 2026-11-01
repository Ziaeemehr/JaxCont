# Explicit setup only. Tests and producers must never include this file.
using Pkg
Pkg.instantiate()
Pkg.precompile()
using BifurcationKit, Accessors, CSV, JSON, LinearAlgebra, SHA
packages = Dict(info.name => string(info.version) for info in values(Pkg.dependencies()) if info.is_direct_dep)
println(JSON.json(Dict(
    "julia" => string(VERSION),
    "packages" => packages,
    "manifest_sha256" => bytes2hex(sha256(read(joinpath(@__DIR__, "Manifest.toml")))),
)))
