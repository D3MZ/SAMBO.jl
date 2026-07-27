using SAMBO
using Documenter

DocMeta.setdocmeta!(SAMBO, :DocTestSetup, :(using SAMBO); recursive=true)

makedocs(;
    modules=[SAMBO],
    authors="Demetrius Michael",
    sitename="SAMBO.jl",
    format=Documenter.HTML(;
        canonical="https://D3MZ.github.io/SAMBO.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        # Contract sources: spec/api.md, spec/spaces.md, spec/termination.md,
        # spec/interop.md is represented by the package-extension contract.
        "API" => "api/index.md",
        "Spaces" => "spaces/index.md",
        "Termination" => "termination/index.md",
        "Interoperability" => "interop/index.md",
    ],
)

deploydocs(;
    repo="github.com/D3MZ/SAMBO.jl",
    devbranch="main",
)
