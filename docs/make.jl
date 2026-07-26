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
    ],
)

deploydocs(;
    repo="github.com/D3MZ/SAMBO.jl",
    devbranch="main",
)
