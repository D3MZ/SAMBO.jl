using Sambo
using Documenter

DocMeta.setdocmeta!(Sambo, :DocTestSetup, :(using Sambo); recursive=true)

makedocs(;
    modules=[Sambo],
    authors="Demetrius Michael",
    sitename="Sambo.jl",
    format=Documenter.HTML(;
        canonical="https://D3MZ.github.io/Sambo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/D3MZ/Sambo.jl",
    devbranch="main",
)
