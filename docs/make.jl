using Documenter
using Siena

DocMeta.setdocmeta!(Siena, :DocTestSetup, :(using Siena); recursive=true)

makedocs(
    sitename = "Siena.jl",
    modules = [Siena],
    authors = "Simone Santoni",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://Statistical-network-analysis-with-Julia.github.io/Siena.jl",
        edit_link = "main",
    ),
    repo = "https://github.com/Statistical-network-analysis-with-Julia/Siena.jl/blob/{commit}{path}#{line}",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Data Preparation" => "guide/data.md",
            "Effects" => "guide/effects.md",
            "Model Estimation" => "guide/estimation.md",
            "Goodness of Fit" => "guide/gof.md",
        ],
        "API Reference" => [
            "Types" => "api/types.md",
            "Effects" => "api/effects.md",
            "Estimation" => "api/estimation.md",
        ],
    ],
    warnonly = [:missing_docs, :docs_block],
)

deploydocs(
    repo = "github.com/Statistical-network-analysis-with-Julia/Siena.jl.git",
    devbranch = "main",
    versions = [
        "stable" => "dev", # serve dev docs at /stable until a release is tagged
        "dev" => "dev",
    ],
    push_preview = true,
)
