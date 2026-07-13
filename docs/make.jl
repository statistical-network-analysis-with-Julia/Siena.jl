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
    repo = Documenter.Remotes.GitHub("Statistical-network-analysis-with-Julia", "Siena.jl"),
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
            # The effects API is split by family: a single page carrying all 131
            # effect docstrings exceeded Documenter's page-size threshold (and
            # was unnavigable).
            "Effects" => [
                "Overview & Management" => "api/effects.md",
                "Network Effects" => "api/effects_network.md",
                "Behavior Effects" => "api/effects_behavior.md",
                "Rate Effects" => "api/effects_rate.md",
                "Two-Mode Effects" => "api/effects_twomode.md",
            ],
            "Estimation" => "api/estimation.md",
        ],
    ],
    # STRICT. Undefined bindings, bad cross-references, duplicate docs and
    # malformed markdown are build ERRORS, so they cannot silently accumulate
    # again (a docs build that passes while warning is one that will rot).
    #
    # `checkdocs = :exports` is the one deliberate exclusion: every *exported*
    # name must be documented, but internal machinery (materialized/private
    # types, `Base`/`Graphs` method extensions, inner constructors) need not be
    # -- filler docstrings for names a user never types are worse than none.
    warnonly = false,
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/statistical-network-analysis-with-Julia/Siena.jl.git",
    devbranch = "main",
    versions = [
        "stable" => "dev", # serve dev docs at /stable until a release is tagged
        "dev" => "dev",
    ],
    push_preview = true,
)
