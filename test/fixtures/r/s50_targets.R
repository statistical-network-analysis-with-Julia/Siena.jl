#!/usr/bin/env Rscript
# Regenerate from the package root:
# Rscript test/fixtures/r/s50_targets.R > test/fixtures/s50_targets.toml
#
# Target statistics (RSiena:::getTargets) of EVERY effect Siena.jl offers under an
# RSiena short name, on RSiena's bundled s50 data. Targets are a deterministic
# function of the observed waves, so they pin the effect FORMULAS (and the moment
# conventions: symmetric halving, lagged degrees at parm 0, ...), not the estimator.
#
# Four blocks (the fourth, `twomode_*`, is a deterministic bipartite panel):
#  * `targets`: the original 34 selections (kept for continuity);
#  * `directed_*`: the directed catalogue on s50 friendship + a second dependent
#    network `fr2`, the dependent behaviour `alcohol`, the actor covariate `smoke1`
#    and the dyadic covariate `dc` (both deterministic);
#  * `undirected_*`: the catalogue RSiena defines for symmetric networks, on the
#    symmetrised s50 friendship waves (modelType 2 is RSiena's default there).
# Each spec is "variable|shortName|interaction1|parm" (empty parm = RSiena default).
suppressPackageStartupMessages(library(RSiena))
seed <- 20260914L
set.seed(seed) # no random draws are needed for these deterministic targets

target_of <- function(d, spec) {
    p <- strsplit(spec, "|", fixed = TRUE)[[1]]
    if (length(p) < 4) p <- c(p, rep("", 4 - length(p)))
    e <- getEffects(d)
    e$include <- e$shortName == "Rate" & e$type == "rate"
    type <- if (p[2] %in% c("Rate") || grepl("Rate", p[2])) "rate" else "eval"
    id <- which(e$name == p[1] & e$shortName == p[2] & e$interaction1 == p[3] &
                e$type == type)
    if (p[2] == "Rate") stop("basic rates are pinned through the `targets` block")
    stopifnot(length(id) == 1)
    e$include[id] <- TRUE
    if (p[4] != "") e$parm[id] <- as.numeric(p[4])
    tg <- RSiena:::getTargets(d, e)
    sum(tg[match(id, which(e$include)), ])
}

emit <- function(key, specs, values) {
    cat(sprintf('%s_specs = [%s]\n', key, paste(sprintf('"%s"', specs), collapse = ", ")))
    cat(sprintf('%s_targets = [%s]\n', key, paste(sprintf("%.17g", values), collapse = ", ")))
}

## ---- the original 34 selections --------------------------------------------
d <- sienaDataCreate(
    friendship = sienaDependent(array(c(s501, s502, s503), dim = c(50, 50, 3))),
    alcohol = sienaDependent(s50a, type = "behavior"),
    smoke1 = coCovar(s50s[, 1]))
e <- getEffects(d)
e$include <- FALSE
specs <- list(
    c("friendship", "Rate", "", "rate"),
    c("alcohol", "Rate", "", "rate"),
    c("friendship", "density", "", "eval"),
    c("friendship", "recip", "", "eval"),
    c("friendship", "transTrip", "", "eval"),
    c("friendship", "transMedTrip", "", "eval"),
    c("friendship", "transRecTrip", "", "eval"),
    c("friendship", "cycle3", "", "eval"),
    c("friendship", "transTies", "", "eval"),
    c("friendship", "between", "", "eval"),
    c("friendship", "nbrDist2", "", "eval"),
    c("friendship", "denseTriads", "", "eval"),
    c("friendship", "inPop", "", "eval"),
    c("friendship", "inPopSqrt", "", "eval"),
    c("friendship", "outPop", "", "eval"),
    c("friendship", "outAct", "", "eval"),
    c("friendship", "outActSqrt", "", "eval"),
    c("friendship", "inAct", "", "eval"),
    c("friendship", "isolateNet", "", "eval"),
    c("friendship", "altX", "smoke1", "eval"),
    c("friendship", "egoX", "smoke1", "eval"),
    c("friendship", "simX", "smoke1", "eval"),
    c("friendship", "sameX", "smoke1", "eval"),
    c("friendship", "egoXaltX", "smoke1", "eval"),
    c("alcohol", "linear", "", "eval"),
    c("alcohol", "quad", "", "eval"),
    c("alcohol", "avSim", "friendship", "eval"),
    c("alcohol", "totSim", "friendship", "eval"),
    c("alcohol", "indeg", "friendship", "eval"),
    c("alcohol", "outdeg", "friendship", "eval"),
    c("alcohol", "avAlt", "friendship", "eval"),
    c("alcohol", "effFrom", "smoke1", "eval"))
indices <- unlist(lapply(specs, function(s) {
    ids <- which(e$name == s[1] & e$shortName == s[2] &
                 e$interaction1 == s[3] & e$type == s[4])
    stopifnot(length(ids) == if (s[4] == "rate") 2 else 1)
    ids
}))
e$include[indices] <- TRUE
targets <- RSiena:::getTargets(d, e)
values <- rowSums(targets)
values <- values[match(indices, which(e$include))]

## ---- directed catalogue -----------------------------------------------------
dc <- matrix(0, 50, 50)
for (i in 1:50) for (j in 1:50) dc[i, j] <- ((i * 7 + j * 3) %% 5)
fr2 <- array(0, dim = c(50, 50, 3))
for (w in 1:3) for (i in 1:50) for (j in 1:50)
    if (i != j && ((i * 7 + j * 3 + w) %% 11) == 0) fr2[i, j, w] <- 1
dd <- sienaDataCreate(
    friendship = sienaDependent(array(c(s501, s502, s503), dim = c(50, 50, 3))),
    fr2 = sienaDependent(fr2),
    alcohol = sienaDependent(s50a, type = "behavior"),
    smoke1 = coCovar(s50s[, 1]), dc = coDyadCovar(dc))
fr <- function(sn, i1 = "", parm = "") sprintf("friendship|%s|%s|%s", sn, i1, parm)
al <- function(sn, i1 = "", parm = "") sprintf("alcohol|%s|%s|%s", sn, i1, parm)
covar <- c("egoX", "egoSqX", "altX", "altSqX", "simX", "sameX", "diffX", "diffSqX",
           "absDiffX", "higher", "egoXaltX", "egoPlusAltX", "sameXRecip",
           "simRecipX", "simXTransTrip")
directed <- c(
    fr("density"), fr("recip"), fr("transTrip"), fr("transMedTrip"),
    fr("transRecTrip"), fr("cycle3"), fr("transTies"), fr("between"),
    fr("nbrDist2"), fr("denseTriads"),
    fr("inPop"), fr("inPopSqrt"), fr("outPop"), fr("outPop", parm = "0"),
    fr("outPopSqrt"), fr("outPopSqrt", parm = "1"), fr("outPopSqrt", parm = "-1"),
    fr("inAct"), fr("inAct", parm = "0"), fr("inActSqrt"),
    fr("inActSqrt", parm = "1"), fr("inActSqrt", parm = "-1"),
    fr("outAct"), fr("outActSqrt"), fr("outTrunc"), fr("outTrunc", parm = "5"),
    fr("outIso"), fr("isolateNet"),
    fr("gwespFF"), fr("gwespBB"), fr("gwespFB"), fr("gwdspFF"),
    sapply(covar, fr, i1 = "smoke1"), sapply(covar, fr, i1 = "alcohol"),
    fr("X", "dc"), fr("crprod", "fr2"), fr("crprodRecip", "fr2"),
    fr("outRate"), fr("inRate"), fr("recipRate"), fr("outRateLog"),
    fr("inRateLog"), fr("outRateInv"), fr("inRateInv"), fr("RateX", "smoke1"),
    al("linear"), al("quad"), al("threshold", parm = "3"),
    al("avAlt", "friendship"), al("avSim", "friendship"), al("totAlt", "friendship"),
    al("totSim", "friendship"), al("avInAlt", "friendship"),
    al("avRecAlt", "friendship"), al("totInAlt", "friendship"),
    al("avAltDist2", "friendship"), al("indeg", "friendship"),
    al("outdeg", "friendship"), al("recipDeg", "friendship"),
    al("effFrom", "smoke1"),
    al("outRate", "friendship"), al("inRate", "friendship"),
    al("recipRate", "friendship"), al("RateX", "smoke1"))
directed <- unname(directed)
directed_values <- sapply(directed, function(s) target_of(dd, s))

## ---- undirected catalogue ---------------------------------------------------
sym <- function(m) { m[m > 1] <- 0; pmax(m, t(m)) }
du <- sienaDataCreate(
    friendship = sienaDependent(array(c(sym(s501), sym(s502), sym(s503)),
                                      dim = c(50, 50, 3))),
    alcohol = sienaDependent(s50a, type = "behavior"),
    smoke1 = coCovar(s50s[, 1]), dc = coDyadCovar(dc + t(dc)))
stopifnot(attr(du$depvars$friendship, "symmetric"))
ucovar <- c("egoX", "egoSqX", "altX", "altSqX", "simX", "sameX", "diffX", "diffSqX",
            "absDiffX", "higher", "egoXaltX", "egoPlusAltX")
undirected <- c(
    fr("density"), fr("transTriads"), fr("transTies"), fr("between"),
    fr("nbrDist2"), fr("gwesp"), fr("inPop"), fr("inPopSqrt"), fr("outAct"),
    fr("outActSqrt"), fr("degPlus"), fr("outTrunc"), fr("outIso"),
    fr("isolateNet"),
    sapply(ucovar, fr, i1 = "smoke1"), sapply(ucovar, fr, i1 = "alcohol"),
    fr("X", "dc"),
    fr("outRate"), fr("outRateLog"), fr("outRateInv"), fr("RateX", "smoke1"),
    al("avAlt", "friendship"), al("avSim", "friendship"),
    al("totAlt", "friendship"), al("totSim", "friendship"),
    al("avAltDist2", "friendship"), al("outdeg", "friendship"))
undirected <- unname(undirected)
undirected_values <- sapply(undirected, function(s) target_of(du, s))
ut <- RSiena:::getTargets(du, getEffects(du))
undirected_rates <- c(ut[1, 1], ut[2, 2])

## ---- two-mode (bipartite) catalogue ------------------------------------------
# A deterministic 50 x 12 affiliation panel for the s50 actors, with the dependent
# behaviour alcohol, the actor covariate smoke1 and an actor x event covariate.
tm <- array(0, dim = c(50, 12, 3))
for (w in 1:3) for (i in 1:50) for (ev in 1:12)
    if (((i * 5 + ev * 7 + w * (i %% 3 + ev)) %% 9) < 2) tm[i, ev, w] <- 1
dce <- matrix(0, 50, 12)
for (i in 1:50) for (ev in 1:12) dce[i, ev] <- (i + 2 * ev) %% 4
actors <- sienaNodeSet(50, nodeSetName = "actors")
events <- sienaNodeSet(12, nodeSetName = "events")
dt <- sienaDataCreate(
    aff = sienaDependent(tm, type = "bipartite", nodeSet = c("actors", "events")),
    alcohol = sienaDependent(s50a, type = "behavior", nodeSet = "actors"),
    smoke1 = coCovar(s50s[, 1], nodeSet = "actors"),
    dce = coDyadCovar(dce, nodeSets = c("actors", "events"), type = "bipartite"),
    nodeSets = list(actors, events))
af <- function(sn, i1 = "", parm = "") sprintf("aff|%s|%s|%s", sn, i1, parm)
twomode <- c(
    af("density"), af("cycle4"), af("inPop"), af("inPopSqrt"), af("outAct"),
    af("outActSqrt"), af("outTrunc"), af("outTrunc", parm = "3"), af("outIso"),
    af("egoX", "smoke1"), af("egoSqX", "smoke1"), af("egoX", "alcohol"),
    af("egoSqX", "alcohol"), af("X", "dce"),
    af("outRate"), af("outRateLog"), af("outRateInv"), af("RateX", "smoke1"))
twomode_values <- sapply(twomode, function(s) target_of(dt, s))
tt <- RSiena:::getTargets(dt, getEffects(dt))
twomode_rates <- c(tt[1, 1], tt[2, 2])

## ---- user-defined interactions (includeInteraction) --------------------------
# Spec: "data|variable|shortName:interaction1+shortName:interaction1[+...]" with
# data one of directed / undirected / twomode (the three data sets above).
interaction_target <- function(spec) {
    p <- strsplit(spec, "|", fixed = TRUE)[[1]]
    d <- switch(p[1], directed = dd, undirected = du, twomode = dt)
    parts <- strsplit(strsplit(p[3], "+", fixed = TRUE)[[1]], ":", fixed = TRUE)
    sn <- sapply(parts, function(x) x[1])
    i1 <- sapply(parts, function(x) if (length(x) > 1) x[2] else "")
    e <- getEffects(d)
    e$include <- e$shortName == "Rate" & e$type == "rate"
    e$include[e$shortName == "density" & e$type == "eval" & e$name == p[2]] <- TRUE
    args <- c(list(e), lapply(sn, as.name),
              list(name = p[2], interaction1 = i1, character = FALSE, verbose = FALSE))
    e <- do.call(includeInteraction, args)
    # getTargets() does not set up interactions; a siena07 run without
    # estimation (nsub = 0) does, and returns the targets it computed.
    alg <- NULL
    invisible(capture.output(
        alg <- sienaAlgorithmCreate(projname = NULL, nsub = 0, n3 = 10, cond = FALSE,
                                    seed = seed), type = "output"))
    ans <- NULL
    invisible(capture.output(
        ans <- siena07(alg, data = d, effects = e, batch = TRUE, verbose = FALSE,
                       silent = TRUE, useCluster = FALSE), type = "output"))
    id <- which(ans$requestedEffects$shortName %in% c("unspInt", "behUnspInt") &
                ans$requestedEffects$name == p[2])
    stopifnot(length(id) == 1)
    ans$targets[id]
}
interactions <- c(
    "directed|friendship|egoX:smoke1+recip",
    "directed|friendship|egoX:smoke1+transTrip",
    "directed|friendship|egoX:smoke1+cycle3",
    "directed|friendship|egoX:smoke1+altX:smoke1",
    "directed|friendship|egoX:alcohol+simX:alcohol",
    "directed|friendship|recip+simX:smoke1",
    "directed|friendship|recip+gwespFF",
    "directed|friendship|recip+X:dc",
    "directed|friendship|inAct+recip",
    "directed|friendship|inActSqrt+altX:smoke1",
    "directed|friendship|egoX:smoke1+outAct",
    "directed|friendship|density+inPop",
    "directed|friendship|outPop+recip",
    "directed|friendship|egoX:smoke1+altX:smoke1+recip",
    "directed|friendship|egoX:smoke1+egoSqX:smoke1+transTrip",
    "directed|friendship|recip+simX:smoke1+sameX:alcohol",
    "undirected|friendship|density+transTriads",
    "undirected|friendship|egoX:smoke1+simX:alcohol",
    "undirected|friendship|altX:smoke1+gwesp",
    "twomode|aff|egoX:smoke1+inPop",
    "twomode|aff|egoX:smoke1+cycle4",
    "twomode|aff|density+X:dce",
    "directed|alcohol|avAlt:friendship+effFrom:smoke1",
    "directed|alcohol|linear+avSim:friendship",
    "directed|alcohol|quad+indeg:friendship",
    "directed|alcohol|avAlt:friendship+outdeg:friendship+effFrom:smoke1")
interaction_values <- sapply(interactions, interaction_target)

cat("# Generated by RSiena, not transcribed from Julia output.\n[provenance]\n")
cat(sprintf('r_version = "%s"\nrsiena_version = "%s"\n', getRversion(), packageVersion("RSiena")))
cat(sprintf('seed = %d\n', seed))
cat('script = "test/fixtures/r/s50_targets.R"\n')
cat(sprintf('date = "%s"\n', Sys.Date()))
cat('dataset = "RSiena::s50: friendship waves (directed, and symmetrised), alcohol, centred smoke1; deterministic dc/fr2"\n')
cat('method = "RSiena:::getTargets; unconditional MoM; sum across periods"\n')
cat('\n[tolerance]\n# Deterministic targets; tolerance covers floating-point summation only.\n')
cat('targets = 1e-9\ndirected_targets = 1e-9\nundirected_targets = 1e-9\nundirected_rates = 1e-9\n')
cat('twomode_targets = 1e-9\ntwomode_rates = 1e-9\ninteraction_targets = 1e-9\n')
cat('\n[values]\n')
cat(sprintf('targets = [%s]\n', paste(sprintf("%.17g", values), collapse = ", ")))
cat(sprintf('effect_names = [%s]\n', paste(sprintf('"%s"', e$effectName[indices]), collapse = ", ")))
emit("directed", directed, directed_values)
emit("undirected", undirected, undirected_values)
cat(sprintf('undirected_rates = [%s]\n', paste(sprintf("%.17g", undirected_rates), collapse = ", ")))
emit("twomode", unname(twomode), twomode_values)
cat(sprintf('twomode_rates = [%s]\n', paste(sprintf("%.17g", twomode_rates), collapse = ", ")))
emit("interaction", unname(interactions), interaction_values)
cat('\n[coverage]\n')
cat(sprintf('scope = "every effect offered under an RSiena short name: %d directed, %d undirected and %d two-mode targets plus the original 34"\n',
            length(directed), length(undirected), length(twomode)))
cat('not_equivalent = ["balanceSimple", "avAttHigherSimple", "avAttLowerSimple"]\n')
cat('not_implemented = ["initiative/pairwise network model types"]\n')
