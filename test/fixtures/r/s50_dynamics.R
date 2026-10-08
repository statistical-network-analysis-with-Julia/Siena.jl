#!/usr/bin/env Rscript
# Golden fixture: RSiena SIMULATED DYNAMICS at fixed parameters.
#
# Regenerate from the package root (about fifty short RSiena simulation runs):
#
#   Rscript test/fixtures/r/s50_dynamics.R > test/fixtures/s50_dynamics.toml
#
# WHY. Target statistics pin an effect's STATISTIC. They cannot pin its CHANGE
# STATISTIC (what the effect contributes to an actor's choice in a ministep): two
# implementations with the same statistic can drive different dynamics -- RSiena's
# GWESP effects are "elementary" (the change statistic is not the difference of the
# statistic), which target tests could not see. So every effect offered under an
# RSiena short name is also simulated here: RSiena simulates the model at a FIXED
# parameter vector with the effect switched on (simOnly, no estimation), and the
# mean of every simulated statistic is frozen with its simulation sd. The Julia test
# simulates the same model at the same parameters and must reproduce the means
# within Monte-Carlo error. A wrong change statistic shifts the whole simulated
# distribution, so it shows up in several statistics at once.
#
# Each model is a base (basic rates; density and, for directed networks,
# reciprocity; linear and quadratic shape for a behaviour) plus a few effects with
# a non-zero parameter. Entry format:
#   "variable|shortName|interaction1|parm|type|theta"       (parm empty = default)
#   "variable|Rate||<period>|rate|theta"                    (basic rate of a period)
#   "variable|a:i1+b:i1[+c:i1]|INT||eval|theta"             (includeInteraction)
suppressMessages(library(RSiena))
seed <- 20261007L
n3 <- 1000L

sym <- function(m) { m[m > 1] <- 0; pmax(m, t(m)) }
dc <- matrix(0, 50, 50)
for (i in 1:50) for (j in 1:50) dc[i, j] <- ((i * 7 + j * 3) %% 5)
tm <- array(0, dim = c(50, 12, 3))
for (w in 1:3) for (i in 1:50) for (ev in 1:12)
    if (((i * 5 + ev * 7 + w * (i %% 3 + ev)) %% 9) < 2) tm[i, ev, w] <- 1
dce <- matrix(0, 50, 12)
for (i in 1:50) for (ev in 1:12) dce[i, ev] <- (i + 2 * ev) %% 4
fr <- array(c(s501, s502, s503), dim = c(50, 50, 3))
actors <- sienaNodeSet(50, nodeSetName = "actors")
events <- sienaNodeSet(12, nodeSetName = "events")
datasets <- list(
    net = sienaDataCreate(friendship = sienaDependent(fr), smoke1 = coCovar(s50s[, 1]),
                          dc = coDyadCovar(dc)),
    coevo = sienaDataCreate(friendship = sienaDependent(fr),
                            alcohol = sienaDependent(s50a, type = "behavior"),
                            smoke1 = coCovar(s50s[, 1])),
    multi = sienaDataCreate(friendship = sienaDependent(fr),
                            advice = sienaDependent(array(c(s502, s503, s501),
                                                          dim = c(50, 50, 3)))),
    und = sienaDataCreate(friendship = sienaDependent(array(c(sym(s501), sym(s502), sym(s503)),
                                                            dim = c(50, 50, 3))),
                          smoke1 = coCovar(s50s[, 1]), dc = coDyadCovar(dc + t(dc))),
    twomode = sienaDataCreate(
        aff = sienaDependent(tm, type = "bipartite", nodeSet = c("actors", "events")),
        smoke1 = coCovar(s50s[, 1], nodeSet = "actors"),
        dce = coDyadCovar(dce, nodeSets = c("actors", "events"), type = "bipartite"),
        nodeSets = list(actors, events)))

fr_base <- c("friendship|Rate||1|rate|6", "friendship|Rate||2|rate|5",
             "friendship|density|||eval|-2.2", "friendship|recip|||eval|2")
base <- list(
    net = fr_base,
    coevo = c(fr_base, "alcohol|Rate||1|rate|1.3", "alcohol|Rate||2|rate|1.7",
              "alcohol|linear|||eval|0.3", "alcohol|quad|||eval|-0.05"),
    multi = c(fr_base, "advice|Rate||1|rate|5", "advice|Rate||2|rate|6",
              "advice|density|||eval|-2.2", "advice|recip|||eval|1.8"),
    und = c("friendship|Rate||1|rate|2.4", "friendship|Rate||2|rate|2",
            "friendship|density|||eval|-1.9"),
    twomode = c("aff|Rate||1|rate|6", "aff|Rate||2|rate|6", "aff|density|||eval|-1.2"))

f <- function(sn, theta, i1 = "", parm = "", type = "eval", var = "friendship")
    sprintf("%s|%s|%s|%s|%s|%s", var, sn, i1, parm, type, theta)
a <- function(sn, theta, i1 = "friendship", parm = "", type = "eval")
    f(sn, theta, i1, parm, type, var = "alcohol")
int <- function(parts, theta, var = "friendship") sprintf("%s|%s|INT||eval|%s", var, parts, theta)
models <- list(
    n1 = list("net", c(f("transTrip", 0.3), f("cycle3", -0.3), f("transTies", 0.4))),
    n2 = list("net", c(f("transMedTrip", 0.2), f("transRecTrip", 0.3), f("between", -0.1))),
    n3 = list("net", c(f("nbrDist2", -0.1), f("denseTriads", 0.3), f("gwdspFF", -0.1))),
    n4 = list("net", c(f("gwespFF", 0.8), f("gwespBB", -0.4), f("gwespFB", 0.4))),
    n5 = list("net", c(f("inPop", 0.08), f("outPop", -0.15), f("inAct", 0.08), f("outAct", -0.05))),
    n6 = list("net", c(f("inPopSqrt", 0.3), f("outPopSqrt", -0.3), f("inActSqrt", 0.2),
                       f("outActSqrt", -0.2))),
    n7 = list("net", c(f("outTrunc", 0.6, parm = "3"), f("isolateNet", 1), f("outIso", 0.8))),
    n8 = list("net", c(f("egoX", 0.3, "smoke1"), f("altX", 0.3, "smoke1"),
                       f("simX", 0.8, "smoke1"), f("sameX", 0.3, "smoke1"))),
    n9 = list("net", c(f("egoSqX", 0.2, "smoke1"), f("altSqX", -0.2, "smoke1"),
                       f("diffX", 0.2, "smoke1"), f("diffSqX", -0.15, "smoke1"))),
    n10 = list("net", c(f("absDiffX", -0.3, "smoke1"), f("higher", 0.4, "smoke1"),
                        f("egoXaltX", 0.3, "smoke1"), f("egoPlusAltX", 0.15, "smoke1"))),
    n11 = list("net", c(f("sameXRecip", 0.5, "smoke1"), f("simRecipX", 0.8, "smoke1"),
                        f("simXTransTrip", 0.6, "smoke1"), f("X", 0.15, "dc"))),
    r1 = list("net", c(f("outRate", 0.1, type = "rate"), f("inRate", 0.1, type = "rate"))),
    r2 = list("net", c(f("recipRate", 0.15, type = "rate"),
                       f("RateX", 0.4, "smoke1", type = "rate"))),
    r3 = list("net", c(f("outRateLog", 0.5, type = "rate"), f("inRateLog", 0.4, type = "rate"))),
    r4 = list("net", c(f("outRateInv", -0.8, type = "rate"), f("inRateInv", 0.8, type = "rate"))),
    i1 = list("net", c(f("egoX", 0.2, "smoke1"), f("transTrip", 0.3),
                       int("egoX:smoke1+recip", -0.4), int("egoX:smoke1+transTrip", 0.1))),
    i2 = list("net", c(f("simX", 0.5, "smoke1"), int("recip+simX:smoke1", 0.5),
                       int("egoX:smoke1+altX:smoke1+recip", 0.3),
                       int("recip+gwespFF", 0.3))),
    c1 = list("coevo", c(a("avAlt", 0.6), a("indeg", 0.1), a("outdeg", -0.1),
                         f("egoX", 0.1, "alcohol"), f("altX", 0.1, "alcohol"),
                         f("simX", 1, "alcohol"))),
    c2 = list("coevo", c(a("avSim", 3), a("totAlt", 0.1), f("sameX", 0.3, "alcohol"),
                         f("higher", 0.3, "alcohol"), f("diffX", -0.1, "alcohol"))),
    c3 = list("coevo", c(a("totSim", 0.8), a("avInAlt", 0.5), a("recipDeg", 0.15),
                         f("egoXaltX", 0.1, "alcohol"), f("absDiffX", -0.2, "alcohol"))),
    c4 = list("coevo", c(a("avRecAlt", 0.5), a("totInAlt", 0.1), a("avAltDist2", 0.8),
                         a("effFrom", 0.3, "smoke1"))),
    c5 = list("coevo", c(a("threshold", 0.5, "", parm = "3"), a("outRate", 0.1, type = "rate"),
                         a("inRate", 0.1, type = "rate"))),
    c6 = list("coevo", c(a("recipRate", 0.15, type = "rate"),
                         a("RateX", 0.4, "smoke1", type = "rate"), a("avAlt", 0.4),
                         int("avAlt:friendship+effFrom:smoke1", 0.3, var = "alcohol"),
                         int("linear+avSim:friendship", 1, var = "alcohol"))),
    m1 = list("multi", c(f("crprod", 0.8, "advice"), f("crprodRecip", 0.5, "advice"))),
    u1 = list("und", c(f("transTriads", 0.5), f("transTies", 0.3), f("between", -0.05))),
    u2 = list("und", c(f("nbrDist2", -0.05), f("gwesp", 0.6), f("degPlus", 0.03))),
    u3 = list("und", c(f("inPop", 0.05), f("outActSqrt", 0.1))),
    u4 = list("und", c(f("inPopSqrt", 0.2), f("outAct", 0.03))),
    u5 = list("und", c(f("outTrunc", 0.5, parm = "3"), f("isolateNet", 1), f("outIso", 0.5))),
    u6 = list("und", c(f("egoX", 0.2, "smoke1"), f("altX", 0.2, "smoke1"),
                       f("simX", 0.6, "smoke1"), f("sameX", 0.3, "smoke1"))),
    u7 = list("und", c(f("egoSqX", 0.1, "smoke1"), f("altSqX", 0.1, "smoke1"),
                       f("diffX", 0.1, "smoke1"), f("diffSqX", -0.1, "smoke1"),
                       f("absDiffX", -0.2, "smoke1"))),
    u8 = list("und", c(f("higher", 0.3, "smoke1"), f("egoXaltX", 0.2, "smoke1"),
                       f("egoPlusAltX", 0.1, "smoke1"), f("X", 0.1, "dc"),
                       int("egoX:smoke1+simX:smoke1", 0.3))),
    u9 = list("und", c(f("outRate", 0.1, type = "rate"), f("RateX", 0.3, "smoke1", type = "rate"))),
    u10 = list("und", c(f("outRateLog", 0.4, type = "rate"), f("outRateInv", -0.5, type = "rate"))),
    t1 = list("twomode", c(f("cycle4", 0.03, var = "aff"), f("inPop", 0.05, var = "aff"),
                           f("outAct", -0.05, var = "aff"))),
    t2 = list("twomode", c(f("inPopSqrt", 0.2, var = "aff"), f("outActSqrt", -0.2, var = "aff"),
                           f("outTrunc", 0.5, parm = "3", var = "aff"),
                           f("outIso", 0.5, var = "aff"))),
    t3 = list("twomode", c(f("egoX", 0.3, "smoke1", var = "aff"),
                           f("egoSqX", 0.2, "smoke1", var = "aff"), f("X", 0.2, "dce", var = "aff"),
                           f("outRate", 0.1, type = "rate", var = "aff"),
                           f("RateX", 0.3, "smoke1", type = "rate", var = "aff"))),
    t4 = list("twomode", c(f("outRateLog", 0.4, type = "rate", var = "aff"),
                           f("outRateInv", -0.5, type = "rate", var = "aff"),
                           f("egoX", 0.2, "smoke1", var = "aff"),
                           int("egoX:smoke1+inPop", 0.05, var = "aff"))))

run_model <- function(dname, entries) {
    d <- datasets[[dname]]
    e <- getEffects(d)
    e$include <- FALSE
    pos <- integer(length(entries))        # row of each entry in `e`
    ints <- which(sapply(strsplit(entries, "|", fixed = TRUE), function(p) p[3] == "INT"))
    for (k in setdiff(seq_along(entries), ints)) {
        p <- strsplit(entries[k], "|", fixed = TRUE)[[1]]
        if (p[2] == "Rate") {
            id <- which(e$name == p[1] & e$shortName == "Rate" & e$type == "rate")[as.integer(p[4])]
        } else {
            id <- which(e$name == p[1] & e$shortName == p[2] & e$interaction1 == p[3] &
                        e$type == p[5])
            stopifnot(length(id) == 1)
            if (p[4] != "") e$parm[id] <- as.numeric(p[4])
        }
        e$include[id] <- TRUE
        e$initialValue[id] <- as.numeric(p[6])
        pos[k] <- e$effectNumber[id]
    }
    for (k in ints) {
        p <- strsplit(entries[k], "|", fixed = TRUE)[[1]]
        parts <- strsplit(strsplit(p[2], "+", fixed = TRUE)[[1]], ":", fixed = TRUE)
        sn <- sapply(parts, function(x) x[1])
        i1 <- sapply(parts, function(x) if (length(x) > 1) x[2] else "")
        before <- e$effectNumber[e$include]
        e <- do.call(includeInteraction, c(list(e), lapply(sn, as.name),
            list(name = p[1], interaction1 = i1, character = FALSE, verbose = FALSE)))
        id <- which(e$include & !(e$effectNumber %in% before) &
                    e$shortName %in% c("unspInt", "behUnspInt"))
        stopifnot(length(id) == 1)
        e$initialValue[id] <- as.numeric(p[6])
        pos[k] <- e$effectNumber[id]
    }
    alg <- NULL
    invisible(capture.output(
        alg <- sienaAlgorithmCreate(projname = NULL, nsub = 0, n3 = n3, simOnly = TRUE,
                                    cond = FALSE, seed = seed), type = "output"))
    ans <- NULL
    invisible(capture.output(
        ans <- siena07(alg, data = d, effects = e, batch = TRUE, verbose = FALSE,
                       silent = TRUE, useCluster = FALSE), type = "output"))
    col <- match(pos, ans$requestedEffects$effectNumber)
    stopifnot(!anyNA(col),
              isTRUE(all.equal(as.numeric(ans$theta[col]),
                               as.numeric(sapply(strsplit(entries, "|", fixed = TRUE),
                                                 function(p) p[6])))))
    list(mean = as.numeric(ans$targets[col] + colMeans(ans$sf)[col]),
         sd = as.numeric(apply(ans$sf, 2, sd)[col]))
}

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
out <- character()
for (key in names(models)) {
    entries <- c(base[[models[[key]][[1]]]], models[[key]][[2]])
    r <- run_model(models[[key]][[1]], entries)
    out <- c(out,
        sprintf('%s_data = "%s"', key, models[[key]][[1]]),
        sprintf('%s_entries = [%s]', key, paste(sprintf('"%s"', entries), collapse = ", ")),
        sprintf('%s_mean = [%s]', key, num(r$mean)),
        sprintf('%s_sd = [%s]', key, num(r$sd)))
}

cat('name = "s50_dynamics"\n\n[provenance]\n')
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('rsiena_version = "%s"\n', as.character(packageVersion("RSiena"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/s50_dynamics.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "RSiena::s50 (friendship, alcohol, smoke1), symmetrised s50, deterministic two-mode panel and dyadic covariates"\n')
cat('method = "siena07(simOnly=TRUE, nsub=0, cond=FALSE): unconditional simulation at fixed theta; mean and sd of the simulated statistics"\n')
cat("\n[tolerance]\n# Monte-Carlo comparison; see the Julia testset (z-test on the two means).\ndefault = 0.0\n")
cat("\n[values]\n")
cat(sprintf("n3 = %d\n", n3))
cat(sprintf("models = [%s]\n", paste(sprintf('"%s"', names(models)), collapse = ", ")))
cat(paste(out, collapse = "\n"), "\n")
