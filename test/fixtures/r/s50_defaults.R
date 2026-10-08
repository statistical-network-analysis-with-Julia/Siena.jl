#!/usr/bin/env Rscript
# Golden fixture: RSiena's DEFAULT MODEL (what getEffects() includes) and its
# up-only/down-only restriction (sienaDependent(allowOnly = TRUE), the default).
#
# Regenerate from the package root:
#
#   Rscript test/fixtures/r/s50_defaults.R > test/fixtures/s50_defaults.toml
#
# Part 1 (deterministic). For each data set below, the effects getEffects() marks
# `include = TRUE`, as "variable|shortName|interaction1|type"; a basic rate is
# "variable|Rate|<period>|rate". RSiena's rules (getEffects, RSiena 1.6.6):
#   * every network: the basic rate of every period, and `density` unless every
#     period only adds ties or every period only drops ties (then `density` is not
#     offered at all);
#   * a directed one-mode network: also `recip`;
#   * a behaviour: the basic rates, `linear` unless every period only goes up or
#     every period only goes down (then `linear` is not offered), and `quad` when
#     the observed range (max - min over all waves) is at least 2.
#
# Part 2 (simulated). In a period in which the observed variable only increases
# (or only decreases), RSiena lets actors make only such changes. Two models are
# simulated at fixed parameters (simOnly, unconditional) on data with one up-only
# or down-only period, and the mean and sd of every simulated statistic are frozen,
# as in s50_dynamics.R.
#
# Part 3 (deterministic). Every effect short name in RSiena's effect table
# (allEffects), so that the test suite can check that the effects it treats as
# Siena.jl's own carry no RSiena name.
suppressMessages(library(RSiena))
seed <- 20261006L
n3 <- 1000L

sym <- function(m) { m[m > 1] <- 0; pmax(m, t(m)) }
bin <- function(m) { m[m > 1] <- 0; m }
w1 <- bin(s501); w2 <- bin(s502); w3 <- bin(s503)
or <- function(a, b) pmax(a, b)
tm <- array(0, dim = c(50, 12, 3))
for (w in 1:3) for (i in 1:50) for (ev in 1:12)
    if (((i * 5 + ev * 7 + w * (i %% 3 + ev)) %% 9) < 2) tm[i, ev, w] <- 1
tmc <- tm                                   # cumulative two-mode panel: up-only
tmc[, , 2] <- pmax(tm[, , 1], tm[, , 2]); tmc[, , 3] <- pmax(tmc[, , 2], tm[, , 3])
dce <- matrix(0, 50, 12)
for (i in 1:50) for (ev in 1:12) dce[i, ev] <- (i + 2 * ev) %% 4
cum <- array(c(w1, or(w1, w2), or(or(w1, w2), w3)), dim = c(50, 50, 3))
alc <- s50a
alc_up <- cbind(alc[, 1], pmax(alc[, 1], alc[, 2]), pmax(alc[, 1], alc[, 2], alc[, 3]))
alc_bin <- (alc >= 3) * 1
fr <- array(c(w1, w2, w3), dim = c(50, 50, 3))
actors <- sienaNodeSet(50, nodeSetName = "actors")
events <- sienaNodeSet(12, nodeSetName = "events")

datasets <- list(
    directed = sienaDataCreate(friendship = sienaDependent(fr),
                               alcohol = sienaDependent(alc, type = "behavior"),
                               smoke1 = coCovar(s50s[, 1])),
    undirected = sienaDataCreate(
        friendship = sienaDependent(array(c(sym(w1), sym(w2), sym(w3)), dim = c(50, 50, 3))),
        alcohol = sienaDependent(alc, type = "behavior"),
        smoke1 = coCovar(s50s[, 1])),
    twomode = sienaDataCreate(
        aff = sienaDependent(tm, type = "bipartite", nodeSet = c("actors", "events")),
        alcohol = sienaDependent(alc, type = "behavior", nodeSet = "actors"),
        smoke1 = coCovar(s50s[, 1], nodeSet = "actors"),
        dce = coDyadCovar(dce, nodeSets = c("actors", "events"), type = "bipartite"),
        nodeSets = list(actors, events)),
    multiplex = sienaDataCreate(friendship = sienaDependent(fr),
                                advice = sienaDependent(array(c(w2, w3, w1), dim = c(50, 50, 3))),
                                alcohol = sienaDependent(alc, type = "behavior")),
    binary = sienaDataCreate(friendship = sienaDependent(fr),
                             drinker = sienaDependent(alc_bin, type = "behavior")),
    uponly = sienaDataCreate(friendship = sienaDependent(cum),
                             alcohol = sienaDependent(alc_up, type = "behavior")),
    downonly = sienaDataCreate(friendship = sienaDependent(cum[, , 3:1]),
                               alcohol = sienaDependent(alc_up[, 3:1], type = "behavior")),
    uponly_undirected = sienaDataCreate(friendship = sienaDependent(
        array(c(sym(cum[, , 1]), sym(cum[, , 2]), sym(cum[, , 3])), dim = c(50, 50, 3)))),
    uponly_twomode = sienaDataCreate(
        aff = sienaDependent(tmc, type = "bipartite", nodeSet = c("actors", "events")),
        nodeSets = list(actors, events)),
    mixed = sienaDataCreate(friendship = sienaDependent(array(c(w1, or(w1, w2), w3),
                                                              dim = c(50, 50, 3))),
                            alcohol = sienaDependent(cbind(alc[, 1], pmax(alc[, 1], alc[, 2]),
                                                           alc[, 3]), type = "behavior")))

included <- function(d) {
    e <- getEffects(d)
    e <- e[e$include, ]
    out <- character(nrow(e))
    for (k in seq_len(nrow(e))) {
        if (e$shortName[k] == "Rate" && e$type[k] == "rate") {
            period <- sum(e$name[1:k] == e$name[k] & e$shortName[1:k] == "Rate" &
                          e$type[1:k] == "rate")
            out[k] <- sprintf("%s|Rate|%d|rate", e$name[k], period)
        } else {
            out[k] <- sprintf("%s|%s|%s|%s", e$name[k], e$shortName[k],
                              e$interaction1[k], e$type[k])
        }
    }
    sort(out)
}

## ---- part 2: the up-only / down-only restriction --------------------------------
run_model <- function(d, entries) {
    e <- getEffects(d)
    e$include <- FALSE
    pos <- integer(length(entries))
    for (k in seq_along(entries)) {
        p <- strsplit(entries[k], "|", fixed = TRUE)[[1]]
        if (p[2] == "Rate") {
            id <- which(e$name == p[1] & e$shortName == "Rate" & e$type == "rate")[as.integer(p[4])]
        } else {
            id <- which(e$name == p[1] & e$shortName == p[2] & e$interaction1 == p[3] &
                        e$type == p[5])
        }
        stopifnot(length(id) == 1)
        e$include[id] <- TRUE
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
    stopifnot(!anyNA(col))
    list(mean = as.numeric(ans$targets[col] + colMeans(ans$sf)[col]),
         sd = as.numeric(apply(ans$sf, 2, sd)[col]))
}

# Period 1 of `mixed` only adds ties and only raises alcohol; period 2 does both.
# Period 1 of `mixed_down` only drops ties and only lowers alcohol.
mixed_down <- sienaDataCreate(
    friendship = sienaDependent(array(c(or(w1, w2), w2, w3), dim = c(50, 50, 3))),
    alcohol = sienaDependent(cbind(pmax(alc[, 1], alc[, 2]), alc[, 2], alc[, 3]),
                             type = "behavior"))
stopifnot(attr(datasets$mixed$depvars$friendship, "uponly")[1],
          !attr(datasets$mixed$depvars$friendship, "uponly")[2],
          !attr(datasets$mixed$depvars$friendship, "downonly")[2],
          attr(datasets$mixed$depvars$alcohol, "uponly")[1],
          !attr(datasets$mixed$depvars$alcohol, "uponly")[2],
          attr(mixed_down$depvars$friendship, "downonly")[1],
          attr(mixed_down$depvars$alcohol, "downonly")[1])
model <- c("friendship|Rate||1|rate|5", "friendship|Rate||2|rate|5",
           "friendship|density|||eval|-1.8", "friendship|recip|||eval|1.5",
           "friendship|transTrip|||eval|0.2",
           "alcohol|Rate||1|rate|1.5", "alcohol|Rate||2|rate|1.5",
           "alcohol|linear|||eval|0.2", "alcohol|quad|||eval|-0.1",
           "alcohol|avAlt|friendship||eval|0.5")
up <- run_model(datasets$mixed, model)
down <- run_model(mixed_down, model)

## ---- output ------------------------------------------------------------------------
q <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
cat("# Generated by RSiena, not transcribed from Julia output.\n[provenance]\n")
cat(sprintf('r_version = "%s"\nrsiena_version = "%s"\n', getRversion(), packageVersion("RSiena")))
cat(sprintf('seed = %d\n', seed))
cat('script = "test/fixtures/r/s50_defaults.R"\n')
cat(sprintf('date = "%s"\n', Sys.Date()))
cat('dataset = "RSiena::s50 friendship (ties > 1 set to 0), alcohol, smoke1; symmetrised, cumulative (up-only), reversed (down-only) and dichotomised variants; the deterministic 50 x 12 two-mode panel of s50_targets.R"\n')
cat(sprintf('method = "part 1: getEffects()$include; part 2: siena07 simOnly, unconditional, n3 = %d, allowOnly = TRUE (default); part 3: unique(allEffects$shortName)"\n', n3))
cat('\n[tolerance]\n')
cat('# Part 1 is exact (set equality). Part 2: |mean_julia - mean_rsiena| within\n')
cat('# 4.5 combined Monte-Carlo standard errors, as for s50_dynamics.toml.\n')
cat('z = 4.5\n')
cat('\n[values]\n')
cat(sprintf('cases = [%s]\n', q(names(datasets))))
for (nm in names(datasets)) cat(sprintf('%s_included = [%s]\n', nm, q(included(datasets[[nm]]))))
cat(sprintf('restricted_model = [%s]\n', q(model)))
cat(sprintf('uponly_mean = [%s]\nuponly_sd = [%s]\n', num(up$mean), num(up$sd)))
cat(sprintf('downonly_mean = [%s]\ndownonly_sd = [%s]\n', num(down$mean), num(down$sd)))
cat(sprintf('n3 = %d\n', n3))
allEffects <- get(data("allEffects", package = "RSiena", envir = environment()))
cat(sprintf('rsiena_short_names = [%s]\n', q(sort(unique(allEffects$shortName)))))
