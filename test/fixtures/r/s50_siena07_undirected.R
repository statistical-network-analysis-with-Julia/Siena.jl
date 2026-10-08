#!/usr/bin/env Rscript
# Golden fixture: RSiena siena07 fit of an UNDIRECTED (symmetric) network.
#
# Regenerate from the package root (six RSiena fits):
#
#   Rscript test/fixtures/r/s50_siena07_undirected.R > test/fixtures/s50_siena07_undirected.toml
#
# Data: the s50 friendship waves symmetrised (a tie in either direction is an
# edge). RSiena treats a symmetric dependent network with model type 2 by default
# (unilateral initiative and reciprocal confirmation, the "forcing" model), and for
# one dependent variable estimates conditionally (cond = NA). Model: degree
# (density), transitive triads, smoke1 similarity. (Adding sqrt degree of alter
# leaves RSiena itself unconverged, tconv.max 0.36-0.50, so it is not used.)
#
# The reference values are the MEAN over six RSiena fits with the seed-to-seed sd
# frozen beside them (see s50_siena07_cond.R for the reasoning).
suppressMessages(library(RSiena))
seed <- 20261003L
rep_seeds <- c(111L, 222L, 333L, 444L, 555L)

sym <- function(m) { m[m > 1] <- 0; pmax(m, t(m)) }
friendship <- sienaDependent(array(c(sym(s501), sym(s502), sym(s503)),
                                   dim = c(50, 50, 3)))
smoke1 <- coCovar(s50s[, 1])
dat <- sienaDataCreate(friendship, smoke1)
stopifnot(attr(dat$depvars$friendship, "symmetric"))
eff <- getEffects(dat)
eff <- includeEffects(eff, transTriads, name = "friendship", verbose = FALSE)
eff <- includeEffects(eff, simX, interaction1 = "smoke1", name = "friendship",
                      verbose = FALSE)

fit_once <- function(s) {
  alg <- NULL
  invisible(capture.output(
    alg <- sienaAlgorithmCreate(projname = NULL, seed = s, nsub = 4, n3 = 1000),
    type = "output"))
  ans <- NULL
  invisible(capture.output(
    ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, verbose = FALSE,
                   silent = TRUE, useCluster = FALSE, returnDeps = FALSE),
    type = "output"))
  ans
}

fits <- lapply(c(seed, rep_seeds), fit_once)
stopifnot(all(fits[[1]]$modelType == 2), fits[[1]]$cconditional)
theta <- t(sapply(fits, function(a) a$theta))
se <- t(sapply(fits, function(a) sqrt(diag(a$covtheta))))
rate <- t(sapply(fits, function(a) a$rate))
vrate <- t(sapply(fits, function(a) a$vrate))
tconv <- sapply(fits, function(a) a$tconv.max)
stopifnot(all(tconv < 0.25))
num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")

cat('name = "s50_siena07_undirected"\n\n[provenance]\n')
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('rsiena_version = "%s"\n', as.character(packageVersion("RSiena"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/s50_siena07_undirected.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "RSiena::s50 friendship, symmetrised; smoke1"\n')
cat('model = "undirected, modelType 2 (forcing), conditional MoM; density, transTriads, smoke1 similarity"\n')
cat('algorithm = "sienaAlgorithmCreate(nsub=4, n3=1000) (cond = NA -> conditional)"\n')
cat(sprintf('replication_seeds = "%s"\n\n', paste(rep_seeds, collapse = ",")))
cat("[tolerance]\n# See the Julia testset: tolerances come from the measured widths.\ndefault = 0.0\n\n")
cat("[values]\n")
cat(sprintf("effect_names = [%s]\n",
            paste(sprintf('"%s"', fits[[1]]$effects$effectName), collapse = ", ")))
cat(sprintf("coefficients = [%s]\n", num(colMeans(theta))))
cat(sprintf("coefficients_seed_sd = [%s]\n", num(apply(theta, 2, sd))))
cat(sprintf("std_errors = [%s]\n", num(colMeans(se))))
cat(sprintf("std_errors_seed_sd = [%s]\n", num(apply(se, 2, sd))))
cat(sprintf("rates = [%s]\n", num(colMeans(rate))))
cat(sprintf("rates_seed_sd = [%s]\n", num(apply(rate, 2, sd))))
cat(sprintf("rate_std_errors = [%s]\n", num(colMeans(vrate))))
cat(sprintf("tconv_max = [%s]\n", num(tconv)))
