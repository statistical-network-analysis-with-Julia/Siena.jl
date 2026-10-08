#!/usr/bin/env Rscript
# Golden fixture: RSiena siena07 fit with an ELEMENTARY effect (gwespFF) and a
# user-defined INTERACTION (includeInteraction), on the bundled s50 data.
#
# Regenerate from the package root (six RSiena fits, a few minutes):
#
#   Rscript test/fixtures/r/s50_siena07_interaction.R > test/fixtures/s50_siena07_interaction.toml
#
# Model: friendship ~ density + reciprocity + gwespFF + smoke1 ego + smoke1 alter
# + (smoke1 ego x reciprocity), conditional (RSiena's default for one network).
# Target statistics cannot tell an elementary effect from a regular one (the
# statistic is the same); only the fitted dynamics can. GWESP's change statistic
# in RSiena is the weight of the toggled tie alone, and an interaction's is the
# product of its components' change statistics.
#
# The reference values are the MEAN over six RSiena fits with the seed-to-seed sd
# frozen beside them (see s50_siena07_cond.R for the reasoning).
suppressMessages(library(RSiena))
seed <- 20261005L
rep_seeds <- c(101L, 202L, 303L, 404L, 505L)

friendship <- sienaDependent(array(c(s501, s502, s503), dim = c(50, 50, 3)))
smoke1 <- coCovar(s50s[, 1])
dat <- sienaDataCreate(friendship, smoke1)
eff <- getEffects(dat)
eff <- includeEffects(eff, gwespFF, name = "friendship", verbose = FALSE)
eff <- includeEffects(eff, egoX, altX, interaction1 = "smoke1",
                      name = "friendship", verbose = FALSE)
eff <- includeInteraction(eff, egoX, recip, interaction1 = c("smoke1", ""),
                          name = "friendship", verbose = FALSE)

fit_once <- function(s) {
  alg <- NULL
  invisible(capture.output(
    alg <- sienaAlgorithmCreate(projname = NULL, seed = s, cond = TRUE,
                                nsub = 4, n3 = 1000),
    type = "output"))
  ans <- NULL
  invisible(capture.output(
    ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, verbose = FALSE,
                   silent = TRUE, useCluster = FALSE, returnDeps = FALSE),
    type = "output"))
  ans
}

fits <- lapply(c(seed, rep_seeds), fit_once)
est <- rep(TRUE, length(fits[[1]]$theta))
theta <- t(sapply(fits, function(a) a$theta[est]))
se <- t(sapply(fits, function(a) sqrt(diag(a$covtheta))[est]))
rate <- t(sapply(fits, function(a) a$rate))
vrate <- t(sapply(fits, function(a) a$vrate))
tconv <- sapply(fits, function(a) a$tconv.max)
stopifnot(all(tconv < 0.25))
num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")

cat('name = "s50_siena07_interaction"\n\n[provenance]\n')
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('rsiena_version = "%s"\n', as.character(packageVersion("RSiena"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/s50_siena07_interaction.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "RSiena::s50: 50 actors, 3 friendship waves, smoke1"\n')
cat('model = "conditional MoM (cond=TRUE); density, recip, gwespFF, smoke1 alter, smoke1 ego, smoke1 ego x reciprocity"\n')
cat('algorithm = "sienaAlgorithmCreate(cond=TRUE, nsub=4, n3=1000)"\n')
cat(sprintf('replication_seeds = "%s"\n\n', paste(rep_seeds, collapse = ",")))

cat("[tolerance]\n")
cat("# Reference = mean of the six RSiena fits; `*_seed_sd` is the per-fit\n")
cat("# seed-to-seed sd. The Julia test compares the mean of its own fits and\n")
cat("# derives each tolerance from both measured widths (see the testset).\n")
cat("default = 0.0\n\n")

cat("[values]\n")
cat(sprintf("effect_names = [%s]\n",
            paste(sprintf('"%s"', fits[[1]]$effects$effectName[est]), collapse = ", ")))
cat(sprintf("coefficients = [%s]\n", num(colMeans(theta))))
cat(sprintf("coefficients_seed_sd = [%s]\n", num(apply(theta, 2, sd))))
cat(sprintf("std_errors = [%s]\n", num(colMeans(se))))
cat(sprintf("std_errors_seed_sd = [%s]\n", num(apply(se, 2, sd))))
cat(sprintf("rates = [%s]\n", num(colMeans(rate))))
cat(sprintf("rates_seed_sd = [%s]\n", num(apply(rate, 2, sd))))
cat(sprintf("rate_std_errors = [%s]\n", num(colMeans(vrate))))
cat(sprintf("rate_std_errors_seed_sd = [%s]\n", num(apply(vrate, 2, sd))))
cat(sprintf("tconv_max = [%s]\n", num(tconv)))
