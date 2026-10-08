#!/usr/bin/env Rscript
# Golden fixture: RSiena siena07 fit of network-behaviour CO-EVOLUTION with
# selection on the co-evolving behaviour (the s50 alcohol selection-and-influence
# model of the RSiena tutorial).
#
# Regenerate from the package root (twenty RSiena fits, about ten minutes):
#
#   Rscript test/fixtures/r/s50_coevolution.R > test/fixtures/s50_coevolution.toml
#
# Model: friendship ~ density + recip + transTrip + alcohol ego/alter/similarity
# (selection on the DEPENDENT behaviour); alcohol ~ linear + quad + average
# similarity (influence). Two dependent variables, so RSiena's default is
# unconditional estimation (cond = NA -> FALSE).
#
# The reference values are the MEAN over twenty RSiena fits with the seed-to-seed
# sd frozen beside them (see s50_siena07_cond.R for the reasoning). Twenty, not
# six: the six fits of the first version of this fixture happened to have a
# seed-to-seed sd of 0.036 for the first friendship rate's standard error, where
# fourteen further seeds gave 0.090. The Julia test's tolerances are multiples of
# this sd, so an understated sd made an agreeing estimator look like an outlier.
suppressMessages(library(RSiena))
seed <- 20261004L
rep_seeds <- c(121L, 232L, 343L, 454L, 565L, 1001:1014)

friendship <- sienaDependent(array(c(s501, s502, s503), dim = c(50, 50, 3)))
alcohol <- sienaDependent(s50a, type = "behavior")
dat <- sienaDataCreate(friendship, alcohol)
eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, name = "friendship", verbose = FALSE)
eff <- includeEffects(eff, egoX, altX, simX, interaction1 = "alcohol",
                      name = "friendship", verbose = FALSE)
eff <- includeEffects(eff, avSim, interaction1 = "friendship", name = "alcohol",
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
stopifnot(!fits[[1]]$cconditional)
theta <- t(sapply(fits, function(a) a$theta))
se <- t(sapply(fits, function(a) sqrt(diag(a$covtheta))))
tconv <- sapply(fits, function(a) a$tconv.max)
num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")

cat('name = "s50_coevolution"\n\n[provenance]\n')
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('rsiena_version = "%s"\n', as.character(packageVersion("RSiena"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/s50_coevolution.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "RSiena::s50: friendship (3 waves) and alcohol use (dependent behaviour)"\n')
cat('model = "unconditional MoM; friendship: density, recip, transTrip, alcohol ego/alter/similarity; alcohol: linear, quad, avSim"\n')
cat('algorithm = "sienaAlgorithmCreate(nsub=4, n3=1000) (cond = NA -> unconditional)"\n')
cat(sprintf('replication_seeds = "%s"\n\n', paste(rep_seeds, collapse = ",")))
cat("[tolerance]\n# See the Julia testset: tolerances come from the measured widths.\ndefault = 0.0\n\n")
cat("[values]\n")
cat(sprintf("n_fits = %d\n", length(fits)))
cat(sprintf("effect_names = [%s]\n",
            paste(sprintf('"%s"', fits[[1]]$effects$effectName), collapse = ", ")))
cat(sprintf("coefficients = [%s]\n", num(colMeans(theta))))
cat(sprintf("coefficients_seed_sd = [%s]\n", num(apply(theta, 2, sd))))
cat(sprintf("std_errors = [%s]\n", num(colMeans(se))))
cat(sprintf("std_errors_seed_sd = [%s]\n", num(apply(se, 2, sd))))
cat(sprintf("tconv_max = [%s]\n", num(tconv)))
