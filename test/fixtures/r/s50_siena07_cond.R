#!/usr/bin/env Rscript
# Golden fixture: RSiena CONDITIONAL siena07 fit (cond = TRUE, RSiena's default for
# one dependent network) on the bundled s50 data, with a score-type test.
#
# Regenerate from the package root (six RSiena fits, a few minutes):
#
#   Rscript test/fixtures/r/s50_siena07_cond.R > test/fixtures/s50_siena07_cond.toml
#
# Model: friendship ~ density + reciprocity + transitive triplets + smoke1
# alter/ego/similarity, conditional on the observed amount of change in each period,
# plus 3-cycles held fixed at 0 and TESTED (fix = TRUE, test = TRUE: RSiena's
# generalised score-type test, Schweinberger 2012).
#
# A fitted SAOM coefficient is a Monte-Carlo quantity. The reference values are
# the MEAN over six RSiena fits (the main seed and five replications), and the
# per-coefficient seed-to-seed sd is frozen beside them, so the Julia test's
# tolerances can be stated as multiples of the measured Monte-Carlo width rather
# than chosen to pass.
suppressMessages(library(RSiena))
seed <- 20261002L
rep_seeds <- c(101L, 202L, 303L, 404L, 505L)

friendship <- sienaDependent(array(c(s501, s502, s503), dim = c(50, 50, 3)))
smoke1 <- coCovar(s50s[, 1])
dat <- sienaDataCreate(friendship, smoke1)
eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, name = "friendship", verbose = FALSE)
eff <- includeEffects(eff, egoX, altX, simX, interaction1 = "smoke1",
                      name = "friendship", verbose = FALSE)
eff <- includeEffects(eff, cycle3, name = "friendship", fix = TRUE, test = TRUE,
                      verbose = FALSE)

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
est <- !fits[[1]]$fix                       # the estimated parameters
theta <- t(sapply(fits, function(a) a$theta[est]))
se <- t(sapply(fits, function(a) sqrt(diag(a$covtheta))[est]))
rate <- t(sapply(fits, function(a) a$rate))
vrate <- t(sapply(fits, function(a) a$vrate))
chisq <- sapply(fits, function(a) a$testresOverall)
onesided <- sapply(fits, function(a) a$testresulto[1])
onestep <- sapply(fits, function(a) a$oneStep[!est][1])
tconv <- sapply(fits, function(a) a$tconv.max)
stopifnot(all(tconv < 0.25))
num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")

cat('name = "s50_siena07_cond"\n\n[provenance]\n')
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('rsiena_version = "%s"\n', as.character(packageVersion("RSiena"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/s50_siena07_cond.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "RSiena::s50: 50 actors, 3 friendship waves, smoke1"\n')
cat('model = "conditional MoM (cond=TRUE); density, recip, transTrip, smoke1 alter/ego/similarity; cycle3 fixed at 0 and score-tested"\n')
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
cat(sprintf("score_test_chisq = %.17g\n", mean(chisq)))
cat(sprintf("score_test_chisq_seed_sd = %.17g\n", sd(chisq)))
cat(sprintf("score_test_one_sided = %.17g\n", mean(onesided)))
cat(sprintf("score_test_one_sided_seed_sd = %.17g\n", sd(onesided)))
cat(sprintf("score_test_one_step = %.17g\n", mean(onestep)))
cat(sprintf("score_test_one_step_seed_sd = %.17g\n", sd(onestep)))
cat(sprintf("tconv_max = [%s]\n", num(tconv)))
