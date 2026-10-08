#!/usr/bin/env Rscript
# Golden fixture: RSiena CONDITIONAL siena07 fits (cond = TRUE) on s50 panels whose
# first period is up-only or down-only, so that RSiena's allowOnly restriction is in
# force in that period. Pins the conditional rate estimate AND its standard error
# in a restricted period.
#
# Regenerate from the package root (forty RSiena fits, about ten minutes):
#
#   Rscript test/fixtures/r/s50_allowonly_cond.R > test/fixtures/s50_allowonly_cond.toml
#
# Panels (s501, s502, s503 are RSiena's s50 friendship waves):
#   uponly:   waves (s501, pmax(s501, s502), s503) -- period 1 only adds ties
#   downonly: waves (s501, pmin(s501, s502), s503) -- period 1 only drops ties
# Model: RSiena's defaults (rates, density, recip) + transitive triplets.
#
# The conditional rate's standard error is RSiena's `vrate`, the sd of the
# simulated stopping times (terminateFRAN: `z$vrate <- apply(z$ntim, 2, sd)`); it
# is printed as the rate's standard error and is ALREADY an sd -- it must not be
# square-rooted. Siena.jl's `rate_standard_errors` is the same quantity
# (simulation rate x sd of the stopping time).
#
# The reference values are the MEAN over twenty RSiena fits per panel with the
# seed-to-seed sd frozen beside them (see s50_siena07_cond.R for the reasoning).
# Twenty, not six: a standard deviation estimated from six fits is itself uncertain
# by about a third, and the tolerances of the Julia test are multiples of it.
suppressMessages(library(RSiena))
seed <- 20261007L
rep_seeds <- c(131L, 242L, 353L, 464L, 575L, 2001:2014)

fit_panel <- function(mid) {
  friendship <- sienaDependent(array(c(s501, mid, s503), dim = c(50, 50, 3)))
  dat <- NULL
  invisible(capture.output(dat <- sienaDataCreate(friendship), type = "output"))
  eff <- NULL
  invisible(capture.output(eff <- getEffects(dat), type = "output"))
  eff <- includeEffects(eff, transTrip, name = "friendship", verbose = FALSE)
  lapply(c(seed, rep_seeds), function(s) {
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
  })
}

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
emit <- function(prefix, fits) {
  cat(sprintf("%s_n_fits = %d\n", prefix, length(fits)))
  stopifnot(all(sapply(fits, function(a) a$cconditional)))
  stopifnot(all(sapply(fits, function(a) a$tconv.max) < 0.25))
  theta <- t(sapply(fits, function(a) a$theta))
  se <- t(sapply(fits, function(a) sqrt(diag(a$covtheta))))
  rate <- t(sapply(fits, function(a) a$rate))
  vrate <- t(sapply(fits, function(a) a$vrate))
  cat(sprintf("%s_effect_names = [%s]\n", prefix,
              paste(sprintf('"%s"', fits[[1]]$effects$effectName), collapse = ", ")))
  cat(sprintf("%s_coefficients = [%s]\n", prefix, num(colMeans(theta))))
  cat(sprintf("%s_coefficients_seed_sd = [%s]\n", prefix, num(apply(theta, 2, sd))))
  cat(sprintf("%s_std_errors = [%s]\n", prefix, num(colMeans(se))))
  cat(sprintf("%s_std_errors_seed_sd = [%s]\n", prefix, num(apply(se, 2, sd))))
  cat(sprintf("%s_rates = [%s]\n", prefix, num(colMeans(rate))))
  cat(sprintf("%s_rates_seed_sd = [%s]\n", prefix, num(apply(rate, 2, sd))))
  cat(sprintf("%s_rate_std_errors = [%s]\n", prefix, num(colMeans(vrate))))
  cat(sprintf("%s_rate_std_errors_seed_sd = [%s]\n", prefix, num(apply(vrate, 2, sd))))
}

up <- fit_panel(pmax(s501, s502))
down <- fit_panel(pmin(s501, s502))

cat('name = "s50_allowonly_cond"\n\n[provenance]\n')
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('rsiena_version = "%s"\n', as.character(packageVersion("RSiena"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/s50_allowonly_cond.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "RSiena::s50 friendship; period 1 made up-only (pmax) or down-only (pmin)"\n')
cat('model = "conditional MoM (cond=TRUE), allowOnly = TRUE; density, recip, transTrip"\n')
cat('algorithm = "sienaAlgorithmCreate(cond=TRUE, nsub=4, n3=1000)"\n')
cat(sprintf('replication_seeds = "%s"\n\n', paste(rep_seeds, collapse = ",")))

cat("[tolerance]\n")
cat("# Reference = mean of the twenty RSiena fits per panel; `*_seed_sd` is the per-fit\n")
cat("# seed-to-seed sd. The rate standard errors are RSiena's vrate (already an sd).\n")
cat("default = 0.0\n\n")

cat("[values]\n")
emit("uponly", up)
emit("downonly", down)
