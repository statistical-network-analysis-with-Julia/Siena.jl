#!/usr/bin/env Rscript
# Golden fixture: RSiena sienaTimeTest (time-heterogeneity score-type tests) on s50.
#
# Regenerate from the package root (six RSiena fits, a few minutes):
#
#   Rscript test/fixtures/r/s50_time_test.R > test/fixtures/s50_time_test.toml
#
# Model: friendship ~ density + reciprocity + transitive triplets + smoke1
# alter/ego/similarity, conditional (RSiena's default for one network), three waves
# = two periods, so every effect gets one dummy (period 2). sienaTimeTest computes
# its statistics from the phase-3 simulations of a fit, so they are Monte-Carlo
# quantities: the reference values are the MEAN over six RSiena fits with the
# seed-to-seed sd frozen beside them (see s50_siena07_cond.R for the reasoning).
suppressMessages(library(RSiena))
seed <- 20261006L
rep_seeds <- c(131L, 242L, 353L, 464L, 575L)

friendship <- sienaDependent(array(c(s501, s502, s503), dim = c(50, 50, 3)))
smoke1 <- coCovar(s50s[, 1])
dat <- sienaDataCreate(friendship, smoke1)
eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, name = "friendship", verbose = FALSE)
eff <- includeEffects(eff, egoX, altX, simX, interaction1 = "smoke1",
                      name = "friendship", verbose = FALSE)

one <- function(s) {
  alg <- NULL
  invisible(capture.output(
    alg <- sienaAlgorithmCreate(projname = NULL, seed = s, cond = TRUE,
                                nsub = 4, n3 = 1000), type = "output"))
  ans <- NULL
  invisible(capture.output(
    ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, verbose = FALSE,
                   silent = TRUE, useCluster = FALSE), type = "output"))
  stopifnot(ans$tconv.max < 0.25)
  tt <- NULL
  invisible(capture.output(tt <- sienaTimeTest(ans), type = "output"))
  list(joint = as.numeric(tt$JointTestStatistics$testresOverall),
       effect = as.numeric(tt$EffectTestStatistics),
       indiv = as.numeric(tt$IndividualTestStatistics),
       group = as.numeric(tt$GroupTestStatistics),
       onestep = as.numeric(tt$IndividualTest[7:12, 2]),
       names = rownames(tt$EffectTest))
}
res <- lapply(c(seed, rep_seeds), one)
mat <- function(f) t(sapply(res, function(r) r[[f]]))
num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
joint <- sapply(res, function(r) r$joint)

cat('name = "s50_time_test"\n\n[provenance]\n')
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('rsiena_version = "%s"\n', as.character(packageVersion("RSiena"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/s50_time_test.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "RSiena::s50: 50 actors, 3 friendship waves, smoke1"\n')
cat('model = "conditional MoM; density, recip, transTrip, smoke1 alter/ego/similarity; sienaTimeTest(ans)"\n')
cat(sprintf('replication_seeds = "%s"\n\n', paste(rep_seeds, collapse = ",")))
cat("[tolerance]\n# See the Julia testset: tolerances come from the measured widths.\ndefault = 0.0\n\n")
cat("[values]\n")
cat(sprintf("effect_names = [%s]\n", paste(sprintf('"%s"', res[[1]]$names), collapse = ", ")))
cat(sprintf("joint_chisq = %.17g\n", mean(joint)))
cat(sprintf("joint_chisq_seed_sd = %.17g\n", sd(joint)))
cat("joint_df = 6\n")
cat(sprintf("effect_chisq = [%s]\n", num(colMeans(mat("effect")))))
cat(sprintf("effect_chisq_seed_sd = [%s]\n", num(apply(mat("effect"), 2, sd))))
cat(sprintf("individual_z = [%s]\n", num(colMeans(mat("indiv")))))
cat(sprintf("individual_z_seed_sd = [%s]\n", num(apply(mat("indiv"), 2, sd))))
cat(sprintf("period_chisq = [%s]\n", num(colMeans(mat("group")))))
cat(sprintf("period_chisq_seed_sd = [%s]\n", num(apply(mat("group"), 2, sd))))
cat(sprintf("one_step = [%s]\n", num(colMeans(mat("onestep")))))
cat(sprintf("one_step_seed_sd = [%s]\n", num(apply(mat("onestep"), 2, sd))))
