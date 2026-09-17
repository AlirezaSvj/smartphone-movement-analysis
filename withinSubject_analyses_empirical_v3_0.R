# =============================================================================
# MyProject: 3D Motion Capture Analysis in Fine Motor Tasks
# Developing a Methodological Framework with Phone Handling as a Pilot Study.
# Copyright (C) 2026 Author(s)
#
# Script: Empirical Data — Extended Feature PCA, LDA, and LOSO RF
# Version: 3.0 (finalized 2026-07-13)
#
# Data sources:
#   dfa_aggregated_extended.csv  — 36 rows, one per empirical participant.
#                                  Extended feature set (45+ variables).
#                                  Used for between-person PCA, LDA, LOSO RF.
#
#   final_extracted_data.csv     — trial-level, one row per trial.
#                                  6 core kinematic variables (already z-scored).
#                                  Used for within-person LOSO RF.
#
# Design rationale:
#   Between-person — full extended pipeline (collinearity screening, KMO,
#     variable selection, PCA + LDA + LOSO RF).
#     DN/DI labels reflect real cohort membership.
#
#   Within-person — LOSO RF only on 6 core variables, no PCA or LDA.
#     Within-person centering removes between-person variance.
#     Predicts environment (G vs W) from trial-level kinematics.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. PACKAGES
# -----------------------------------------------------------------------------

packages <- c("psych", "DescTools", "MASS", "caret", "randomForest",
              "dplyr", "tidyr", "ggplot2", "MVN", "biotools", "patchwork")

lapply(packages, library, character.only = TRUE)

# -----------------------------------------------------------------------------
# 2. OUTPUT DIRECTORY
# -----------------------------------------------------------------------------

dir.create("figures/PCA_LDA_RF_empirical", recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 3. SHARED PLOT THEME
# -----------------------------------------------------------------------------

theme_paper <- theme_minimal(base_size = 13) +
  theme(
    plot.title      = element_text(face = "bold", size = 13),
    plot.subtitle   = element_text(size = 11, color = "grey40"),
    axis.title      = element_text(size = 11),
    axis.text       = element_text(size = 10),
    legend.position = "top",
    legend.title    = element_text(size = 10),
    legend.text     = element_text(size = 9)
  )

# =============================================================================
# PART A — BETWEEN-PERSON PIPELINE
# Source: dfa_aggregated_extended.csv (N = 36, extended feature set)
# =============================================================================

# -----------------------------------------------------------------------------
# 4A. LOAD AND PREPARE BETWEEN-PERSON DATA
# -----------------------------------------------------------------------------

between_raw <- read.csv("dfa_aggregated_extended.csv")

between_raw$group  <- factor(between_raw$group)
between_raw$gender <- factor(between_raw$gender)
between_raw$file   <- factor(between_raw$file)

cat("Between-person dataset:\n")
cat("  Rows:", nrow(between_raw), "\n")
cat("  Participants:", nlevels(between_raw$file), "\n")
cat("  Group distribution:\n")
print(table(between_raw$group))

# Drop metadata columns — retain numeric kinematic features only
# Metadata: file, group, gender, education, levelOfUse, age (columns 1-6)
pca_data <- between_raw[, -c(1:6)]

# Winsorize and z-standardize all features
pca_data[] <- lapply(pca_data, winsor, trim = 0.05)
pca_data[] <- lapply(pca_data, scale)

# -----------------------------------------------------------------------------
# 5A. COLLINEARITY SCREENING
# -----------------------------------------------------------------------------

cor_result <- corr.test(pca_data, method = "spearman")
cor_mat    <- cor_result$r

to_remove   <- findCorrelation(cor_mat, cutoff = 0.95, names = TRUE)
pca_reduced <- pca_data[, !(names(pca_data) %in% to_remove)]

cat("\nVariables removed by collinearity screening (r > 0.95):\n")
print(to_remove)
cat("\nVariables retained after screening:", ncol(pca_reduced), "\n")

# KMO before manual variable selection
cat("\n--- KMO after collinearity screening ---\n")
KMO(pca_reduced)

# -----------------------------------------------------------------------------
# 6A. MANUAL VARIABLE SELECTION (KMO-guided)
#     Variables below MSA = 0.50 removed.
#     This list reflects the empirical data's correlation structure.
# -----------------------------------------------------------------------------

vars_to_keep <- c("m_motion_rms", "m_tilt_rms_rad", "max_tilt_freq_Hz",
                  "min_motion_rms", "sd_motion_rms", "sd_motion_ptp",
                  "sd_tilt_freq_Hz", "sd_tilt_rms_rad", "mid_motion_ptp",
                  "mid_tilt_rms_rad", "range_motion_freq_Hz",
                  "range_tilt_freq_Hz", "cv_motion_ptp",
                  "cv_tilt_freq_Hz", "cv_tilt_rms_rad", "cv_tilt_ptp_rad")

# Filter to only variables that survived collinearity screening
vars_to_keep <- vars_to_keep[vars_to_keep %in% names(pca_reduced)]
pca_reduced  <- pca_reduced[, vars_to_keep]

cat("\nFinal variables retained:", length(vars_to_keep), "\n")
print(vars_to_keep)

# Final KMO and Bartlett on cleaned dataset
cat("\n--- Final KMO and Bartlett (empirical) ---\n")
KMO(pca_reduced)
psych::cortest.bartlett(pca_reduced)

# -----------------------------------------------------------------------------
# 7A. PARALLEL ANALYSIS
# -----------------------------------------------------------------------------

fa.parallel(pca_reduced, fa = "pc", n.iter = 100,
            main = "Scree Plot — Between-person (empirical)")

# Set based on parallel analysis output above
n_components <- 3

# -----------------------------------------------------------------------------
# 8A. PCA WITH VARIMAX ROTATION
# -----------------------------------------------------------------------------

pca1    <- prcomp(pca_reduced, center = TRUE, scale. = TRUE)
pca_rot <- principal(pca_reduced, nfactors = n_components, rotate = "varimax")

cat("\n--- Varimax loadings: Between-person (empirical) ---\n")
print(pca_rot$loadings, cutoff = 0.3)

var_explained <- pca1$sdev^2 / sum(pca1$sdev^2)
cum_var       <- cumsum(var_explained)

cat("\nCumulative variance explained:\n")
print(round(cum_var, 3))

p_cumvar <- data.frame(component  = seq_along(cum_var),
                       cumulative = cum_var) %>%
  ggplot(aes(x = component, y = cumulative)) +
  geom_point(color = "steelblue", size = 2.5) +
  geom_line(color  = "steelblue") +
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "firebrick") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(title    = "Cumulative Variance Explained — Between-person (empirical)",
       subtitle = "Red dashed line = 80% threshold",
       x = "Principal Component",
       y = "Cumulative Proportion of Variance Explained") +
  theme_paper

ggsave("figures/PCA_LDA_RF_empirical/pca_cumvar_between.png",
       p_cumvar, width = 7, height = 5, dpi = 300)

# Store PC scores with explicit column names
pc_scores <- as.data.frame(pca1$x[, 1:n_components])
colnames(pc_scores) <- paste0("PC", 1:n_components)

# -----------------------------------------------------------------------------
# 9A. LDA
#     Predicts group (DN vs DI) from PC scores.
# -----------------------------------------------------------------------------

cat("\n=== BETWEEN-PERSON LDA: Group (DN vs DI) — Empirical ===\n")

pca_reduced$group <- factor(between_raw$group)

lda_mardia <- mvn(pca_reduced[, vars_to_keep], mvn_test = "mardia")
print(lda_mardia)
lda_boxM   <- suppressWarnings(
  boxM(pca_reduced[, vars_to_keep], pca_reduced$group)
)
print(lda_boxM)

lda_model <- lda(group ~ pca1$x[, 1:n_components], data = pca_reduced)
pred      <- predict(lda_model)

df_lda <- data.frame(
  Observed     = pca_reduced$group,
  Predicted    = pred$class,
  LD1          = pred$x[, 1],
  Posterior_DI = pred$posterior[, "DI"],
  Posterior_DN = pred$posterior[, "DN"]
) %>%
  mutate(
    Confidence = pmax(Posterior_DI, Posterior_DN),
    Correct    = Observed == Predicted
  )

centroids <- df_lda %>%
  group_by(Observed) %>%
  summarise(LD1 = mean(LD1), .groups = "drop")

boundary <- mean(centroids$LD1)

eigvals      <- lda_model$svd^2
wilks_lambda <- prod(1 / (1 + eigvals))
n            <- nrow(pca_reduced)
p            <- n_components
n_groups     <- length(unique(pca_reduced$group))
chi_sq       <- -((n - 1) - (p + n_groups) / 2) * log(wilks_lambda)
df_chi       <- p * (n_groups - 1)
p_value      <- pchisq(chi_sq, df = df_chi, lower.tail = FALSE)

cat("Wilks lambda:", round(wilks_lambda, 4), "\n")
cat("Chi-square:", round(chi_sq, 3), "| df:", df_chi,
    "| p =", round(p_value, 4), "\n")

p_lda_sep <- ggplot(df_lda, aes(x = LD1, fill = Observed)) +
  geom_density(alpha = 0.4, color = "black") +
  geom_vline(data = centroids, aes(xintercept = LD1, color = Observed),
             linetype = "dashed", linewidth = 1.2) +
  geom_vline(xintercept = boundary, color = "firebrick",
             linetype = "solid", linewidth = 1) +
  labs(title    = "LDA Separation — Group (DN vs DI) (empirical)",
       subtitle = "Dashed = group centroids | Red = decision boundary",
       x = "LD1 discriminant score", y = "Density") +
  theme_paper

ggsave("figures/PCA_LDA_RF_empirical/lda_separation_between_group.png",
       p_lda_sep, width = 7, height = 5, dpi = 300)

p_lda_conf <- ggplot(df_lda, aes(x = LD1, y = Confidence, color = Correct)) +
  geom_point(size = 3, alpha = 0.75) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "grey40") +
  geom_vline(xintercept = boundary, color = "firebrick", linetype = "solid") +
  scale_color_manual(values = c("TRUE" = "forestgreen", "FALSE" = "firebrick"),
                     labels = c("TRUE" = "Correct", "FALSE" = "Incorrect"),
                     name   = "Prediction") +
  labs(title    = "Classification Confidence — Group (empirical)",
       subtitle = "Points = individual cases | Color = correct vs incorrect",
       x = "LD1 discriminant score",
       y = "Model confidence (posterior probability)") +
  theme_paper

ggsave("figures/PCA_LDA_RF_empirical/lda_confidence_between_group.png",
       p_lda_conf, width = 7, height = 5, dpi = 300)

# -----------------------------------------------------------------------------
# 10A. BETWEEN-PERSON LOSO RF (empirical)
#      Operates at participant level (N = 36).
#      Each fold: train on 35 participant means, test on 1.
#      Majority class baseline = 23/36 = 63.9% (DN).
# -----------------------------------------------------------------------------

cat("\n=== BETWEEN-PERSON LOSO RF: Group (DN vs DI) — Empirical ===\n")
cat("Majority class baseline (always predict DN): 63.9%\n\n")

participants_between <- levels(between_raw$file)
cat("Participants in LOSO:", length(participants_between), "\n\n")

set.seed(42)
loso_between_results <- do.call(rbind, lapply(participants_between, function(p) {

  test_idx  <- which(between_raw$file == p)
  train_idx <- which(between_raw$file != p)

  train_pcs <- pc_scores[train_idx, , drop = FALSE]
  test_pcs  <- pc_scores[test_idx,  , drop = FALSE]

  train_pcs$outcome <- between_raw$group[train_idx]
  test_pcs$outcome  <- between_raw$group[test_idx]

  rf <- randomForest(
    outcome ~ .,
    data  = train_pcs,
    ntree = 500,
    mtry  = max(1, floor(sqrt(n_components)))
  )

  preds <- predict(rf, newdata = test_pcs)

  data.frame(
    participant = p,
    observed    = as.character(between_raw$group[test_idx]),
    predicted   = as.character(preds)
  )
}))

loso_between_results$observed  <- factor(loso_between_results$observed,
                                         levels = levels(between_raw$group))
loso_between_results$predicted <- factor(loso_between_results$predicted,
                                         levels = levels(between_raw$group))

cat("Total predictions (must equal 36):", nrow(loso_between_results), "\n")

cm_between <- confusionMatrix(loso_between_results$predicted,
                              loso_between_results$observed)

cat("\n--- LOSO RF result: Group (DN vs DI) — Empirical ---\n")
print(cm_between)

p_between_loso <- loso_between_results %>%
  mutate(correct = observed == predicted) %>%
  group_by(participant, observed) %>%
  summarise(accuracy = mean(correct), .groups = "drop") %>%
  ggplot(aes(x = participant, y = accuracy, fill = observed)) +
  geom_col(position = "dodge", alpha = 0.8) +
  geom_hline(yintercept = 0.5,    linetype = "dashed", color = "grey40") +
  geom_hline(yintercept = 0.6389, linetype = "dotted", color = "firebrick") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(title    = "LOSO RF Accuracy per Participant — Group (empirical)",
       subtitle = "Grey dashed = chance | Red dotted = majority class baseline (63.9%)",
       x = "Participant", y = "Classification Accuracy") +
  theme_paper +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7))

ggsave("figures/PCA_LDA_RF_empirical/rf_loso_accuracy_between_group.png",
       p_between_loso, width = 10, height = 5, dpi = 300)

# =============================================================================
# PART B — WITHIN-PERSON PIPELINE
# Source: final_extracted_data.csv (trial-level, 6 core variables)
# Data is already z-scored — skip winsorize and scale.
# Participant ID = file column. Platform = environment column.
# LOSO RF only — predicts environment (G vs W).
# =============================================================================

# -----------------------------------------------------------------------------
# 4B. LOAD AND PREPARE WITHIN-PERSON DATA
# -----------------------------------------------------------------------------

main_vars <- c("motion_freq_Hz", "motion_rms", "motion_ptp",
               "tilt_freq_Hz",   "tilt_rms_rad", "tilt_ptp_rad")

df_trials <- read.csv("final_extracted_data.csv")

# Keep only valid trials (status == "ok")
df_trials <- df_trials[df_trials$status == "ok", ]

# Participant ID and environment as factors
df_trials$participant_id <- factor(df_trials$file)
df_trials$environment    <- factor(df_trials$environment)

# Data is already z-scored — do NOT re-scale
cat("\nWithin-person dataset:\n")
cat("  Rows:", nrow(df_trials), "\n")
cat("  Participants:", nlevels(df_trials$participant_id), "\n")
cat("  Environment distribution:\n")
print(table(df_trials$environment))

# Keep only complete cases on the 6 variables
df_clean <- df_trials %>%
  select(participant_id, environment, all_of(main_vars)) %>%
  filter(complete.cases(.))

cat("  Complete cases:", nrow(df_clean), "\n")

# -----------------------------------------------------------------------------
# 5B. WITHIN-PERSON CENTERING
#     Remove between-person variance so classifier must rely only on
#     within-person platform differences.
# -----------------------------------------------------------------------------

within_df <- df_clean %>%
  group_by(participant_id) %>%
  mutate(across(all_of(main_vars), ~ . - mean(.))) %>%
  ungroup()

# Check for zero-variance columns after centering
zero_var_cols <- sapply(within_df[, main_vars],
                        function(x) var(x, na.rm = TRUE) < 1e-10)

if (any(zero_var_cols)) {
  cat("\nZero-variance columns after centering:\n")
  cat(names(zero_var_cols)[zero_var_cols], "\n")
  cat("Adding numerical stabilization jitter (SD = 1e-6).\n")
  set.seed(42)
  within_df[, main_vars[zero_var_cols]] <- lapply(
    within_df[, main_vars[zero_var_cols]],
    function(x) x + rnorm(length(x), 0, 1e-6)
  )
}

# -----------------------------------------------------------------------------
# 6B. WITHIN-PERSON LOSO RF (empirical)
#     Predicts environment (G vs W) from centered raw features.
#     Chance baseline = 50% (balanced platform design).
# -----------------------------------------------------------------------------

cat("\n=== WITHIN-PERSON LOSO RF: Environment (G vs W) — Empirical ===\n")
cat("Chance baseline: 50% (balanced platform design)\n\n")

participants_within <- levels(within_df$participant_id)
cat("Participants in LOSO:", length(participants_within), "\n\n")

set.seed(42)
loso_within_results <- do.call(rbind, lapply(participants_within, function(p) {

  train_idx <- which(within_df$participant_id != p)
  test_idx  <- which(within_df$participant_id == p)

  train_feats <- within_df[train_idx, main_vars]
  test_feats  <- within_df[test_idx,  main_vars]

  train_feats$outcome <- within_df$environment[train_idx]
  test_feats$outcome  <- within_df$environment[test_idx]

  rf <- randomForest(
    outcome ~ .,
    data  = train_feats,
    ntree = 500,
    mtry  = max(1, floor(sqrt(length(main_vars))))
  )

  preds <- predict(rf, newdata = test_feats)

  data.frame(
    participant = p,
    observed    = as.character(within_df$environment[test_idx]),
    predicted   = as.character(preds)
  )
}))

loso_within_results$observed  <- factor(loso_within_results$observed,
                                        levels = levels(within_df$environment))
loso_within_results$predicted <- factor(loso_within_results$predicted,
                                        levels = levels(within_df$environment))

cat("Total predictions:", nrow(loso_within_results), "\n")

cm_within <- confusionMatrix(loso_within_results$predicted,
                             loso_within_results$observed)

cat("\n--- LOSO RF result: Environment (G vs W) — Empirical ---\n")
print(cm_within)

p_within_loso <- loso_within_results %>%
  mutate(correct = observed == predicted) %>%
  group_by(participant, observed) %>%
  summarise(accuracy = mean(correct), .groups = "drop") %>%
  ggplot(aes(x = participant, y = accuracy, fill = observed)) +
  geom_col(position = "dodge", alpha = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey40") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(title    = "LOSO RF Accuracy per Participant — Environment (empirical)",
       subtitle = "Dashed = chance level (50%)",
       x = "Participant", y = "Classification Accuracy") +
  theme_paper +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7))

ggsave("figures/PCA_LDA_RF_empirical/rf_loso_accuracy_within_environment.png",
       p_within_loso, width = 10, height = 5, dpi = 300)

# =============================================================================
# License: GNU General Public License v3.0 (GPL-3.0)
# See LICENSE file or https://www.gnu.org/licenses/
# =============================================================================
