############################################################
# 1. LOAD REQUIRED PACKAGES
############################################################
library(dplyr)
library(caret)
library(glmnet)

############################################################
# 2. READ AND FILTER THE DATA
############################################################

cbcl <- read.csv("mh_p_cbcl.csv", stringsAsFactors = FALSE)
asr  <- read.csv("mh_p_asr.csv",  stringsAsFactors = FALSE)
abcl <- read.csv("mh_p_abcl.csv", stringsAsFactors = FALSE)

# Subset each data frame to 2-year follow-up rows
cbcl_2year <- subset(cbcl, eventname == "2_year_follow_up_y_arm_1")
asr_2year  <- subset(asr,  eventname == "2_year_follow_up_y_arm_1")
abcl_2year <- subset(abcl, eventname == "2_year_follow_up_y_arm_1")

# Merge them together on "src_subject_id"
merged_cbcl_asr <- merge(cbcl_2year, asr_2year, by = "src_subject_id", all = TRUE)
merged_df <- merge(merged_cbcl_asr, abcl_2year, by = "src_subject_id", all = TRUE)

# Rename for clarity
merged_df <- merged_df %>% rename(id = src_subject_id)

############################################################
# 3. SELECT VARIABLES OF INTEREST
############################################################
outcome_var <- "cbcl_scr_syn_external_t"

predictor_vars <- c(
  "asr_scr_perstr_t", "asr_scr_anxdep_t", "asr_scr_withdrawn_t", 
  "asr_scr_thought_t", "asr_scr_attention_t",
  "asr_scr_aggressive_t", "asr_scr_rulebreak_t", "asr_scr_intrusive_t",
  "asr_scr_internal_t", "asr_scr_external_t", "asr_scr_totprob_t",
  "asr_scr_depress_t", "asr_scr_anxdisord_t", "asr_scr_somaticpr_t",
  "asr_scr_avoidant_t", "asr_scr_adhd_t", "asr_scr_antisocial_t",
  "asr_scr_hyperactive_t",
  "abcl_scr_sub_use_tobacco_t", "abcl_scr_sub_use_alcohol_t",
  "abcl_scr_sub_use_drugs_t", "abcl_scr_sub_use_t_mean",
  "abcl_scr_adapt_friends_t", "abcl_scr_prob_anxious_t",
  "abcl_scr_prob_withdrawn_t", "abcl_scr_prob_somatic_t",
  "abcl_scr_prob_thought_t", "abcl_scr_prob_attention_t",
  "abcl_scr_prob_aggressive_t", "abcl_scr_prob_rulebreak_t",
  "abcl_scr_prob_intrusive_t", "abcl_scr_prob_internal_t",
  "abcl_scr_prob_external_t", "abcl_scr_prob_total_t",
  "abcl_scr_prob_critical_t"
)

# Keep only predictors that exist in merged_df (in case any are missing)
predictor_vars <- predictor_vars[predictor_vars %in% names(merged_df)]


############################################################
# 4. PREPARE MATRIX FOR GLMNET
############################################################
analysis_df <- merged_df %>%
  dplyr::select(
    all_of(outcome_var),
    all_of(predictor_vars)
  ) %>%
  na.omit()

X <- model.matrix(as.formula(paste(outcome_var, "~ .")), data = analysis_df)[, -1]
y <- analysis_df[[outcome_var]]


############################################################
# 5. FIT RIDGE, ELASTIC NET, LASSO VIA CV
############################################################
# Ridge (alpha = 0)
cv_ridge <- cv.glmnet(X, y, alpha = 0)  

# Elastic Net (alpha = 0.5)
cv_elastic <- cv.glmnet(X, y, alpha = 0.5)

# Lasso (alpha = 1)
cv_lasso <- cv.glmnet(X, y, alpha = 1)

############################################################
# 6. COMPARE CROSS-VALIDATED R²
############################################################
# "1 - cvm / var(y)" measures how much variance is explained by the model
ridge_r2   <- max(1 - cv_ridge$cvm / var(y))
elastic_r2 <- max(1 - cv_elastic$cvm / var(y))
lasso_r2   <- max(1 - cv_lasso$cvm / var(y))

cat("\n--- Cross-Validated R² ---\n")
cat("Ridge R²:      ", ridge_r2, "\n")
cat("Elastic Net R²:", elastic_r2, "\n")
cat("Lasso R²:      ", lasso_r2, "\n")

############################################################
# 7. EXTRACT FULL COEFFICIENTS AT lambda.min & lambda.1se
############################################################
# Utility function: convert coef() output to a nice data.frame
get_coefs_df <- function(cv_model, model_name, s_value = c("lambda.min", "lambda.1se")) {
  s_value <- match.arg(s_value)
  cfs <- coef(cv_model, s = s_value)
  df <- data.frame(
    variable = row.names(cfs),
    coefficient = as.numeric(cfs),
    model = model_name,
    lambda_type = s_value,
    stringsAsFactors = FALSE
  )
  return(df)
}

# Ridge
ridge_coefs_min  <- get_coefs_df(cv_ridge,    "Ridge",      "lambda.min")
ridge_coefs_1se  <- get_coefs_df(cv_ridge,    "Ridge",      "lambda.1se")

# Elastic Net
elastic_coefs_min  <- get_coefs_df(cv_elastic,  "ElasticNet", "lambda.min")
elastic_coefs_1se  <- get_coefs_df(cv_elastic,  "ElasticNet", "lambda.1se")

# Lasso
lasso_coefs_min  <- get_coefs_df(cv_lasso,    "Lasso",      "lambda.min")
lasso_coefs_1se  <- get_coefs_df(cv_lasso,    "Lasso",      "lambda.1se")

# Combine into one data frame
all_coefs <- dplyr::bind_rows(
  ridge_coefs_min,
  ridge_coefs_1se,
  elastic_coefs_min,
  elastic_coefs_1se,
  lasso_coefs_min,
  lasso_coefs_1se
) %>%
  dplyr::arrange(model, lambda_type, variable)


cat("\n--- Sample of Coefficients Data Frame ---\n")
print(head(all_coefs, 20))

write.csv(all_coefs, "penalized_regression_coefficients.csv", row.names = FALSE)

############################################################
# DONE
############################################################

cat("\nScript completed. Check above for model R² and coefficients.\n")

