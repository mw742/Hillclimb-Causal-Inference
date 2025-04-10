############################################################
# pick_factors_check_corrected.R
# PURPOSE:
#   1. Check correlations among predictors
#   2. Automatically remove highly correlated predictors
#   3. Fit linear model and check for aliased (redundant) coefficients
#   4. If aliasing remains, remove offending variables
############################################################

############################################################
# 1. LOAD REQUIRED PACKAGES
############################################################
library(dplyr)            # for data wrangling
library(car)              # for alias() and vif()
library(caret)            # for findCorrelation()
library(corrplot)
############################################################
# 2. READ AND FILTER THE DATA
############################################################
cbcl <- read.csv("mh_p_cbcl.csv", stringsAsFactors = FALSE)
asr  <- read.csv("mh_p_asr.csv",  stringsAsFactors = FALSE)
abcl <- read.csv("mh_p_abcl.csv", stringsAsFactors = FALSE)

cbcl_2year <- subset(cbcl, eventname == "2_year_follow_up_y_arm_1")
asr_2year  <- subset(asr,  eventname == "2_year_follow_up_y_arm_1")
abcl_2year <- subset(abcl, eventname == "2_year_follow_up_y_arm_1")

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
  "asr_scr_somatic_t", "asr_scr_thought_t", "asr_scr_attention_t",
  "asr_scr_aggressive_t", "asr_scr_rulebreak_t", "asr_scr_intrusive_t",
  "asr_scr_internal_t", "asr_scr_external_t", "asr_scr_totprob_t",
  "asr_scr_depress_t", "asr_scr_anxdisord_t", "asr_scr_somaticpr_t",
  "asr_scr_avoidant_t", "asr_scr_adhd_t", "asr_scr_antisocial_t",
  "asr_scr_inattention_t", "asr_scr_hyperactive_t",
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

# Keep only predictors that exist in merged_df
predictor_vars <- predictor_vars[predictor_vars %in% names(merged_df)]

# Select outcome + predictors and remove rows with any NA in those columns
analysis_df <- merged_df %>%
  dplyr::select(
    id,
    all_of(outcome_var),
    all_of(predictor_vars)
  ) %>%
  na.omit()

cat("\n>>> Dimensions of analysis_df:", dim(analysis_df), "\n")

############################################################
# 4. CORRELATION MATRIX
############################################################
cat("\n>>> Computing correlation matrix...\n")
# Only compute correlation among the predictor variables
corr_matrix <- cor(analysis_df[, predictor_vars], use = "complete.obs")

# Save correlation matrix to CSV
write.csv(corr_matrix, file = "corr_matrix.csv", row.names = TRUE)


# 3. Visualize with corrplot

library(pheatmap)
library(ggplot2)

pdf("corrplot_heatmap.pdf", width = 10, height = 8)
corrplot(
  corr_matrix,
  method = "color",
  order = "hclust",
  # Use a blue-white-red palette, so negative = blue, positive = red
  col = colorRampPalette(c("blue", "white", "red"))(200),
  addrect = 3,        
  tl.cex = 0.9,
  cl.cex = 0.9
)
dev.off()


############################################################
# 5. PRINT HIGHLY CORRELATED PAIRS
############################################################
cat("\n>>> Checking for highly correlated pairs...\n")

threshold <- 0.90
high_cor_idx <- which(
  abs(corr_matrix) > threshold &
    row(corr_matrix) != col(corr_matrix), 
  arr.ind = TRUE
)

# Keep only upper triangle to avoid printing each pair twice
high_cor_idx <- high_cor_idx[high_cor_idx[, 1] < high_cor_idx[, 2], , drop = FALSE]

if(nrow(high_cor_idx) == 0){
  cat("No variable pairs have correlation above", threshold, "\n")
} else {
  cat("Variable pairs with correlation above", threshold, ":\n")
  for(i in seq_len(nrow(high_cor_idx))){
    var1 <- rownames(corr_matrix)[high_cor_idx[i, 1]]
    var2 <- colnames(corr_matrix)[high_cor_idx[i, 2]]
    cor_val <- corr_matrix[var1, var2]
    cat(sprintf("  %s vs %s: %.3f\n", var1, var2, cor_val))
  }
}

############################################################
# 6. AUTOMATICALLY REMOVE HIGHLY CORRELATED VARIABLES
############################################################
cat("\n>>> Identifying columns to remove with findCorrelation()...\n")
highCorr <- findCorrelation(corr_matrix, cutoff = threshold)

if(length(highCorr) > 0){
  cat("Removing these variables due to correlation above", threshold, ":\n")
  cat(predictor_vars[highCorr], sep = "\n")
  
  # Subset to keep only uncorrelated variables
  predictor_vars_reduced <- predictor_vars[-highCorr]
  
  cat("\nRemaining predictor vars:\n")
  print(predictor_vars_reduced)
  
  # Create reduced analysis_df
  analysis_df_reduced <- analysis_df %>%
    dplyr::select(
      id,
      all_of(outcome_var),
      all_of(predictor_vars_reduced)
    )
} else {
  cat("No variables exceed the correlation cutoff of", threshold, "\n")
  predictor_vars_reduced <- predictor_vars
  analysis_df_reduced <- analysis_df
}

############################################################
# 7. FIT LINEAR MODEL + CHECK FOR ALIASING
############################################################
cat("\n>>> Fitting linear model with reduced predictors...\n")

# IMPORTANT: Remove `id` from the model formula. We don't want `id` as a predictor.
analysis_df_reduced_for_model <- analysis_df_reduced %>%
  dplyr::select(-id)

lm_model_reduced <- lm(
  as.formula(paste0(outcome_var, " ~ .")), 
  data = analysis_df_reduced_for_model
)

cat("\n--- Aliased Coefficients Check ---\n")
alias_info <- alias(lm_model_reduced)
print(alias_info)

# If any aliased coefficients remain, remove or merge offending variables
if (length(alias_info$Complete) > 0 || length(alias_info$Partial) > 0) {
  cat("\nWARNING: There are still aliased coefficients.\n")
  cat("Consider removing or combining redundant variables further.\n")
} else {
  cat("\nNo aliased coefficients detected. Computing VIF...\n")
  vif_values <- vif(lm_model_reduced)
  cat("\n>>> VIF Values:\n")
  print(vif_values)
}

cat("\n>>> Script complete. Check outputs above.\n")

