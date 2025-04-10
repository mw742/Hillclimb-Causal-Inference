library(dplyr)
library(car)
library(caret)
library(corrplot)
library(pheatmap)
library(ggplot2)

# Load data
cbcl <- read.csv("../mh_p_cbcl.csv", stringsAsFactors = FALSE)
asr  <- read.csv("../mh_p_asr.csv",  stringsAsFactors = FALSE)
abcl <- read.csv("../mh_p_abcl.csv", stringsAsFactors = FALSE)

# Filter data
cbcl_2year <- subset(cbcl, eventname == "2_year_follow_up_y_arm_1")
asr_2year  <- subset(asr,  eventname == "2_year_follow_up_y_arm_1")
abcl_2year <- subset(abcl, eventname == "2_year_follow_up_y_arm_1")

merged_cbcl_asr <- merge(cbcl_2year, asr_2year, by = "src_subject_id", all = TRUE)
merged_df <- merge(merged_cbcl_asr, abcl_2year, by = "src_subject_id", all = TRUE)

merged_df <- merged_df %>% rename(id = src_subject_id)

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

# Short variable names
# Descriptive full names from the provided table exactly matching the original order
short_names <- c(
  "ASR - Personality Strengths",
  "ASR - Anxious/Depressed",
  "ASR - Withdrawn",
  "ASR - Somatic Complaints",
  "ASR - Thought Problems",
  "ASR - Attention Problems",
  "ASR - Aggressive Behavior",
  "ASR - Rule-breaking Behavior",
  "ASR - Intrusive Behavior",
  "ASR - Internalizing Problems",
  "ASR - Externalizing Problems",
  "ASR - Total Problems",
  "ASR - Depressive Problems",
  "ASR - Anxiety Problems",
  "ASR - Somatic Problems",
  "ASR - Avoidant Personality",
  "ASR - ADHD Problems",
  "ASR - Antisocial Personality",
  "ASR - Inattention Problems",
  "ASR - Hyperactivity Problems",
  "ABCL - Substance Use Tobacco",
  "ABCL - Substance Use Alcohol",
  "ABCL - Substance Use Drugs",
  "ABCL - Substance Use Mean",
  "ABCL - Adaptive Functioning Friends",
  "ABCL - Anxious/Depressed",
  "ABCL - Withdrawn",
  "ABCL - Somatic Complaints",
  "ABCL - Thought Problems",
  "ABCL - Attention Problems",
  "ABCL - Aggressive Behavior",
  "ABCL - Rule-breaking Behavior",
  "ABCL - Intrusive Behavior",
  "ABCL - Internalizing Problems",
  "ABCL - Externalizing Problems",
  "ABCL - Total Problems",
  "ABCL - Critical Items"
)

predictor_vars <- predictor_vars[predictor_vars %in% names(merged_df)]

analysis_df <- merged_df %>%
  dplyr::select(id, all_of(outcome_var), all_of(predictor_vars)) %>%
  na.omit()

corr_matrix <- cor(analysis_df[, predictor_vars], use = "complete.obs")

colnames(corr_matrix) <- short_names
rownames(corr_matrix) <- short_names

write.csv(corr_matrix, file = "corr_matrix.csv", row.names = TRUE)

palette_length <- 100
my_colors <- colorRampPalette(c("blue", "white", "red"))(palette_length)
my_breaks <- seq(-1, 1, length.out = palette_length)

# Heatmap visualization
pdf("pheatmap_heatmap.pdf", width = 16, height = 14)
pheatmap(
  corr_matrix,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  display_numbers = TRUE,
  fontsize_number = 8,
  fontsize_row = 10,
  fontsize_col = 10,
  cellwidth = 20,
  cellheight = 20,
  color = my_colors,
  breaks = my_breaks
)
dev.off()

