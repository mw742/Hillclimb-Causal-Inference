###############################################################################
# Fully reproducible code that calculates direct, indirect, and total effects
# and saves them (including original variable names) to a single CSV.
###############################################################################

# 0. Load Required Libraries
###############################################################################
library(tidyverse)
library(lavaan)
library(semPlot)

###############################################################################
# 1. Data Import and Preparation
###############################################################################

mh_p_asr <- read_csv("mh_p_asr.csv")
mh_p_abcl <- read_csv("mh_p_abcl.csv")
adhd_phenotype <- read_csv("mh_p_cbcl.csv")

prs_data <- read.table(
  "/gpfs/group/ehlers/mwei/ABCD/QC_results_round3/PRS_results_perm_all.best", 
  header = TRUE
)

# Define event of interest
event_of_interest <- "2_year_follow_up_y_arm_1"

# Define parental variables (from ASR and ABCL)
parental_vars <- c(
  "asr_scr_aggressive_t",
  "asr_scr_antisocial_t",
  "asr_scr_avoidant_t",
  "asr_scr_anxdep_t",
  "asr_scr_depress_t",
  "asr_scr_intrusive_t",
  "asr_scr_rulebreak_t",
  "asr_scr_somaticpr_t",
  "asr_scr_attention_t",
  "asr_scr_adhd_t",
  "asr_scr_inattention_t",
  "asr_scr_hyperactive_t",
  "asr_scr_external_t",
  "asr_scr_internal_t",
  "asr_scr_totprob_t",
  "abcl_scr_prob_aggressive_t",
  "abcl_scr_prob_rulebreak_t",
  "abcl_scr_prob_external_t",
  "abcl_scr_prob_internal_t",
  "abcl_scr_prob_total_t",
  "abcl_scr_sub_use_alcohol_t",
  "abcl_scr_sub_use_drugs_t",
  "abcl_scr_sub_use_t_mean",
  "abcl_scr_sub_use_tobacco_t"
)

# Define the CBCL externalizing variable (keep ID and event columns)
cbcl_vars <- c("src_subject_id", "eventname", "cbcl_scr_syn_external_t")

# Filter data for the event of interest
mh_p_asr_filtered <- mh_p_asr %>% 
  filter(eventname == event_of_interest)

mh_p_abcl_filtered <- mh_p_abcl %>% 
  filter(eventname == event_of_interest)

adhd_filtered <- adhd_phenotype %>%
  filter(eventname == event_of_interest) %>%
  select(all_of(cbcl_vars))

# Merge ASR, ABCL, and CBCL data
parental_df <- mh_p_asr_filtered %>%
  inner_join(mh_p_abcl_filtered, by = c("src_subject_id", "eventname")) %>%
  inner_join(adhd_filtered,         by = c("src_subject_id", "eventname"))

# Merge PRS data (rename IID to src_subject_id)
prs_data_cleaned <- prs_data %>%
  select(IID, PRS) %>%
  rename(src_subject_id = IID)

parental_df <- parental_df %>%
  inner_join(prs_data_cleaned, by = "src_subject_id")

# Rename CBCL external score column for convenience
parental_df <- parental_df %>%
  rename(ExternalScore = cbcl_scr_syn_external_t)

# Drop rows with any missing values in the required columns
required_cols <- c(parental_vars, "PRS", "ExternalScore")
parental_df <- parental_df %>% drop_na(all_of(required_cols))
cat("After drop_na, dimensions:", dim(parental_df), "\n")

###############################################################################
# 2. Clustering and PCA Aggregation for Parental Factors
###############################################################################
cluster_and_pca <- function(df, varnames, corr_threshold = 0.5) {
  # Compute correlation
  corr <- cor(df[, varnames], use = "pairwise.complete.obs")
  corr_abs <- abs(corr)
  
  # Hierarchical clustering
  dist_matrix <- as.dist(1 - corr_abs)
  hc <- hclust(dist_matrix, method = "average")
  clusters <- cutree(hc, h = 1 - corr_threshold)
  
  # Build a dictionary: cluster_id -> variable vector
  cluster_dict <- split(varnames, clusters)
  
  # For each cluster, either do PCA (if >1 var) or keep original
  agg_features <- list()
  pca_results <- list()
  
  for(cluster_id in names(cluster_dict)) {
    variables <- cluster_dict[[cluster_id]]
    new_col_name <- paste0("agg_parental_", cluster_id)
    if (length(variables) > 1) {
      pca <- prcomp(df[, variables], center = TRUE, scale. = TRUE)
      pc1 <- pca$x[, 1]
      agg_features[[new_col_name]] <- pc1
      pca_results[[new_col_name]] <- pca
    } else {
      agg_features[[new_col_name]] <- df[[variables]]
      pca_results[[new_col_name]] <- NULL
    }
  }
  
  agg_df <- as.data.frame(agg_features)
  list(agg_df = agg_df, cluster_dict = cluster_dict, pca_results = pca_results)
}

cluster_results <- cluster_and_pca(parental_df, parental_vars, corr_threshold = 0.5)
parental_agg_df <- cluster_results$agg_df
parental_clusters <- cluster_results$cluster_dict
parental_pca_results <- cluster_results$pca_results

cat("Parental clusters (cluster_id -> variables):\n")
print(parental_clusters)

saveRDS(parental_pca_results, file = "parental_pca_constraint_results.rds")
cat("Saved PCA results to 'parental_pca_constraint_results.rds'\n")

# Combine the newly aggregated columns with the main columns
analysis_df <- cbind(
  parental_df %>% select(src_subject_id, PRS, ExternalScore),
  parental_agg_df
)

###############################################################################
# 3. Rename Aggregated Columns to Syntactic Names
###############################################################################
rename_dict <- c(
  "agg_parental_1" = "Parents_Alcohol_General_Substance_Use",
  "agg_parental_2" = "Parents_Drug_Use",
  "agg_parental_3" = "Parents_Tobacco_Use",
  "agg_parental_4" = "Parents_Overall_Externalizing_Internalizing_Issues_Spouse_Report", 
  "agg_parental_5" = "Parents_Broad_Behavioral_And_Emotional_Dysregulation_Self_Report",
  "agg_parental_6" = "Parents_Somatic_Complaints",
  "agg_parental_7" = "Parents_Intrusive_Behaviors"
)

for (old_name in names(rename_dict)) {
  if (old_name %in% names(analysis_df)) {
    new_name <- rename_dict[[old_name]]
    names(analysis_df)[names(analysis_df) == old_name] <- new_name
  }
}

###############################################################################
# 4. Standardize Data for Modeling
###############################################################################
modeling_cols <- setdiff(names(analysis_df), "src_subject_id")
scaled_data <- scale(analysis_df[, modeling_cols])
df_scaled <- as.data.frame(scaled_data)
cat("Preview of standardized data:\n")
print(head(df_scaled))

# Also keep the subject ID in a separate vector or data frame
df_final <- cbind(src_subject_id = analysis_df$src_subject_id, df_scaled)

###############################################################################
# 5. Read the CSV of Paths and Build the Full Model
###############################################################################
paths_df <- read_csv("simple_paths_to_ExternalScore_constraint.csv")

library(stringr)

clean_varname <- function(x) {
  x %>%
    str_replace_all("/", "_") %>%         # Turn slash into underscore
    str_replace_all("\\(", " ") %>%       # Replace "(" with a space
    str_replace_all("\\)", " ") %>%       # Replace ")" with a space
    str_replace_all("&", "And") %>%       # Replace '&' with 'And'
    str_squish() %>%                      # Remove any extra spaces
    str_replace_all(" ", "_")             # Finally, replace spaces with underscores
}

# Parse each path from "X -> M -> Y" into c("X","M","Y"), cleaning names
parse_and_clean_path <- function(path_string) {
  raw_vars <- str_split(path_string, " -> ", simplify = TRUE)
  raw_vars <- as.vector(raw_vars)
  map_chr(raw_vars, clean_varname)
}

# 5a) Identify all unique direct edges from the path CSV
edges <- paths_df %>%
  mutate(var_chain = map(path, parse_and_clean_path)) %>%
  mutate(edge_pairs = map(var_chain, ~ {
    # If a path is X->M->Y->..., we break it into edges:
    # X->M, M->Y, ...
    if(length(.x) < 2) return(NULL)
    tibble(rhs = .x[-length(.x)], lhs = .x[-1])
  })) %>%
  select(edge_pairs) %>%
  unnest(cols = edge_pairs) %>%
  distinct()

# Label each edge with a parameter name, e.g. p_X_Y
edges <- edges %>%
  mutate(param_label = paste0("p_", rhs, "_", lhs))

# Build the direct paths in lavaan syntax: "lhs ~ param_label*rhs"
edge_lines <- edges %>%
  mutate(model_line = paste0(lhs, " ~ ", param_label, "*", rhs)) %>%
  pull(model_line)

direct_model_spec <- paste(edge_lines, collapse = "\n")

# 5b) Define each chain's indirect parameter (for interpretation)
# We already have edges with param_label. For each path in the CSV,
# we build an indirect effect label: Ind_X_..._Final, = product of edges.
build_product_string <- function(chain_vars, edges_df) {
  param_labels <- c()
  for(i in seq_len(length(chain_vars) - 1)) {
    from <- chain_vars[i]
    to   <- chain_vars[i + 1]
    row_found <- edges_df %>%
      filter(rhs == from, lhs == to)
    if(nrow(row_found) == 1) {
      param_labels <- c(param_labels, row_found$param_label)
    } else {
      return("")  # Edge not found
    }
  }
  paste(param_labels, collapse = " * ")
}

path_params <- paths_df %>%
  mutate(var_chain = map(path, parse_and_clean_path)) %>%
  rowwise() %>%
  mutate(
    param_name = paste0("Ind_", paste(var_chain, collapse = "_")),
    product_str = build_product_string(var_chain, edges)
  ) %>%
  ungroup() %>%
  filter(product_str != "") %>%
  mutate(param_line = paste0(param_name, " := ", product_str))

indirect_model_spec <- paste(path_params$param_line, collapse = "\n")

# 5c) Define total effects for each variable that eventually leads to ExternalScore
# If variable X directly affects ExternalScore, that path is p_X_ExternalScore.
# Plus any indirect paths that start with X and end with ExternalScore.
variables_start <- edges$rhs %>% unique()
external_var <- "ExternalScore"

# We'll see which variables eventually lead to ExternalScore in the path CSV
# We can do this by checking in 'path_params' which have .x[1] = a variable
# and .x[last] = ExternalScore, or any chain that starts with X and ends with ExternalScore.
# Then sum up the direct param (if it exists) + all relevant Ind_X_..._ExternalScore.
total_lines <- c()

for (v in variables_start) {
  # Check if there's a direct path from v -> ExternalScore
  direct_label <- NA
  direct_row <- edges %>%
    filter(rhs == v, lhs == external_var)
  if(nrow(direct_row) == 1) {
    direct_label <- direct_row$param_label
  }
  
  # Collect all Ind_ param_labels that start with v and end with ExternalScore
  # Those are the path_params whose 'var_chain' starts with v, ends with ExternalScore
  relevant_paths <- path_params %>%
    filter(
      map_lgl(var_chain, ~ .x[1] == v && .x[length(.x)] == external_var)
    )
  
  # If neither direct nor indirect exist, skip
  if(nrow(relevant_paths) == 0 && is.na(direct_label)) {
    next
  }
  
  # total effect param name
  total_param_name <- paste0("Tot_", v, "_", external_var)
  
  sum_str <- c()
  if(!is.na(direct_label)) {
    sum_str <- c(sum_str, direct_label)
  }
  if(nrow(relevant_paths) > 0) {
    sum_str <- c(sum_str, relevant_paths$param_name)
  }
  
  if(length(sum_str) == 0) next
  
  total_line <- paste0(total_param_name, " := ", paste(sum_str, collapse = " + "))
  total_lines <- c(total_lines, total_line)
}

total_model_spec <- paste(total_lines, collapse = "\n")

# 5d) Combine everything into one big model
full_model_spec <- paste(
  "# Direct paths:",
  direct_model_spec,
  "",
  "# Indirect definitions:",
  indirect_model_spec,
  "",
  "# Total effects definitions:",
  total_model_spec,
  sep = "\n"
)

cat("Full model specification:\n")
cat(full_model_spec, "\n")

###############################################################################
# 6. Fit the SEM with bootstrap
###############################################################################
sem_fit <- sem(full_model_spec, data = df_final, se = "boot", bootstrap = 500)
summary(sem_fit, standardized = TRUE, fit.measures = TRUE, ci = TRUE)

###############################################################################
# 7. Extract and Save Results (Direct, Indirect, Total) to CSV
###############################################################################
# We'll retrieve all parameter estimates (including user-defined ':=' parameters).
all_params <- parameterEstimates(sem_fit, standardized = TRUE, ci = TRUE)

# We'll create columns mapping the cleaned variable names back to "original" names,
# where possible. For the aggregated factors, we store the cluster membership.
###############################################################################
# Build a dictionary for main variables
clean_to_original <- list(
  "ExternalScore" = "cbcl_scr_syn_external_t",
  "PRS" = "PRS"
)

# For each aggregated factor we created, attach the full membership of original vars:
for (cluster_id in names(parental_clusters)) {
  agg_name <- paste0("agg_parental_", cluster_id)
  # If we renamed it, see rename_dict:
  final_name <- if (agg_name %in% names(rename_dict)) {
    rename_dict[[agg_name]]
  } else {
    agg_name
  }
  # The original variables in this cluster
  vars_in_cluster <- parental_clusters[[cluster_id]]
  # Store it in the dictionary
  clean_to_original[[final_name]] <- paste(vars_in_cluster, collapse = ", ")
}

# Also add any renamed aggregated columns if not in the dictionary
for (clean_name in names(rename_dict)) {
  # If it wasn't in the cluster dictionary for some reason:
  if (! rename_dict[[clean_name]] %in% names(clean_to_original)) {
    clean_to_original[[ rename_dict[[clean_name]] ]] <- clean_name
  }
}

# A small helper to revert from the cleaned name to original
lookup_original <- function(clean_nm) {
  # If it's in the dictionary, use that
  if (clean_nm %in% names(clean_to_original)) {
    return(clean_to_original[[clean_nm]])
  }
  # Otherwise just return the same string
  clean_nm
}

# We'll augment all_params with two columns: lhs_full, rhs_full
all_params <- all_params %>%
  mutate(
    lhs_full = map_chr(lhs, lookup_original),
    rhs_full = map_chr(rhs, lookup_original)
  )

# Output: We keep everything. The user-defined parameters (op == ":=") won't have
# meaningful "lhs" or "rhs" in the original sense, so those columns can remain as is.
# We'll just store them for clarity.

# Reorder columns for clarity
final_results <- all_params %>%
  select(lhs, rhs, op, label, est, ci.lower, ci.upper, pvalue, std.all,
         lhs_full, rhs_full)

# Write to CSV
write_csv(final_results, "SEM_results_with_direct_indirect_total.csv")
cat("Saved all SEM results (direct, indirect, total) to 'SEM_results_with_direct_indirect_total.csv'\n")

###############################################################################
# End of Script
###############################################################################

