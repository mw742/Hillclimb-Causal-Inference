#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sys
import torch
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pgmpy.estimators import HillClimbSearch, StructureScore
from pgmpy.models import BayesianNetwork
import pgmpy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pickle
from sklearn.utils import resample
import os
import re

# ------------------------------------------------------------------------
# 0. Basic checks
# ------------------------------------------------------------------------
print("Which python:", sys.executable)
print("pgmpy version:", pgmpy.__version__)

# ------------------------------------------------------------------------
# 1. Load & Merge Data for Parental Behavior Factors and Child Externalizing Score
# ------------------------------------------------------------------------
mh_p_asr = pd.read_csv("mh_p_asr.csv")
mh_p_abcl = pd.read_csv("mh_p_abcl.csv")
adhd_phenotype = pd.read_csv("mh_p_cbcl.csv", low_memory=False)

# Load PRS data
prs_data = pd.read_csv(
    "/gpfs/group/ehlers/mwei/ABCD/QC_results_round3/PRS_results_perm_all.best",
    sep=r'\s+',
    header=0
)

event_of_interest = "2_year_follow_up_y_arm_1"

# Define parental factors (from ASR and ABCL) – excluding ID and event columns.
parental_vars = [
    "asr_scr_aggressive_t", "asr_scr_antisocial_t", "asr_scr_avoidant_t", "asr_scr_anxdep_t", "asr_scr_depress_t",
    "asr_scr_intrusive_t", "asr_scr_rulebreak_t", "asr_scr_somaticpr_t", "asr_scr_attention_t", "asr_scr_adhd_t",
    "asr_scr_inattention_t", "asr_scr_hyperactive_t", "asr_scr_external_t", "asr_scr_internal_t", "asr_scr_totprob_t",
    "abcl_scr_prob_aggressive_t", "abcl_scr_prob_rulebreak_t", "abcl_scr_prob_external_t", "abcl_scr_prob_internal_t",
    "abcl_scr_prob_total_t", "abcl_scr_sub_use_alcohol_t", "abcl_scr_sub_use_drugs_t", "abcl_scr_sub_use_t_mean",
    "abcl_scr_sub_use_tobacco_t"
]

# Child behavior externalizing score (from CBCL)
cbcl_vars = ["src_subject_id", "eventname", "cbcl_scr_syn_external_t"]

# Filter parental data by the event of interest
mh_p_asr_filtered = mh_p_asr[mh_p_asr["eventname"] == event_of_interest]
mh_p_abcl_filtered = mh_p_abcl[mh_p_abcl["eventname"] == event_of_interest]
adhd_filtered = adhd_phenotype[adhd_phenotype["eventname"] == event_of_interest][cbcl_vars]

# Merge ASR and ABCL data (and then merge with CBCL externalizing score)
parental_df = (
    mh_p_asr_filtered
    .merge(mh_p_abcl_filtered, on=["src_subject_id", "eventname"], how="inner")
    .merge(adhd_filtered,        on=["src_subject_id", "eventname"], how="inner")
)

# Merge PRS data (rename IID to src_subject_id)
prs_data_cleaned = prs_data[["IID", "PRS"]].rename(columns={"IID": "src_subject_id"})
parental_df = parental_df.merge(prs_data_cleaned, on="src_subject_id", how="inner")

# Rename CBCL external score column for convenience
parental_df = parental_df.rename(columns={"cbcl_scr_syn_external_t": "ExternalScore"})

# 2. Drop NaNs on Relevant Columns
required_cols = parental_vars + ["PRS", "ExternalScore"]
parental_df = parental_df.dropna(subset=required_cols)
print("Shape after dropna on required_cols:", parental_df.shape)

# ------------------------------------------------------------------------
# 3. Clustering and PCA Aggregation for Dimensionality Reduction of Parental Factors
# ------------------------------------------------------------------------
def cluster_and_pca(df, varnames, corr_threshold=0.5):
    """
    Clusters the variables by correlation > corr_threshold and
    runs PCA (n_components=1) within each cluster to get a single aggregated feature.
    """
    corr = df[varnames].corr().abs()
    dist = 1 - corr
    dist_condensed = squareform(dist.values, checks=False)
    Z = linkage(dist_condensed, method='average')
    distance_threshold = 1 - corr_threshold
    clusters = fcluster(Z, t=distance_threshold, criterion='distance')
    
    cluster_dict = {}
    for var, cluster_id in zip(varnames, clusters):
        cluster_dict.setdefault(cluster_id, []).append(var)
    
    agg_features = {}
    pca_results = {}
    for cluster_id, variables in cluster_dict.items():
        new_col_name = f"agg_parental_{cluster_id}"
        if len(variables) > 1:
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(df[variables])
            agg_features[new_col_name] = pc1.flatten()
            pca_results[new_col_name] = pca
        else:
            # Only one variable, no PCA needed
            agg_features[new_col_name] = df[variables[0]]
            pca_results[new_col_name] = None
    
    agg_df = pd.DataFrame(agg_features, index=df.index)
    return agg_df, cluster_dict, pca_results

parental_agg_df, parental_clusters, parental_pca_results = cluster_and_pca(
    parental_df, parental_vars, corr_threshold=0.5
)
print("Parental clusters found (cluster_id -> variables):")
for cid, vars_ in parental_clusters.items():
    print(cid, ":", vars_)

# Save PCA objects if needed
with open("parental_pca_constraint_results.pkl", "wb") as f:
    pickle.dump(parental_pca_results, f)
print("Saved PCA results to 'parental_pca_constraint_results.pkl'")

# Create the final analysis DataFrame
analysis_df = pd.concat(
    [parental_df[["src_subject_id", "PRS", "ExternalScore"]], parental_agg_df],
    axis=1
)

# Rename PCA-based columns to something more interpretable
rename_dict = {
    "agg_parental_1": "Parents Alcohol/General Substance Use",
    "agg_parental_2": "Parents Drug Use",
    "agg_parental_3": "Parents Tobacco Use",
    "agg_parental_4": "Parents Overall Externalizing/Internalizing Issues (Spouse Report)",
    "agg_parental_5": "Parents Broad Behavioral & Emotional Dysregulation (Self Report)",
    "agg_parental_6": "Parents Somatic Complaints",
    "agg_parental_7": "Parents Intrusive Behaviors"
}
analysis_df.rename(columns={k: v for k, v in rename_dict.items() if k in analysis_df.columns}, inplace=True)

# ------------------------------------------------------------------------
# 4. Scale the data
# ------------------------------------------------------------------------
modeling_cols = [c for c in analysis_df.columns if c != "src_subject_id"]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(analysis_df[modeling_cols])
df_scaled = pd.DataFrame(scaled_data, columns=modeling_cols)
print("Scaled data preview:")
print(df_scaled.head())

# ------------------------------------------------------------------------
# 5a. Custom BIC Scorer (LinearGaussianBIC)
# ------------------------------------------------------------------------
class LinearGaussianBIC(StructureScore):
    def __init__(self, data):
        super(LinearGaussianBIC, self).__init__(data=data, state_names=None)
        self.N = len(self.data)
        
    def local_score(self, variable, parents):
        y = self.data[variable].values
        n = len(y)
        
        if len(parents) == 0:
            y_mean = np.mean(y)
            residuals = y - y_mean
            sse = np.sum(residuals ** 2)
            # 2 degrees of freedom -> intercept + variance
            dof = 2  
        else:
            X = self.data[list(parents)].values
            X_ = np.column_stack([np.ones(n), X])
            try:
                beta = np.linalg.lstsq(X_, y, rcond=None)[0]
                predictions = X_ @ beta
                residuals = y - predictions
                sse = np.sum(residuals ** 2)
            except np.linalg.LinAlgError:
                return -1e9  # big negative to penalize singular designs
            # dof = number_of_parameters + 1 (the variance)
            dof = X_.shape[1] + 1
        
        rss_over_n = sse / n
        if rss_over_n <= 0:
            return -1e9
        
        log_likelihood = -0.5 * n * (
            np.log(2 * np.pi) + 1 + np.log(rss_over_n)
        )
        # BIC penalty
        bic_score = log_likelihood - 0.5 * dof * np.log(n)
        return bic_score

# ------------------------------------------------------------------------
# 5b. Define Constraints (Black List)
# ------------------------------------------------------------------------
final_nodes = list(df_scaled.columns)

# (a) ExternalScore cannot point to any other node
black_list = []
for node in final_nodes:
    if node != 'ExternalScore':
        black_list.append(('ExternalScore', node))

# (b) PRS should not point to any of the parental (aggregated) nodes
parental_nodes = [col for col in final_nodes if col not in ['PRS', 'ExternalScore']]
for p_node in parental_nodes:
    black_list.append(('PRS', p_node))

print("Black list of edges:")
for edge in black_list:
    print(edge)

# ------------------------------------------------------------------------
# 6. Function to Store Edge Weights after DAG structure is found
# ------------------------------------------------------------------------
def store_edge_weights(dag, data):
    """
    For each node, fit a linear regression of node ~ its parents.
    Then store the regression coefficient on each parent->node edge as 'weight'.
    """
    for node in dag.nodes():
        parents = list(dag.predecessors(node))
        if len(parents) == 0:
            # no parents, no coefficients
            continue
        
        # Fit OLS: node ~ parents
        X = data[parents].values
        y = data[node].values
        n = len(y)
        X_ = np.column_stack([np.ones(n), X])
        
        try:
            beta = np.linalg.lstsq(X_, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # If singular, skip or set weights to 0
            beta = np.zeros(X_.shape[1])
        
        intercept = beta[0]
        coefs = beta[1:]
        
        # Assign each parent's coefficient
        for parent, w in zip(parents, coefs):
            dag[parent][node]['weight'] = w
    
    return dag

# ------------------------------------------------------------------------
# 7. Bootstrapping the Model (with black_list option)
# ------------------------------------------------------------------------
def bootstrap(data, n_iterations=1000, black_list=None):
    """
    Repeatedly resample the dataset, learn a DAG structure with given black_list,
    then store the edge weights on each iteration.
    """
    dag_list = []
    for i in range(n_iterations):
        # Resample
        data_resampled = resample(data, replace=True)

        # Fit the structure with our custom linear BIC
        score_resampled = LinearGaussianBIC(data_resampled)
        est_resampled = HillClimbSearch(data_resampled)

        best_model = est_resampled.estimate(
            scoring_method=score_resampled,
            black_list=black_list  # Pass the black_list here
        )

        # Convert to a BayesianNetwork so edges are recognized
        dag_resampled = BayesianNetwork(best_model.edges())
        
        # Make sure all nodes exist in the DAG
        for col in data_resampled.columns:
            if col not in dag_resampled.nodes():
                dag_resampled.add_node(col)

        # Now store regression coefficients as 'weight'
        dag_resampled = store_edge_weights(dag_resampled, data_resampled)

        # Print edges for debugging (optional)
        print(f"Bootstrap iteration {i}, edges found: {list(dag_resampled.edges())}")

        dag_list.append(dag_resampled)
    return dag_list

# ------------------------------------------------------------------------
# 8. Calculate Edge Stats for *All* Edges That Appear
# ------------------------------------------------------------------------
def collect_edge_stats(dag_list, alpha=0.05):
    """
    Gathers all edges that appear in any DAG (at least once).
    For each edge, calculates:
      - frequency of appearance (times edge appears / total # DAGs)
      - bootstrap confidence intervals for the regression 'weight' (lower, upper)
    
    Returns a dict: edge -> (frequency, lower, upper).
    """
    edge_weights = {}
    n_dags = len(dag_list)

    # Collect all weights from all DAGs
    for dag in dag_list:
        for edge in dag.edges():
            if edge not in edge_weights:
                edge_weights[edge] = []
            w = dag[edge[0]][edge[1]].get('weight', None)
            if w is not None:
                edge_weights[edge].append(w)

    # Build results
    results = {}
    for edge, weights in edge_weights.items():
        appearance_count = len(weights)
        freq = appearance_count / n_dags  # times edge appeared / total DAGs

        if appearance_count > 0:
            # Compute bootstrap 95% CI via percentiles
            lower = np.percentile(weights, alpha * 100 / 2)  # e.g. 2.5%
            upper = np.percentile(weights, 100 - alpha * 100 / 2)  # e.g. 97.5%
        else:
            # Shouldn't happen here if it 'appears' at least once,
            # but let's be safe:
            lower, upper = np.nan, np.nan

        results[edge] = (freq, lower, upper)

    return results

# ------------------------------------------------------------------------
# 9. Plot Edge Variability
# ------------------------------------------------------------------------
matplotlib.use('Agg')  # non-interactive backend
def plot_edge_variability(dag_list, save_dir="edge_histograms"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    edge_weights = {}
    for dag in dag_list:
        for edge in dag.edges():
            if edge not in edge_weights:
                edge_weights[edge] = []
            w = dag[edge[0]][edge[1]].get('weight', None)
            if w is not None:
                edge_weights[edge].append(w)
    
    for edge, weights in edge_weights.items():
        if len(weights) == 0:
            continue
        plt.figure()
        plt.hist(weights, bins=30, color='blue', alpha=0.7)
        plt.title(f"Edge {edge}: Distribution of bootstrapped weights\n(n={len(weights)} appearances)")
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        
        safe_edge0 = re.sub(r'[^\w\s-]', '_', edge[0]).replace(" ", "_")
        safe_edge1 = re.sub(r'[^\w\s-]', '_', edge[1]).replace(" ", "_")
        filename = f"edge_{safe_edge0}_to_{safe_edge1}_histogram.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved histogram for {edge} as {filepath}")

# ------------------------------------------------------------------------
# 10. Actually Run the Bootstrapping
# ------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nStarting bootstrap with HillClimbSearch + LinearGaussianBIC ...")
    
    # Number of bootstrap replicates:
    n_iterations = 100 
    
    # Perform bootstrapping
    dag_list = bootstrap(df_scaled, n_iterations=n_iterations, black_list=black_list)

    # Collect stats for all edges that appear at least once
    edge_stats = collect_edge_stats(dag_list, alpha=0.05)

    # Save them to a CSV
    rows = []
    for edge, (freq, lower, upper) in edge_stats.items():
        src, tgt = edge
        rows.append([src, tgt, freq, lower, upper])
    df_stats = pd.DataFrame(rows, columns=["Source", "Target", "Frequency", "CI_lower", "CI_upper"])
    df_stats.sort_values("Frequency", ascending=False, inplace=True)
    df_stats.to_csv("bootstrap_edge_stats.csv", index=False)
    print("\nAll edge statistics saved to 'bootstrap_edge_stats.csv'.")

    # Print them to screen too:
    print("\nEdge statistics (sorted by Frequency):")
    for _, row in df_stats.iterrows():
        print(f"Edge ({row['Source']} -> {row['Target']}): "
              f"freq={row['Frequency']:.2f}, "
              f"CI=({row['CI_lower']:.3f}, {row['CI_upper']:.3f})")

    # Plot edge weight distributions
    print("\nPlotting edge weight distributions ...")
    plot_edge_variability(dag_list, save_dir="edge_histograms")

    print("\nDone.")

