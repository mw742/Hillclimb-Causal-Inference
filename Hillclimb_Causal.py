#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import torch
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pgmpy.estimators import HillClimbSearch, StructureScore
import pgmpy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pickle
import textwrap
print("Which python:", sys.executable)
print("pgmpy version:", pgmpy.__version__)

###############################################################################
# 1. Load & Merge Data for Parental Behavior Factors and Child Externalizing Score
###############################################################################
# Load parental factors data
mh_p_asr = pd.read_csv("mh_p_asr.csv")
mh_p_abcl = pd.read_csv("mh_p_abcl.csv")
phenotype = pd.read_csv("mh_p_cbcl.csv", low_memory=False)

# Load PRS data
prs_data = pd.read_csv(
    "/gpfs/group/ehlers/mwei/ABCD/QC_results_round3/PRS_results_perm_all.best",
    sep=r'\s+',
    header=0
)

event_of_interest = "2_year_follow_up_y_arm_1"

# Define parental factors (from ASR and ABCL) – excluding ID and event columns.
parental_vars = [
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
]

# Child behavior externalizing score (from CBCL)
cbcl_vars = [
    "src_subject_id",
    "eventname",
    "cbcl_scr_syn_external_t"
]

# Filter parental data by the event of interest
mh_p_asr_filtered = mh_p_asr[mh_p_asr["eventname"] == event_of_interest]
mh_p_abcl_filtered = mh_p_abcl[mh_p_abcl["eventname"] == event_of_interest]
filtered = phenotype[phenotype["eventname"] == event_of_interest][cbcl_vars]

# Merge ASR and ABCL data (and then merge with CBCL externalizing score)
parental_df = (
    mh_p_asr_filtered.merge(mh_p_abcl_filtered, on=["src_subject_id", "eventname"], how="inner")
    .merge(filtered, on=["src_subject_id", "eventname"], how="inner")
)

# Merge PRS data (rename IID to src_subject_id)
prs_data_cleaned = prs_data[["IID", "PRS"]].rename(columns={"IID": "src_subject_id"})
parental_df = parental_df.merge(prs_data_cleaned, on="src_subject_id", how="inner")

# Rename CBCL external score column for convenience
parental_df = parental_df.rename(columns={"cbcl_scr_syn_external_t": "ExternalScore"})

###############################################################################
# 2. Drop NaNs on Relevant Columns
###############################################################################
# We require all parental variables, PRS, and ExternalScore to be non-missing
required_cols = parental_vars + ["PRS", "ExternalScore"]
parental_df = parental_df.dropna(subset=required_cols)
print("Shape after dropna on required_cols:", parental_df.shape)

###############################################################################
# 3. Clustering and PCA Aggregation for Dimensionality Reduction of Parental Factors
###############################################################################
def cluster_and_pca(df, varnames, corr_threshold=0.5):
    """
    Clusters variables based on their absolute correlations and aggregates each cluster by PCA (first component).
    
    Args:
      df: DataFrame containing the variables.
      varnames: List of column names to cluster.
      corr_threshold: Variables with an absolute correlation above this threshold are grouped together.
    
    Returns:
      agg_df: A DataFrame with one aggregated feature per cluster (first PCA component or the single variable).
      cluster_dict: A dictionary mapping each cluster label to its member variables.
      pca_results: A dictionary mapping the aggregated feature name to its PCA object (or None if only 1 var).
    """
    # Compute the absolute correlation matrix
    corr = df[varnames].corr().abs()
    # Define a distance matrix (1 - correlation)
    dist = 1 - corr
    # Convert to condensed distance matrix for clustering
    dist_condensed = squareform(dist.values, checks=False)
    # Perform hierarchical clustering using average linkage
    Z = linkage(dist_condensed, method='average')
    # Use fcluster to determine clusters – variables with distance < (1 - corr_threshold) cluster together
    distance_threshold = 1 - corr_threshold
    clusters = fcluster(Z, t=distance_threshold, criterion='distance')
    
    # Build a dictionary mapping cluster labels to variable names
    cluster_dict = {}
    for var, cluster_id in zip(varnames, clusters):
        cluster_dict.setdefault(cluster_id, []).append(var)
    
    agg_features = {}
    pca_results = {}
    for cluster_id, variables in cluster_dict.items():
        # Name the new column based on the group
        new_col_name = f"agg_parental_{cluster_id}"
        if len(variables) > 1:
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(df[variables])
            agg_features[new_col_name] = pc1.flatten()
            pca_results[new_col_name] = pca
        else:
            # If only one variable, no PCA needed
            agg_features[new_col_name] = df[variables[0]]
            pca_results[new_col_name] = None
    agg_df = pd.DataFrame(agg_features, index=df.index)
    return agg_df, cluster_dict, pca_results

# Cluster and aggregate parental factors using PCA
parental_agg_df, parental_clusters, parental_pca_results = cluster_and_pca(parental_df, parental_vars, corr_threshold=0.5)
print("Parental clusters found (cluster_id -> variables):")
for cid, vars_ in parental_clusters.items():
    print(cid, ":", vars_)

# Save the PCA results for later analysis
with open("parental_pca_constraint_results.pkl", "wb") as f:
    pickle.dump(parental_pca_results, f)
print("Saved PCA results to 'parental_pca_constraint_results.pkl'")

# Combine the aggregated parental features with PRS and ExternalScore
analysis_df = pd.concat([parental_df[["src_subject_id", "PRS", "ExternalScore"]],
                           parental_agg_df], axis=1)

###############################################################################
# 3b. Rename the aggregated columns to more meaningful names
###############################################################################
# NOTE: This mapping assumes the cluster labels match exactly based on our previous analysis. Please verify, if use!
rename_dict = {
    # Original -> New Name
    "agg_parental_1": "Parents Alcohol/General Substance Use",
    "agg_parental_2": "Parents Drug Use",
    "agg_parental_3": "Parents Tobacco Use",
    "agg_parental_4": "Parents Overall Externalizing/Internalizing Issues (Spouse Report)",
    "agg_parental_5": "Parents Broad Behavioral & Emotional Dysregulation (Self Report)",
    "agg_parental_6": "Parents Somatic Complaints",
    "agg_parental_7": "Parents Intrusive Behaviors"
}
analysis_df.rename(columns=rename_dict, inplace=True)

###############################################################################
# 4. Standardize Final Data for Causal Discovery
###############################################################################
# We exclude 'src_subject_id' from the final modeling columns.
modeling_cols = [c for c in analysis_df.columns if c != "src_subject_id"]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(analysis_df[modeling_cols])
df_scaled = pd.DataFrame(scaled_data, columns=modeling_cols)
print("Scaled data preview:")
print(df_scaled.head())

###############################################################################
# 5. Causal Discovery using Hill Climb Search with Constraints
###############################################################################
class LinearGaussianBIC(StructureScore):
    """
    A simple linear-Gaussian BIC Score for continuous data.
    Assumes each variable is modeled as a linear regression on its parents.
    """
    def __init__(self, data):
        super(LinearGaussianBIC, self).__init__(data=data, state_names=None)
        self.N = len(self.data)
        
    def local_score(self, variable, parents):
        y = self.data[variable].values
        if len(parents) == 0:
            y_mean = np.mean(y)
            residuals = y - y_mean
            sse = np.sum(residuals ** 2)
            dof = 2  
        else:
            X = self.data[list(parents)].values
            n = len(y)
            X_ = np.column_stack([np.ones(n), X])
            try:
                beta = np.linalg.lstsq(X_, y, rcond=None)[0]
                predictions = X_ @ beta
                residuals = y - predictions
                sse = np.sum(residuals ** 2)
            except np.linalg.LinAlgError:
                return -1e9
            dof = X_.shape[1] + 1
        n = self.N
        rss_over_n = sse / n
        if rss_over_n <= 0:
            return -1e9
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + 1 + np.log(rss_over_n))
        bic_score = log_likelihood - 0.5 * dof * np.log(n)
        return bic_score

#np.random.seed(5) 
score = LinearGaussianBIC(df_scaled)
est = HillClimbSearch(df_scaled)

###############################################################################
# 5b. Define Constraints (White/Black Lists)
###############################################################################
final_nodes = list(df_scaled.columns)

# (a) ExternalScore cannot point to any other node
black_list = []
for node in final_nodes:
    if node != 'ExternalScore':
        black_list.append(('ExternalScore', node))

# (b) PRS should not point to any of the parental aggregated nodes.
# Here, we consider "parental aggregated" to be all columns except PRS & ExternalScore.
parental_nodes = [col for col in final_nodes if col not in ['PRS', 'ExternalScore']]
for p_node in parental_nodes:
    black_list.append(('PRS', p_node))

best_dag = est.estimate(scoring_method=score, black_list=black_list)
print("Learned edges:", best_dag.edges())

###############################################################################
# 6. Plotting the Learned DAG
###############################################################################
G = nx.DiGraph(best_dag.edges())

layout_functions = [
    ("spring_layout",  lambda g: nx.spring_layout(g, k=1.5, seed=42)),
    ("shell_layout",   lambda g: nx.shell_layout(g)),
    ("circular_layout",lambda g: nx.circular_layout(g)),
    ("graphviz_dot",   lambda g: graphviz_layout(g, prog='dot')),
]

for layout_name, layout_func in layout_functions:
    plt.figure(figsize=(10, 8))
    pos = layout_func(G)
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=300)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True,
                           arrowstyle='->', arrowsize=15, connectionstyle='arc3,rad=0.1')
    plt.axis("off")
    plt.title(f"Learned DAG - {layout_name}")
    filename = f"DAG_{layout_name}_constraint.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

###############################################################################
# 7. Save Learned Edges, Grouped Edges, and Adjacency Matrix
###############################################################################
edges = list(best_dag.edges())
edges_df = pd.DataFrame(edges, columns=["source", "target"])
edges_df.to_csv("learned_edges_constraint.csv", index=False)
print("Saved learned edges to 'learned_edges_constraint.csv'")

# Group edges by source
grouped_edges = edges_df.groupby("source")["target"].apply(list).reset_index()
grouped_edges.to_csv("grouped_edges_by_source_constraint.csv", index=False)
print("Saved grouped edges by source to 'grouped_edges_by_source_constraint.csv'")

# Create and save the adjacency matrix
nodes = final_nodes
adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
for source, target in best_dag.edges():
    adj_matrix.loc[source, target] = 1
adj_matrix.to_csv("edges_adjacency_matrix_constraint.csv")
print("Saved edges adjacency matrix to 'edges_adjacency_matrix_constraint.csv'")

###############################################################################
# 8. Visualize and Save the DAG with Highlighted Paths to 'ExternalScore'
###############################################################################

target = "ExternalScore"
if target not in G.nodes():
    print(f"Target '{target}' not found in the graph.")
else:
    # Find all ancestors of the target (any node that can reach 'target')
    ancestors = nx.ancestors(G, target)
    sorted_ancestors = sorted(ancestors)
    all_paths = []
    for source in sorted_ancestors:
        for path in nx.all_simple_paths(G, source=source, target=target):
            all_paths.append(path)
    all_paths_sorted = sorted(all_paths)
    
    # Save the simple paths to CSV
    paths_list = [{"path": " -> ".join(path)} for path in all_paths_sorted]
    paths_df = pd.DataFrame(paths_list)
    paths_df.to_csv("simple_paths_to_ExternalScore_constraint.csv", index=False)
    print("Saved simple paths to 'simple_paths_to_ExternalScore_constraint.csv'")
    
    # Plot with highlighted edges
    pos = graphviz_layout(G, prog='dot')
    plt.figure(figsize=(20, 20))

    # --------------------------
    # 1) WRAP THE LONG LABELS
    # --------------------------
    wrapped_labels = {}
    for node in G.nodes():
        # node is a string, so just wrap it at 20 chars
        wrapped_labels[node] = textwrap.fill(str(node), width=20)

    # Define node color mappings
    node_colors = {}
    yellow_nodes = ['Parents Alcohol/General Substance Use', 'Parents Drug Use', 'Parents Tobacco Use']
    green_nodes = ['Parents Overall Externalizing/Internalizing Issues (Spouse Report)', 'Parents Broad Behavioral & Emotional Dysregulation (Self Report)']
    blue_nodes = ['PRS']
    red_nodes = [target]
    purple_nodes = ['Parents Somatic Complaints']
    light_gray_nodes = ['Parents Intrusive Behaviors']
    
    for node in G.nodes():
        if node in yellow_nodes:
            node_colors[node] = 'yellow'
        elif node in green_nodes:
            node_colors[node] = 'green'
        elif node in blue_nodes:
            node_colors[node] = 'blue'
        elif node in red_nodes:
            node_colors[node] = 'red'
        elif node in purple_nodes:
            node_colors[node] = 'purple'
        elif node in light_gray_nodes:
            node_colors[node] = 'lightgray'
        else:
            node_colors[node] = 'lightblue'

    # Draw nodes with specified colors
    nx.draw_networkx_nodes(G, pos, node_color=[node_colors[node] for node in G.nodes()],
                           node_size=300)

    # -------------------------------------------------------------
    # 2) PASS WRAPPED LABELS TO THE draw_networkx_labels CALL
    # -------------------------------------------------------------
    nx.draw_networkx_labels(G, pos, labels=wrapped_labels,
                            font_size=15,
                            font_weight="bold")

    # Define edge color mappings
    edge_colors = []
    for u, v in G.edges():
        if u in yellow_nodes or v in yellow_nodes:
            edge_colors.append('yellow')
        elif u in green_nodes or v in green_nodes:
            edge_colors.append('green')
        elif u in blue_nodes or v in blue_nodes:
            edge_colors.append('blue')
        elif u in red_nodes or v in red_nodes:
            edge_colors.append('red')
        elif u in purple_nodes or v in purple_nodes:
            edge_colors.append('purple')
        elif u in light_gray_nodes or v in light_gray_nodes:
            edge_colors.append('lightgray')
        else:
            edge_colors.append('lightblue') 

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowstyle='->',
                           arrowsize=20, connectionstyle='arc3,rad=0.1')

    # Highlight paths to the target
    n_paths = len(all_paths_sorted)
    cmap = matplotlib.colormaps["hsv"]
    if n_paths > 1:
        color_list = [cmap(i / (n_paths - 1)) for i in range(n_paths)]
    else:
        color_list = [cmap(0.5)]

    print(f"All simple paths to '{target}':")
    for i, path in enumerate(all_paths_sorted):
        print(" -> ".join(path))
        color = color_list[i % len(color_list)]
        
        for u, v in zip(path[:-1], path[1:]):
            if v in yellow_nodes:
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], edge_color='yellow', arrows=True, 
                    arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', width=3.0
                )
            elif v in green_nodes:
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], edge_color='green', arrows=True, 
                    arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', width=3.0
                )
            elif v in blue_nodes:
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], edge_color='blue', arrows=True, 
                    arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', width=3.0
                )
            elif v in red_nodes:
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], edge_color='red', arrows=True, 
                    arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', width=3.0
                )
            elif v in purple_nodes:
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], edge_color='purple', arrows=True, 
                    arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', width=3.0
                )
            elif v in light_gray_nodes:
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], edge_color='lightgray', arrows=True, 
                    arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', width=3.0
                )
            else:
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], edge_color='lightblue', arrows=True, 
                    arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', width=3.0
                )

    plt.axis("off")
    plt.title(
        f"Highlighted Paths to '{target}'",
        fontdict={'fontsize': 40, 'fontweight': 'bold'}
    )

    plt.tight_layout()
    plt.savefig("highlighted_paths_constraint.png", dpi=300, bbox_inches='tight')
    plt.show()



