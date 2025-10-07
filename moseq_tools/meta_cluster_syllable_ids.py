"""
This module performs meta-clustering of syllable labels across multiple
experiments. The idea is as follows:

MoSeq is inherently a clustering algorithm, finding motifs of mouse
pose dynamics that are similar within an experiment. However, the
assigned syllable labels are randomly assigned across model runs/experiments,
making it difficult to compare behavior across experiments.

Here, we utilize features of individual syllables to cluster them based on
similarity. The module provides multiple clustering approaches to choose
from, and allows the user to integrate a "ground truth" set of pre-labeled
syllables to guide hyperparameter selection for clustering.

Authored by: @wingillis (Winthrop Gillis; win.gillis@gmail.com)
"""

import os
import click
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform


def zscore(df):
    return (df - df.mean()) / df.std()


def robust_zscore(ser):
    """Compute a robust z-score using the median and MAD on a pandas Series."""
    c = 1.4826
    return (ser - ser.median()) / (c * np.nanmedian(np.abs(ser - ser.median())))


def plot_cross_corr(df, path):
    corr = df.corr()

    g = sns.clustermap(
        corr,
        cmap="RdBu_r",
        center=0,
        figsize=(10, 10),
        cbar_kws={"label": "Correlation"},
    )
    g.ax_heatmap.set(xlabel="Features", ylabel="Features")

    file = os.path.join(path, "feature_cross_correlation_clustermap.png")
    g.savefig(file)
    plt.close()


def plot_syllable_clustermap(df, path):
    # compute the correlation distance between syllables
    dist = pdist(df.values, metric="cosine")
    dist = squareform(dist)

    g = sns.clustermap(
        dist,
        figsize=(10, 10),
        cmap="mako_r",
        cbar_kws={"label": "Cosine Distance"},
    )
    g.ax_heatmap.set(
        xlabel="Syllables Across Experiments", ylabel="Syllables Across Experiments"
    )

    file = os.path.join(path, "syllable_feature_clustermap.png")
    g.savefig(file)
    plt.close()


def cluster_syllables(
    df,
    output_path,
    n_clusters=None,
    algorithm="kmeans",
    seed=0,
    pc_components=20,
    max_clusters=300,
):
    """Cluster syllables based on their features.

    Args:
        df (pd.DataFrame): DataFrame of syllable features, indexed by (experiment_name, syllable_id).
        n_clusters (int, optional): Number of clusters to use for KMeans clustering. If None, tries to determine optimal number automatically.
        algorithm (str, optional): Clustering algorithm to use. Options are 'kmeans' or 'affinity'.
        seed (int, optional): Random seed for clustering and t-SNE.
        pc_components (int, optional): Number of principal components to use for clustering.
        output_path (str, optional): Path to save output plots.

    Returns:
        pd.DataFrame: DataFrame with an additional 'meta_cluster' column indicating the assigned cluster.
    """
    # first, reduce dimensionality with PCA
    pca = PCA(n_components=pc_components)
    pcs = pca.fit_transform(df.values)

    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.savefig(os.path.join(output_path, "pca_explained_variance.png"))
    plt.close()

    pcs = normalize(pcs, axis=1, norm="l2")
    print(
        f"Explained variance by first {pc_components} PCs: {np.sum(pca.explained_variance_ratio_):.0%}"
    )

    # next, cluster the syllables
    if algorithm == "kmeans":
        if n_clusters is None:
            # try to determine optimal number of clusters using silhouette score
            sil_scores = []
            # only allow for a minimum of 20 clusters.
            cluster_range = range(20, min(max_clusters, len(df) // 2), 3)
            for k in tqdm(cluster_range, desc="Finding optimal number of clusters"):
                kmeans = KMeans(n_clusters=k, random_state=seed)
                labels = kmeans.fit_predict(pcs)
                sil = silhouette_score(pcs, labels)
                sil_scores.append(sil)
            best_k = cluster_range[np.argmax(sil_scores)]
            print(f"Optimal number of clusters determined to be: {best_k}")
            n_clusters = best_k

            # plot silhouette scores
            plt.figure()
            plt.plot(cluster_range, sil_scores)
            plt.xlabel("Number of clusters")
            plt.ylabel("Silhouette score")
            plt.title("Silhouette scores for different cluster numbers")
            plt.savefig(os.path.join(output_path, "silhouette_scores.png"))
            plt.close()

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        df["meta_cluster"] = kmeans.fit_predict(pcs)

    elif algorithm == "affinity":
        affinity = AffinityPropagation(affinity="euclidean")
        df["meta_cluster"] = affinity.fit_predict(pcs)
        n_clusters = df["meta_cluster"].nunique()
        print(f"Affinity Propagation found {n_clusters} clusters.")

    # save the clustered syllables
    df.to_csv(os.path.join(output_path, "clustered_syllables.csv"))

    # make a simplified dataframe for reading with columns:
    # experiment_name, syllable_id, meta_cluster
    simple_df = df.reset_index()[["experiment_name", "syllable_id", "meta_cluster"]]
    simple_df.to_csv(
        os.path.join(output_path, "clustered_syllables_simple.csv"), index=False
    )

    return simple_df, pcs


def compare_clusters_with_ground_truth(cluster_df, ground_truth_df, output_path):
    """Compare clustered syllables with ground truth labels.

    Args:
        cluster_df (pd.DataFrame): DataFrame with columns 'experiment_name', 'syllable_id', 'meta_cluster'.
        ground_truth_df (pd.DataFrame): DataFrame with columns 'experiment_name', 'syllable_id', 'ground_truth_label'.
        output_path (str): Path to save output plots.
    """
    merged = pd.merge(
        cluster_df, ground_truth_df, on=["experiment_name", "syllable_id"], how="inner"
    )
    merged = merged.astype({"meta_cluster": int})
    print("Merged clustered syllables with ground truth labels:")
    print(merged.head(2))
    contingency = pd.crosstab(merged["ground_truth_label"], merged["meta_cluster"])
    cluster_size = merged.groupby("meta_cluster").size().sort_values(ascending=False)

    cols = [c for c in cluster_size.index if c in contingency.columns]

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        contingency[cols],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Syllable Label Count"},
    )
    plt.xlabel("Meta Cluster")
    plt.ylabel("Ground Truth Label")
    plt.title("Contingency Table of Meta Clusters vs Ground Truth Labels")
    plt.savefig(os.path.join(output_path, "contingency_table.png"))
    plt.close()

    unique_labels_per_cluster = merged.groupby("meta_cluster")[
        "ground_truth_label"
    ].nunique()

    # filter for large clusters
    cluster_size = cluster_size[cluster_size > 1].sort_values(ascending=False)

    print("Unique labels per cluster - top 5 clusters")
    print(unique_labels_per_cluster.loc[cluster_size.index[:5]])

    print("Cluster uniqueness:")
    uniqueness = 1 / (unique_labels_per_cluster / cluster_size)
    print(uniqueness.loc[cluster_size.index[:5]])

    # print(merged.groupby("meta_cluster")["ground_truth_label"].value_counts())


@click.command()
@click.option("--comparison-files", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--ground-truth-path", type=click.Path(exists=True, dir_okay=False), default=None
)
@click.option(
    "--output-path",
    type=click.Path(dir_okay=True, file_okay=False),
    default=datetime.now().strftime("meta-cluster-%Y%m%d"),
    help="Output directory for clustering results. (default: timestamped meta-cluster directory)",
)
@click.option(
    "--n-clusters",
    type=int,
    default=None,
    help="Number of clusters to use for KMeans clustering. If not specified, tries to determine optimal number automatically (default: None).",
)
@click.option(
    "--clustering-algorithm",
    type=click.Choice(["kmeans", "affinity"]),
    default="kmeans",
    help="Clustering algorithm to use. If 'affinity', n_clusters is ignored (default: 'kmeans').",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    help="Random seed for clustering and t-SNE (default: 0).",
)
@click.option(
    "--pc-components",
    type=int,
    default=20,
    help="Number of principal components to use for clustering (default: 20).",
)
@click.option(
    "--plot-cluster-dendrogram",
    type=bool,
    default=True,
    help="If true, plots a dendrogram of the clustered syllables (default: True).",
)
@click.option(
    "--initial-normalization-method",
    type=click.Choice(["zscore", "robust_zscore"]),
    default="robust_zscore",
    help="Method to use for initial normalization of features before clustering (default: robust_zscore).",
)
def main(
    comparison_files,
    ground_truth_path,
    output_path,
    n_clusters,
    clustering_algorithm,
    seed,
    pc_components,
    plot_cluster_dendrogram,
    initial_normalization_method,
):
    """Run meta-clustering of syllable IDs across multiple experiments.

    Args:

        comparison_files (str): Path to CSV file containing columns:

            - experiment_name: Name of experiment (must be unique from other experiments)

            - syllable_features_path: Path to CSV file containing syllable features for the experiment

        ground_truth_path (str, optional): Path to CSV file containing ground truth syllable labels. Must contain columns:

            - experiment_name: Name of experiment (must match those in comparison_files)

            - syllable_id: Original syllable ID from MoSeq model

            - ground_truth_label: Ground truth label for the syllable (e.g., "groom", "rear", "run", "scrunch", etc.)
    """
    # load the comparison file df
    comparisons = pd.read_csv(comparison_files)
    if comparisons["experiment_name"].nunique() != len(comparisons):
        raise ValueError(
            "Experiment names in comparison file must ALL be unique. Found duplicates."
        )
    print("Loaded comparison file:")
    print(comparisons.head(2))

    # go through the comparison file df, load each of the feature files
    # and add the experiment name as a column
    if initial_normalization_method not in ["zscore", "robust_zscore"]:
        print(
            "Initial normalization type not recognized. No initial normalization applied."
        )
    else:
        print(f"Applying initial normalization: {initial_normalization_method}")
    feature_dfs = []
    for idx, row in comparisons.iterrows():
        feat_df = pd.read_csv(row["syllable_features_path"], index_col=0)
        feat_df["experiment_name"] = row["experiment_name"]
        feature_dfs.append(feat_df)

    # concatenate all the feature dfs and set the experiment name and syllable_id as indices
    feature_dfs = pd.concat(feature_dfs)
    # check for NaNs
    nan_count = feature_dfs.isna().sum()
    nan_count = nan_count[nan_count > 0]
    nan_count.name = "NaN count"
    if len(nan_count) > 0:
        print("Found NaNs in the following columns:")
        print(nan_count)
        print("Removing rows with NaNs. Note that some degree of NaNs is expected.")
        feature_dfs = feature_dfs.dropna()

    feature_dfs.set_index(["experiment_name"], inplace=True, append=True)
    print(
        f"Loaded features for {len(feature_dfs)} total syllables across {comparisons['experiment_name'].nunique()} experiments."
    )

    if initial_normalization_method == "zscore":
        feature_dfs = feature_dfs.groupby("experiment_name").transform(zscore)
    elif initial_normalization_method == "robust_zscore":
        feature_dfs = feature_dfs.groupby("experiment_name").transform(robust_zscore)

    # make sure the output path exists
    os.makedirs(output_path, exist_ok=True)

    # plot clustermap of feature cross-correlations
    plot_cross_corr(feature_dfs, output_path)
    print("Plotted feature cross-correlation clustermap.")

    # plot clustermap of syllable feature similarities - this is ultimately what we will cluster on
    if plot_cluster_dendrogram:
        plot_syllable_clustermap(feature_dfs, output_path)

    # run clustering - check which algorithm to run
    #  - select cluster number from click.option (i.e., auto/None, or specific number)
    simple_df, pcs = cluster_syllables(
        feature_dfs,
        n_clusters=n_clusters,
        algorithm=clustering_algorithm,
        seed=seed,
        pc_components=pc_components,
        output_path=output_path,
    )

    # plot t-sne of clustered syllable features
    tsne = TSNE(n_components=2, random_state=seed)
    tsne_embedding = tsne.fit_transform(pcs)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    im = ax.scatter(
        tsne_embedding[:, 0],
        tsne_embedding[:, 1],
        c=simple_df["meta_cluster"],
        cmap="viridis",
        s=15,
        alpha=0.8,
        lw=0,
    )
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="t-SNE of Clustered Syllables", xticks=[], yticks=[])
    fig.colorbar(im, ax=ax, label="Meta Cluster ID")
    plt.savefig(os.path.join(output_path, "tsne_clustered_syllables.png"))

    if ground_truth_path is not None:
        gt_df = pd.read_csv(ground_truth_path)
        print("Loaded ground truth data:")
        print(gt_df.head())
        print("Comparing clustered syllables with ground truth labels...")
        compare_clusters_with_ground_truth(simple_df, gt_df, output_path)


if __name__ == "__main__":
    main()
