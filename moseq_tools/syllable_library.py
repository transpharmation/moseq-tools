# /// script
# requires-python = ">=3.7"
# dependencies = [
#     "click",
#     "joblib",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy",
#     "seaborn",
#     "tqdm",
# ]
# ///
import os
import click
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import joblib
import json
import re


# Compiled regex for version parsing
VERSION_PATTERN = re.compile(r'v(\d+)\.(\d+)')


def parse_version(version_str):
    """
    Parse a simplified version string into a tuple of integers.

    Parameters:
    - version_str: Version string (e.g., "v1.0")

    Returns:
    - Tuple of (major, minor) integers, or (0, 0) if parsing fails
    """
    match = VERSION_PATTERN.match(version_str)
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0)


def load_manifest(output_path):
    """
    Load the library manifest file or return default structure.

    Parameters:
    - output_path: Directory where library files are stored

    Returns:
    - Dictionary with manifest data
    """
    manifest_file = Path(output_path) / "library_manifest.json"
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            return json.load(f)
    return {"versions": [], "original_library": None, "last_updated": None}


def get_library_filenames(version):
    """
    Generate consistent library filenames for a given version.

    Parameters:
    - version: Version string (e.g., "v1.0")

    Returns:
    - Tuple of (csv_filename, joblib_filename)
    """
    return (
        f"library_{version}.csv",
        f"library_{version}_model_objects.joblib"
    )


def compute_tsne_embedding(data, n_components=2, random_state=0, metric="cosine"):
    """
    Compute t-SNE embedding for the given data.
    
    Parameters:
    - data: Input data array (n_samples, n_features)
    - n_components: Number of dimensions for the embedding (default: 2)
    - random_state: Random seed for reproducibility (default: 0)
    - metric: Distance metric to use (default: "cosine")
    
    Returns:
    - t-SNE embedding array (n_samples, n_components)
    """
    tsne = TSNE(n_components=n_components, random_state=random_state, metric=metric, perplexity=30)
    return tsne.fit_transform(data)


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


def plot_tsne_clusters(
    pcs,
    labels,
    output_path,
    title="t-SNE of Clustered Syllables",
    cbar_label="Cluster ID",
    filename=None,
):
    """Plot t-SNE of clustered syllable features."""
    tsne_embedding = compute_tsne_embedding(pcs)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    im = ax.scatter(
        tsne_embedding[:, 0],
        tsne_embedding[:, 1],
        c=labels,
        cmap="viridis",
        s=15,
        alpha=0.8,
        lw=0,
    )
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title=title, xticks=[], yticks=[])
    fig.colorbar(im, ax=ax, label=cbar_label)
    if filename is None:
        filename = "tsne_clusters.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()


def plot_individual_clusters(
    all_pcs,
    all_labels,
    existing_mask,
    output_path,
    title_prefix="Cluster",
):
    """
    Create individual t-SNE plots for clusters that contain new syllables, colored by new vs existing syllables.
    
    Parameters:
    - all_pcs: Combined principal components for all syllables
    - all_labels: Cluster labels for all syllables
    - existing_mask: Boolean mask where True indicates existing syllables, False indicates new syllables
    - output_path: Directory to save the plots
    - title_prefix: Prefix for plot titles
    """
    # Create a new folder for individual cluster plots
    cluster_plots_dir = os.path.join(output_path, "individual_cluster_plots")
    os.makedirs(cluster_plots_dir, exist_ok=True)
    
    # Compute t-SNE on all data
    tsne_embedding = compute_tsne_embedding(all_pcs)
    
    # Get unique cluster labels
    unique_clusters = np.unique(all_labels)
    
    # First, identify which clusters have new datapoints
    clusters_with_new_data = []
    for cluster_id in unique_clusters:
        cluster_mask = all_labels == cluster_id
        # Check if this cluster has any new datapoints (where existing_mask is False)
        if np.any(~existing_mask[cluster_mask]):
            clusters_with_new_data.append(cluster_id)
    
    print("Creating individual plots for {} clusters with new data...".format(len(clusters_with_new_data)))
    
    # Create a plot only for clusters that have new datapoints
    for cluster_id in tqdm(clusters_with_new_data, desc="Generating cluster plots"):
        # Get indices for this cluster
        cluster_mask = all_labels == cluster_id
        
        # Get t-SNE coordinates for this cluster
        cluster_tsne = tsne_embedding[cluster_mask]
        
        # Determine which points are existing vs new
        is_new = ~existing_mask[cluster_mask]  # True for new syllables
        
        # Create the plot
        plt.figure(figsize=(8, 6))

        plt.scatter(
            tsne_embedding[:, 0],
            tsne_embedding[:, 1],
            c="lightgray",
            label="All Syllables",
            alpha=0.3,
            s=10,
        )
        
        # Plot existing syllables
        plt.scatter(
            cluster_tsne[~is_new, 0],
            cluster_tsne[~is_new, 1],
            c="blue",
            label="Existing",
            alpha=0.7,
            s=30,
        )
        
        plt.scatter(
            cluster_tsne[is_new, 0],
            cluster_tsne[is_new, 1],
            c="red",
            label="New",
            alpha=0.7,
            s=30,
        )
        
        plt.title("{} {}".format(title_prefix, cluster_id))
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(
            os.path.join(cluster_plots_dir, "cluster_{}.png".format(cluster_id)),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    
    print("Individual cluster plots saved to: {}".format(cluster_plots_dir))


def cluster_syllables(
    df,
    output_path,
    n_clusters=None,
    algorithm="kmeans",
    seed=0,
    pc_components=20,
    max_clusters=300,
):
    """Cluster syllables based on their features."""
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
        "Explained variance by first {} PCs: {:.0%}".format(pc_components, np.sum(pca.explained_variance_ratio_))
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
            print("Optimal number of clusters determined to be: {}".format(best_k))
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
        df["library_label"] = kmeans.fit_predict(pcs)
    else:
        raise NotImplementedError("Only kmeans clustering is currently supported.")

    return df, pcs, pca, kmeans


def load_and_preprocess_data(comparisons, initial_normalization_method="robust_zscore"):
    """Load and preprocess syllable feature data from multiple experiments."""
    zscore_fun = robust_zscore if initial_normalization_method == "robust_zscore" else zscore
    feature_dfs = []
    for idx, row in comparisons.iterrows():
        feat_df = zscore_fun(pd.read_csv(row["syllable_features_path"], index_col=0))
        feat_df["experiment_name"] = row["experiment_name"]
        feat_df["syllable_features_path"] = row["syllable_features_path"]
        # remove syllable usage from dataframe, since not every syllable will be used the same amount
        if "normalized_frequency" in feat_df.columns:
            feat_df = feat_df.drop(columns=["normalized_frequency"])
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

    feature_dfs.set_index(
        ["experiment_name", "syllable_features_path"], inplace=True, append=True
    )
    print(
        "Loaded features for {} total syllables across {} experiments.".format(
            len(feature_dfs), comparisons['experiment_name'].nunique()
        )
    )

    print("Feature dataframe")
    print(feature_dfs.head())

    return feature_dfs


def get_next_version(output_path, is_initial_library=False):
    """
    Determine the next version for the library using simplified vMAJOR.MINOR format.

    Parameters:
    - output_path: Directory where library files are stored
    - is_initial_library: True if this is from generate-library command

    Returns:
    - Next version string (e.g., "v1.0")
    """
    output_path = Path(output_path)
    manifest_file = output_path / "library_manifest.json"

    # Load existing manifest
    manifest = load_manifest(output_path)

    # Get existing versions
    existing_versions = []
    for version_info in manifest.get("versions", []):
        version_str = version_info.get("version", "")
        parsed = parse_version(version_str)
        if parsed != (0, 0):
            existing_versions.append(parsed)

    # First library ever
    if not existing_versions:
        return "v1.0"

    # Version increment logic
    major, minor = max(existing_versions)

    if is_initial_library:
        # generate-library: increment major version, reset minor to 0
        return f"v{major + 1}.0"
    else:
        # add-to-library: increment minor version
        return f"v{major}.{minor + 1}"


def create_version_manifest(output_path, version_info, is_original_library=False):
    """
    Create or update the version manifest file.

    Parameters:
    - output_path: Directory where library files are stored
    - version_info: Dictionary with version information
    - is_original_library: True if this is the original library from generate-library
    """
    output_path = Path(output_path)
    manifest_file = output_path / "library_manifest.json"

    # Load existing manifest
    manifest = load_manifest(output_path)

    # Add version info
    version_info["created"] = datetime.now().isoformat()
    manifest["versions"].append(version_info)

    # Set original library if this is the first one
    if is_original_library and not manifest.get("original_library"):
        manifest["original_library"] = version_info["version"]

    # Sort versions by creation time
    manifest["versions"].sort(key=lambda x: x.get("created", ""))

    # Update last updated timestamp
    manifest["last_updated"] = datetime.now().isoformat()

    # Save manifest
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("Version manifest updated: {}".format(manifest_file))


def manage_library_versions(output_path, keep_recent=3, archive_folder="archive"):
    """
    Manage library versions by keeping only recent versions in main directory
    and archiving older ones, while preserving the original library.

    Parameters:
    - output_path: Directory where library files are stored
    - keep_recent: Number of recent versions to keep in main directory (excluding original)
    - archive_folder: Name of archive subfolder
    """
    output_path = Path(output_path)
    archive_path = output_path / archive_folder
    archive_path.mkdir(exist_ok=True)

    manifest_file = output_path / "library_manifest.json"

    if not manifest_file.exists():
        print("No manifest file found. Skipping version management.")
        return

    manifest = load_manifest(output_path)

    original_version = manifest.get("original_library")
    versions = manifest.get("versions", [])

    # Separate original library from other versions
    original_files = []
    other_files = []

    for version_info in versions:
        version = version_info.get("version", "")
        csv_file, joblib_file = get_library_filenames(version)
        if version == original_version:
            # Keep original library files in main directory
            original_files.extend([csv_file, joblib_file])
        else:
            other_files.extend([csv_file, joblib_file])

    # Archive older versions (keep only the most recent 'keep_recent' versions)
    if len(other_files) > keep_recent * 2:  # *2 because each version has 2 files
        # Sort files by version (newest first)
        version_files = []
        for file in other_files:
            if file.endswith('.csv'):
                version = file.replace('library_', '').replace('.csv', '')
                version_files.append((version, file))

        version_files.sort(key=lambda x: parse_version(x[0]), reverse=True)

        # Keep the most recent versions, archive the rest
        files_to_keep = set()
        for i, (version, file) in enumerate(version_files[:keep_recent]):
            files_to_keep.add(file)
            files_to_keep.add(file.replace('.csv', '_model_objects.joblib'))

        # Archive older files
        for file in other_files:
            if file not in files_to_keep and (output_path / file).exists():
                archive_file = archive_path / file
                (output_path / file).rename(archive_file)
                print("Archived: {} -> {}".format(file, archive_file))

    print("Version management completed. Original library preserved: {}".format(original_version))


def save_library_objects(
    library_df, pca, pcs, kmeans, output_path, timestamp, normalization_type, is_initial_library=False, **extras
):
    """Save library data and preprocessing objects with semantic versioning."""
    output_path = Path(output_path)

    # Get next semantic version
    version = get_next_version(output_path, is_initial_library)

    # Get filenames for this version
    csv_filename, joblib_filename = get_library_filenames(version)

    # Save library dataframe with semantic version
    library_file = output_path / csv_filename
    library_df.to_csv(str(library_file), index=False)

    # Save preprocessing objects with semantic version
    model_objects_file = output_path / joblib_filename

    model_objects = {
        "pca": pca,
        "pcs": pcs,
        "kmeans": kmeans,
        # this one is for future, in case I change model types
        "clustering": kmeans,
        "normalization_type": normalization_type,
        "version": version,
        "timestamp": timestamp,
        **extras,
    }
    joblib.dump(model_objects, str(model_objects_file))

    print("Library saved to: {}".format(library_file))
    print("PCA and KMeans model objects saved to: {}".format(model_objects_file))
    print("Version: {}".format(version))

    # Create version manifest entry
    version_info = {
        "version": version,
        "library_file": library_file.name,
        "model_objects_file": model_objects_file.name,
        "num_syllables": len(library_df),
        "num_clusters": kmeans.n_clusters,
        "normalization_type": normalization_type,
        "timestamp": timestamp,
        "is_initial_library": is_initial_library
    }
    
    # Add additional info if available
    if "previous_library_file" in extras:
        version_info["previous_library_file"] = extras["previous_library_file"]
    
    create_version_manifest(output_path, version_info, is_initial_library)
    
    # Manage versions (archive older ones, preserve original)
    manage_library_versions(output_path, keep_recent=3)

    return library_file, model_objects_file


def load_library_objects(library_path, clustering_file):
    """Load existing library and preprocessing objects."""
    library_df = pd.read_csv(library_path)
    model_objects = joblib.load(clustering_file)

    print("Model objects loaded:", list(model_objects.keys()))

    pca = model_objects["pca"]
    pcs = model_objects["pcs"]
    kmeans = model_objects["clustering"]

    return library_df, pca, pcs, kmeans, model_objects


def assign_to_existing_clusters(
    new_features_pcs, kmeans, output_path, distance_threshold=0.65
):
    """Assign new syllables to existing clusters or flag for new clusters."""
    # Get distances to all centroids
    distances = kmeans.transform(new_features_pcs)

    min_distances = np.min(distances, axis=1)
    print("Distance range: {:.2f} - {:.2f}".format(min_distances.min(), min_distances.max()))

    # assigned_clusters = np.argmin(distances, axis=1)
    assigned_clusters = kmeans.predict(new_features_pcs)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(
        distances[np.argsort(assigned_clusters)], cmap="cubehelix", interpolation="none"
    )

    plt.colorbar(label="Distance to Centroid")
    fig.savefig(os.path.join(output_path, "distances_heatmap.png"))

    # Flag syllables that are too far from any centroid
    new_cluster_mask = min_distances > distance_threshold

    return assigned_clusters, new_cluster_mask, min_distances


def incremental_kmeans_update(kmeans_model, existing_pcs, new_pcs, new_cluster_mask, distance_threshold=0.65):
    """
    Update a k-means model with new data, potentially adding new clusters.
    
    Parameters:
    - kmeans_model: Pre-trained KMeans model
    - existing_pcs: Existing principal components
    - new_pcs: New data points in PC space
    - new_cluster_mask: Boolean mask indicating which new points need new clusters
    - distance_threshold: Distance threshold for creating new clusters
    
    Returns:
    - Updated KMeans model with potentially more clusters
    - Updated labels for all data points
    """
    # Get current centroids
    centroids = kmeans_model.cluster_centers_
    
    # Predict which existing clusters new data belongs to
    distances = kmeans_model.transform(new_pcs)
    closest_clusters = np.argmin(distances, axis=1)
    
    # Initialize labels for new data
    new_labels = closest_clusters.copy()
    
    # Check if we need to create new clusters
    if np.any(new_cluster_mask):
        # Get the outlier points that need new clusters
        outlier_points = new_pcs[new_cluster_mask]
        
        # Cluster the outliers to determine how many new clusters to create
        # Use a reasonable limit to avoid creating too many clusters
        max_new_clusters = min(5, len(outlier_points))  # Limit to 5 new clusters at most
        
        if max_new_clusters > 1:
            # Use KMeans to cluster the outliers
            outlier_kmeans = KMeans(
                n_clusters=max_new_clusters,
                random_state=42,
                n_init=10
            )
            outlier_kmeans.fit(outlier_points)
            new_centroids = outlier_kmeans.cluster_centers_
            
            # Assign the outlier points to their new clusters
            outlier_labels = outlier_kmeans.predict(outlier_points)
            
            # Update the labels for these points with new cluster IDs
            new_cluster_indices = np.where(new_cluster_mask)[0]
            new_labels[new_cluster_indices] = len(centroids) + outlier_labels
        else:
            # If only one outlier, create a single new cluster
            new_centroids = np.mean(outlier_points, axis=0).reshape(1, -1)
            new_cluster_indices = np.where(new_cluster_mask)[0]
            new_labels[new_cluster_indices] = len(centroids)
        
        # Add new centroids to existing ones
        updated_centroids = np.vstack([centroids, new_centroids])
        
        # Create new KMeans model with updated centroids
        updated_kmeans, all_labels = update_centroids_slightly(
            kmeans_model, existing_pcs, new_pcs, max_iter=1, centroids=updated_centroids
        )
    else:
        
        # Create a new model with the same number of clusters
        updated_kmeans, all_labels = update_centroids_slightly(
            kmeans_model, existing_pcs, new_pcs, max_iter=3
        )
        
    return updated_kmeans, all_labels


def update_centroids_slightly(kmeans_model, existing_pcs, new_pcs, max_iter=3, centroids=None):
    """
    Slightly update existing centroids with new data without creating new clusters.
    
    Parameters:
    - kmeans_model: Pre-trained KMeans model
    - existing_pcs: Existing principal components
    - new_pcs: New data points in PC space
    - max_iter: Maximum number of iterations for centroid updates
    
    Returns:
    - Updated KMeans model with same number of clusters
    - Updated labels for all data points
    """
    # Get current centroids
    if centroids is None:
        centroids = kmeans_model.cluster_centers_
    
    # Combine existing and new data
    combined_pcs = np.vstack([existing_pcs, new_pcs])
    
    # Create a new model with the same number of clusters
    updated_kmeans = KMeans(
        n_clusters=len(centroids),
        init=centroids,
        max_iter=max_iter,  # Allow some optimization but limit it
        n_init=1
    )
    
    updated_kmeans.fit(combined_pcs)
    all_labels = updated_kmeans.predict(combined_pcs)
    
    return updated_kmeans, all_labels


def update_labels_and_model(library_df, new_syllables_df, all_labels, pcs):
    """
    Update labels for both existing and new syllables and return the updated KMeans model.
    
    Parameters:
    - library_df: Existing library dataframe
    - new_syllables_df: New syllables dataframe
    - all_labels: Labels for all data points (existing + new)
    - pcs: Existing principal components
    - kmeans: Updated KMeans model
    
    Returns:
    - Updated library_df
    - Updated new_syllables_df
    - Updated kmeans model
    """
    # Update the labels for new syllables
    new_syllables_df["library_label"] = all_labels[len(pcs):]
    
    # Update existing library labels
    library_df["library_label"] = all_labels[:len(pcs)]
    
    return library_df, new_syllables_df


def get_user_feedback(new_clusters_count):
    """Get user feedback on whether to create new clusters."""
    print(
        "\nFound {} syllables that are far from existing centroids.".format(new_clusters_count)
    )
    response = input("Do you want to create new clusters for these syllables? (y/n): ")
    return response.lower() == "y"


def apply_classifier(df, classifier_path):
    """
    Apply a trained classifier to the syllable dataframe.
    
    Parameters:
    - df: Dataframe containing syllable features
    - classifier_path: Path to the classifier bundle .pkl file
    
    Returns:
    - df: Dataframe with added classification columns
    """
    if classifier_path is None:
        print("No classifier supplied, skipping classification")
        return df

    classifier_path = Path(classifier_path)
    if not classifier_path.exists():
        print(f"Classifier path {classifier_path} does not exist. Skipping classification.")
        return df

    print(f"Loading classifier from {classifier_path}...")
    try:
        bundle = joblib.load(classifier_path)
        
        # Extract components
        syllable_pipeline = bundle.get('syllable_pipeline')
        quality_pipeline = bundle.get('quality_pipeline')
        train_cols = bundle.get('train_cols')
        
        if not all(x is not None for x in [syllable_pipeline, train_cols, quality_pipeline]):
            print("Warning: Classifier bundle missing required components. Skipping.")
            return df
            
        # Check if all training columns exist in df
        missing_cols = set(train_cols) - set(df.columns)
        if missing_cols:
            print(f"Warning: Missing features for classification: {missing_cols}. Skipping.")
            return df
            
        # Run predictions
        print("Running syllable classification...")
        X = df[train_cols]
        
        df['syllable_class'] = syllable_pipeline.predict(X)
        df = df.set_index('syllable_class', append=True)
        print("Added 'syllable_class' column to df.")

        df['syllable_quality'] = quality_pipeline.predict(X)
        df = df.set_index('syllable_quality', append=True)
        print("Added 'syllable_quality' column to df.")
        
        print(df.head())
        
    except Exception as e:
        print(f"Error applying classifier: {e}")
        
    return df


@click.group()
def cli():
    pass


@cli.command()
@click.argument("comparison_files")
@click.option(
    "--output-path",
    type=click.Path(dir_okay=True, file_okay=False),
    default=".",
    help="Output directory for library files (default: current directory)",
)
@click.option(
    "--n-clusters",
    type=int,
    default=None,
    help="Number of clusters to use for KMeans clustering. If not specified, tries to determine optimal number automatically.",
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
    "--initial-normalization-method",
    type=click.Choice(["zscore", "robust_zscore"]),
    default="robust_zscore",
    help="Method to use for initial normalization of features before clustering (default: robust_zscore).",
)
@click.option(
    "--classifier-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional path to a syllable classifier .pkl file.",
)
@click.option(
    "--filter-quality",
    is_flag=True,
    help="Filter out low quality syllables (default: False).",
)
def generate_library(
    comparison_files,
    output_path,
    n_clusters,
    seed,
    pc_components,
    initial_normalization_method,
    classifier_path,
    filter_quality,
):
    """Generate a new syllable library from multiple experiments.

    COMPARISON_FILES: Path to CSV file containing columns:
        - experiment_name: Name of experiment (must be unique)
        - syllable_features_path: Path to CSV file containing syllable features
    
    CLASSIFIER_PATH: Optional path to a classifier bundle to predict syllable types and quality.
    """
    # load the comparison file df
    comparisons = pd.read_csv(comparison_files)
    if comparisons["experiment_name"].nunique() != len(comparisons):
        raise ValueError(
            "Experiment names in comparison file must ALL be unique. Found duplicates. "
            "TIP: add experiment date to experiment for extra uniqueness."
        )
    print("Loaded comparison file:")
    print(comparisons.head(2))

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load and preprocess data
    print("Loading and preprocessing the data")
    feature_dfs = load_and_preprocess_data(comparisons, initial_normalization_method)

    # Apply classifier if provided
    feature_dfs = apply_classifier(feature_dfs, classifier_path)

    if filter_quality and classifier_path is not None:
        print("Filtering out low quality syllables...")
        low_quality_syllables = feature_dfs.query("syllable_quality == 'Low'").copy().index
        feature_dfs = feature_dfs.query("syllable_quality != 'Low'").copy()
    elif filter_quality:
        print("Warning: filter_quality is set to True but no classifier path was provided. Skipping quality filtering.")

    # plot clustermap of feature cross-correlations
    plot_cross_corr(feature_dfs, output_path)
    print("Plotted feature cross-correlation clustermap.")

    # run clustering
    clustered_df, pcs, pca, kmeans = cluster_syllables(
        feature_dfs,
        output_path=output_path,
        n_clusters=n_clusters,
        seed=seed,
        pc_components=pc_components,
    )

    print("Created clustered dataframe:")
    print(clustered_df.head())

    # Create library dataframe with required columns
    clustered_df = clustered_df.reset_index()
    keep_cols = ["experiment_name", "syllable_id", "library_label", "syllable_features_path"]
    if "syllable_class" in clustered_df.columns:
        keep_cols += ["syllable_class", "syllable_quality"]

    library_df = clustered_df[keep_cols]

    # Generate t-SNE visualization
    plot_tsne_clusters(
        pcs, library_df["library_label"], output_path, "t-SNE of Library Syllables"
    )

    # Save library objects
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_library_objects(
        library_df,
        pca,
        pcs,
        kmeans,
        output_path,
        timestamp,
        initial_normalization_method,
        is_initial_library=True,
        low_quality_syllables=locals().get("low_quality_syllables", None),
    )


@cli.command()
@click.argument("new_data_files")
@click.argument("library_file")
@click.argument("clustering_file")
@click.option(
    "--output-path",
    type=click.Path(dir_okay=True, file_okay=False),
    default=".",
    help="Output directory for updated library files (default: current directory)",
)
@click.option(
    "--distance-threshold",
    type=float,
    default=0.65,
    help="Distance threshold for creating new clusters (default: 0.65).",
)
@click.option(
    "--update-centroids",
    type=bool,
    default=True,
    help="Whether to slightly update existing centroids when adding new data (default: True).",
)
@click.option(
    "--classifier-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional path to a syllable classifier bundle .pkl file.",
)
@click.option(
    "--filter-quality",
    is_flag=True,
    help="Filter out low quality syllables (default: False).",
)
def add_to_library(
    new_data_files, library_file, clustering_file, output_path, distance_threshold, update_centroids, classifier_path, filter_quality
):
    """Add new experiments to an existing syllable library.

    NEW_DATA_FILES: Path to CSV file with new experiment data (same format as generate-library)
    LIBRARY_FILE: Path to existing library CSV file
    CLUSTERING_FILE: Path to saved object containing PCA and KMeans objects for maximum reproducibility
    
    This function uses incremental clustering to:
    1. Assign new syllables to existing clusters
    2. Optionally create new clusters for syllables that are far from existing centroids
    3. Slightly update existing centroids to account for new data (if update_centroids=True)
    """
    # Load existing library and objects
    library_df, pca, pcs, kmeans, model_objects = load_library_objects(
        library_file, clustering_file
    )
    initial_normalization_method = model_objects["normalization_type"]

    # Load new data
    new_comparisons = pd.read_csv(new_data_files)
    new_features = load_and_preprocess_data(
        new_comparisons, initial_normalization_method
    )
    
    # Apply classifier if provided
    new_features = apply_classifier(new_features, classifier_path)
    if filter_quality and classifier_path is not None:
        print("Filtering out low quality syllables...")
        low_quality_syllables = new_features.query("syllable_quality == 'Low'").copy().index
        new_features = new_features.query("syllable_quality != 'Low'").copy()
    elif filter_quality:
        print("Warning: filter_quality is set to True but no classifier path was provided. Skipping quality filtering.")

    # Apply existing preprocessing
    new_pcs = pca.transform(new_features.values)
    new_pcs = normalize(new_pcs, axis=1, norm="l2")

    # Assign to existing clusters
    assigned_clusters, new_cluster_mask, distances = assign_to_existing_clusters(
        new_pcs, kmeans, output_path, distance_threshold
    )
    # Create dataframe for new syllables
    new_features = new_features.reset_index()
    keep_cols = ["experiment_name", "syllable_id", "syllable_features_path"]
    if "syllable_class" in new_features.columns:
        keep_cols += ["syllable_class", "syllable_quality"]

    new_syllables_df = new_features[keep_cols]
    new_syllables_df["library_label"] = assigned_clusters
    # Handle new clusters using incremental clustering
    new_clusters_count = np.sum(new_cluster_mask)
    print("Found {} syllables that exceed the distance threshold of {}".format(new_clusters_count, distance_threshold))

    if new_clusters_count > 0:
        new_syllables_df['cluster_candidate'] = new_cluster_mask
        filt_df = new_syllables_df.loc[new_syllables_df['cluster_candidate'], ['experiment_name', 'syllable_id']]
        msk = filt_df['syllable_id'].isin([30, 50, 42, 79])
        print("Has these sylls", msk.sum())
        print(filt_df[msk])
    
    if update_centroids:
        if new_clusters_count > 0 and get_user_feedback(new_clusters_count):
            print("Creating new clusters using incremental clustering...")
            # Use incremental clustering to update the model
            kmeans, all_labels = incremental_kmeans_update(
                kmeans, pcs, new_pcs, new_cluster_mask, distance_threshold
            )
        else:
            kmeans, all_labels = update_centroids_slightly(
                kmeans, pcs, new_pcs, max_iter=2
            )
            
        # Update labels and model using helper function
        library_df, new_syllables_df = update_labels_and_model(
            library_df, new_syllables_df, all_labels, pcs
        )
            
        print("Updated KMeans model with {} total clusters".format(kmeans.n_clusters))
    else:
        # Keep original centroids and assignments
        print("No new clusters needed and centroid updates disabled. Keeping original assignments.")

    # Combine old and new library
    updated_library = pd.concat([library_df, new_syllables_df], ignore_index=True)
    
    # Generate visualizations
    os.makedirs(output_path, exist_ok=True)

    # Combined t-SNE plot
    all_pcs = np.vstack([pcs, new_pcs])
    all_labels = updated_library["library_label"]

    # Plot each cluster individually, coloring by existing vs new syllables
    existing_mask = np.concatenate(
        [np.ones(len(library_df), dtype=bool), np.zeros(len(new_syllables_df), dtype=bool)]
    )
    plot_tsne_clusters(
        all_pcs,
        ~existing_mask,  # Invert to show new=1, existing=0
        output_path,
        "t-SNE of Updated Library",
        cbar_label="0=Existing, 1=New",
        filename="tsne_updated_library.png",
    )
    
    # Also plot the actual cluster assignments
    plot_tsne_clusters(
        all_pcs,
        all_labels,
        output_path,
        "t-SNE of Updated Library with Cluster Labels",
        cbar_label="Cluster ID",
        filename="tsne_updated_library_clusters.png",
    )

    plot_individual_clusters(
        all_pcs,
        all_labels,
        existing_mask,
        output_path,
        title_prefix="Cluster",
    )
    
    # Save updated library
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_library_objects(
        updated_library,
        pca,
        all_pcs,
        kmeans,
        output_path,
        timestamp,
        initial_normalization_method,
        is_initial_library=False,
        previous_library_file=os.path.basename(library_file),
        low_quality_syllables=locals().get("low_quality_syllables", None),
    )


@cli.command()
@click.option(
    "--output-path",
    type=click.Path(dir_okay=True, file_okay=False),
    default=".",
    help="Directory containing library files (default: current directory)",
)
@click.option(
    "--show-archived",
    is_flag=True,
    help="Show archived versions as well",
)
def list_versions(output_path, show_archived):
    """List all library versions and their information."""
    output_path = Path(output_path)
    manifest_file = output_path / "library_manifest.json"

    if not manifest_file.exists():
        print("No library manifest found. No versions to display.")
        return

    manifest = load_manifest(output_path)

    original_version = manifest.get("original_library")
    versions = manifest.get("versions", [])

    print("\nLibrary Versions in: {}".format(output_path))
    print("=" * 60)

    if original_version:
        print("Original Library: {}".format(original_version))

    if not versions:
        print("No versions found.")
        return

    print("\nVersion History:")
    print("-" * 60)

    for version_info in versions:
        version = version_info.get("version", "Unknown")
        created = version_info.get("created", "Unknown")
        num_syllables = version_info.get("num_syllables", "Unknown")
        num_clusters = version_info.get("num_clusters", "Unknown")
        is_initial = version_info.get("is_initial_library", False)

        status = "Original" if is_initial else "Update"
        if version == original_version:
            status = "Original (Preserved)"

        print("Version: {} ({})".format(version, status))
        print("  Created: {}".format(created))
        print("  Syllables: {}".format(num_syllables))
        print("  Clusters: {}".format(num_clusters))

        if "previous_library_file" in version_info:
            print("  Previous: {}".format(version_info['previous_library_file']))

        print()

    # Check for archived versions
    archive_path = output_path / "archive"
    if show_archived and archive_path.exists():
        archived_files = list(archive_path.glob("library_*.csv"))
        if archived_files:
            print("Archived Versions ({} files):".format(len(archived_files)))
            print("-" * 60)
            for file in sorted(archived_files):
                version = file.name.replace("library_", "").replace(".csv", "")
                print("  {}".format(version))
        else:
            print("No archived versions found.")


@cli.command()
@click.argument("version")
@click.option(
    "--output-path",
    type=click.Path(dir_okay=True, file_okay=False),
    default=".",
    help="Directory containing library files (default: current directory)",
)
def restore_version(version, output_path):
    """Restore a specific version from archive to main directory."""
    output_path = Path(output_path)
    archive_path = output_path / "archive"

    # Ensure version starts with 'v' if not provided
    if not version.startswith('v'):
        version = "v{}".format(version)

    # Files to restore
    csv_file, joblib_file = get_library_filenames(version)

    # Check if files exist in archive
    csv_archive = archive_path / csv_file
    joblib_archive = archive_path / joblib_file

    if not csv_archive.exists() or not joblib_archive.exists():
        print("Version {} not found in archive.".format(version))
        return

    # Move files back to main directory
    try:
        csv_archive.rename(output_path / csv_file)
        joblib_archive.rename(output_path / joblib_file)
        print("Restored version {} to main directory.".format(version))
    except Exception as e:
        print("Error restoring version {}: {}".format(version, e))


if __name__ == "__main__":
    cli()
