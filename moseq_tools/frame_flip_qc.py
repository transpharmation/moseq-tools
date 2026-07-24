#!/usr/bin/env python
import h5py
import click
import shutil
import joblib
import logging
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from toolz import valmap
from tqdm.auto import tqdm
from toolz.curried import get
from scipy.spatial.distance import cdist, correlation

# Constants
MIN_SYLLABLE_DURATION = 0.13
MAX_SYLLABLE_DURATION = 1.0
SIMILARITY_THRESHOLD = 1e-1


def _dict_to_str(d: dict):
    """Convert a dictionary with float values to a string representation."""
    return "\n".join([f"{k}: {v:0.3f}" for k, v in d.items()])


def _run_command(command: List[str], error_msg: str):
    """Run a subprocess command with error handling."""
    logging.info(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_msg}: {e}")
        raise


def flip_frames(files: List[Path], flip_qc_folder: Path):
    for file in tqdm(files, leave=False, desc="Flipping frames", unit="files"):
        logging.info(f"Copying {file} to {flip_qc_folder}")
        new_file_path = flip_qc_folder / file.name
        if new_file_path.exists():
            logging.warning(
                f"{new_file_path} already exists, assuming flipped and skipping"
            )
            continue

        # copy h5 file to flip_qc_folder
        shutil.copy2(file, new_file_path)
        # copy yaml file too
        shutil.copy2(file.with_suffix(".yaml"), new_file_path.with_suffix(".yaml"))

        with h5py.File(new_file_path, "r+") as h5f:
            # flip frames
            frames = h5f["frames"][()]
            flipped_frames = np.flip(frames, axis=1)
            h5f["frames"][:] = flipped_frames

        logging.info(f"Flipped frames in {new_file_path}")


def load_labels(model_file: str):
    """
    Load labels from a MoSeq model file, and filter out the
    -5 starting values.

    Args:
        model_file: Path to the MoSeq model file.

    Returns:
        A dictionary mapping keys to labels.
    """
    mdl = joblib.load(model_file)
    data = dict(zip(mdl["keys"], mdl["labels"]))
    return valmap(lambda v: v[v != -5], data)


def one_hot_encode(labels: np.ndarray, num_classes: int = 100):
    """
    One-hot encode the labels.

    Args:
        labels: Array of labels.
        num_classes: Number of classes for one-hot encoding.

    Returns:
        One-hot encoded array.
    """
    one_hot = np.zeros((len(labels), num_classes), dtype="int8")
    one_hot[np.arange(len(labels)), labels.astype("int16")] = 1
    return one_hot


def compare_syllable_similarity(original_model_file: str, flipped_model_file: str):

    original_labels = load_labels(original_model_file)
    flipped_labels = load_labels(flipped_model_file)
    # sort session UUIDs to ensure consistent and comparable ordering
    flipped_labels = {k: flipped_labels[k] for k in original_labels}
    if set(original_labels.keys()) != set(flipped_labels.keys()):
        raise ValueError(
            "Sessions within the original and flipped datasets do not match. "
            "Ensure both datasets contain the same sessions. "
            "This might mean deleting the flipped frames folder and re-running this step from scratch."
        )

    # 1: compute overall similarity
    overall_similarity = {
        k: (original_labels[k] == flipped_labels[k]).mean() for k in original_labels
    }
    avg_similarity = np.mean(list(overall_similarity.values())) * 100
    logging.info(f"% of matching syllable IDs: {avg_similarity:.2f}")

    # one-hot encode labels for per-syllbale similarity comparison
    original_one_hot = one_hot_encode(np.concatenate(list(original_labels.values())))
    flipped_one_hot = one_hot_encode(np.concatenate(list(flipped_labels.values())))

    # 2: compute per-syllable similarity
    logging.info("Computing per-syllable Jaccard similarity")
    per_syllable_similarity = 1 - cdist(
        original_one_hot.T, flipped_one_hot.T, metric="jaccard"
    )
    # get diagonal elements (i.e., per-syllable similarity)
    within_syllable_similarity = {
        i: d for i, d in enumerate(np.diag(per_syllable_similarity))
    }

    within_syllable_similarity = dict(
        sorted(within_syllable_similarity.items(), key=get(1), reverse=True)
    )

    logging.info("Per-syllable similarity:")
    logging.info(_dict_to_str(within_syllable_similarity))

    return within_syllable_similarity, avg_similarity, per_syllable_similarity


def compute_syllable_aligned_features(df: pd.DataFrame):
    """
    Compute the following syllable-aligned features from the dataframe:
    - syllable count: gives a unique identifier for each syllable instance
    - syllable duration: duration of each syllable in frames
    - syllable duration (s): duration of each syllable in seconds
    - syllable onset aligned time: computed so that the first frame of each
        syllable instance is aligned to 0 seconds.
    - syllable onset aligned angle: computed so that the first frame of each
        syllable instance is aligned to 0 degrees.

    Args:
        df: DataFrame containing the timeseries of extracted features.

    Returns:
        DataFrame with additional columns with syllable-aligned features.
    """
    logging.info("Computing syllable count and duration")
    df["syllable_count"] = df["onset"].cumsum()
    duration = df.query("onset")["syllable index"].diff()
    idx = df.query("onset").index[:-1]
    df.loc[idx, "syllable_duration"] = duration.values[1:]

    df["syllable_duration"] = df["syllable_duration"].ffill()
    fps = 1000 / df.groupby("uuid")["timestamps"].diff().median()
    df["syllable_duration (s)"] = df["syllable_duration"] / fps

    logging.info("Computing syllable onset-aligned time and angle")
    df["onset_aligned_time"] = df.groupby("syllable_count", sort=False)["syllable index"].transform(
        lambda x: (x - x.iloc[0]) / fps
    )
    # angle should already be unwrapped
    df["onset_aligned_angle"] = df.groupby("syllable_count", sort=False)["angle"].transform(
        lambda x: x - x.iloc[:2].mean()
    )

    return df


def compute_kinematic_trajectories(df: pd.DataFrame, max_duration: float = 0.4):
    """
    Compute kinematic trajectories for each syllable.

    Args:
        df: DataFrame containing the timeseries of extracted features.

    Returns:
        DataFrame with kinematic trajectories for each syllable.
    """
    trajectories = df.groupby(["labels (original)", "onset_aligned_time"])[
        ["onset_aligned_angle", "velocity_3d_mm", "height_ave_mm", "length_mm"]
    ].mean()

    trajectories = trajectories[trajectories.index.get_level_values(1) <= max_duration]
    return trajectories


def detrend(df: pd.DataFrame):
    """
    Detrend the DataFrame by removing the min value from each column.

    Args:
        df: DataFrame containing the kinematic trajectories.

    Returns:
        Detrended DataFrame.
    """
    return df - df.min()


def zscore(df: pd.DataFrame):
    """
    Z-score normalize the DataFrame.

    Args:
        df: DataFrame containing the kinematic trajectories.

    Returns:
        Z-score normalized DataFrame.
    """
    return (df - df.mean()) / df.std()


def compute_kinematic_similarity(
    trajectories: pd.DataFrame, indices, confusion_rate
) -> pd.DataFrame:
    correlation_df = []

    for syll_i, syll_j in indices:
        try:
            syll_i_df = detrend(trajectories.loc[syll_i])
            syll_j_df = trajectories.loc[syll_j].copy()
            syll_j_df["onset_aligned_angle"] = -syll_j_df["onset_aligned_angle"]
            syll_j_df = detrend(syll_j_df)
            corr = correlation(syll_i_df.melt()["value"], syll_j_df.melt()["value"])

            syll_i_df = zscore(syll_i_df)
            syll_j_df = zscore(syll_j_df)
            corr_z = correlation(syll_i_df.melt()["value"], syll_j_df.melt()["value"])

            correlation_df.append(
                {
                    "syll_i": syll_i,
                    "syll_j": syll_j,
                    "correlation_z": 1 - corr_z,
                    "correlation": 1 - corr,
                    "confusion_rate": confusion_rate[syll_i, syll_j],
                }
            )
        except (ValueError, KeyError):
            logging.error(
                f"Error computing correlation for syllables {syll_i} and {syll_j}."
            )
    correlation_df = pd.DataFrame(correlation_df)
    return correlation_df


def generate_plots(
    full_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    per_syllable_similarity: np.ndarray,
    flip_qc_folder: Path,
    flip_similarity_threshold: float,
    kinematic_similarity_threshold: float,
):
    """
    Generate plots for the kinematic similarity results.

    Args:
        correlation_df: DataFrame containing the kinematic similarity results.
        flip_qc_folder: Path to the folder where the plots will be saved.
        flip_similarity_threshold: Threshold for flip similarity.
        kinematic_similarity_threshold: Threshold for kinematic similarity.
    """
    def _create_scatter_plot(hue_column: str, filename: str):
        """Create a scatter plot with the given hue column and save to filename."""
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=correlation_df,
            x="correlation",
            y="confusion_rate",
            hue=hue_column,
            palette="rocket",
            edgecolor="black",
            linewidth=0.35,
            ax=ax
        )
        # Add rectangle to highlight area of acceptance
        ax.add_patch(
            plt.Rectangle(
                (kinematic_similarity_threshold, flip_similarity_threshold),
                1 - kinematic_similarity_threshold,
                1 - flip_similarity_threshold,
                color="lightblue",
                alpha=0.5,
                fill=True,
                lw=0,
                zorder=-1,
            )
        )
        ax.set(
            title="Similarity after flip vs. Kinematic similarity",
            xlabel="Kinematic similarity",
            ylabel="Similarity after flip",
        )
        ax.legend(title=hue_column.replace("_", " ").capitalize(), frameon=False)

        # Annotate each point with pairs of syllable IDs
        for _, row in correlation_df.iterrows():
            ax.annotate(
                f"i: {row['syll_i']}, j: {row['syll_j']}",
                (row["correlation"], row["confusion_rate"] + 0.02),
                fontsize=8,
                horizontalalignment='center',
            )
        
        fig.savefig(flip_qc_folder / filename)
        plt.close(fig)

    # Create scatter plots for both usage types
    _create_scatter_plot("syll_i_usage", "confusion_vs_kinematic_similarity_syll_i_usage.png")
    _create_scatter_plot("syll_j_usage", "confusion_vs_kinematic_similarity_syll_j_usage.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(
        data=correlation_df,
        x="syll_i_usage",
        y="syll_j_usage",
        hue="confusion_rate",
        palette="rocket",
        edgecolor="black",
        linewidth=0.35,
        ax=ax
    )
    ax.set(
        xlabel="Syllable i usage",
        ylabel="Syllable j usage",
        title="Syllable usage",
        aspect="equal",
    )
    ax.legend(title="Similarity after flip", frameon=False)
    fig.savefig(flip_qc_folder / "syllable_usage_scatter.png")
    plt.close(fig)

    # plot heatmap of per-syllable similarity
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(
        per_syllable_similarity,
        cmap="viridis",
        cbar_kws={"label": "Jaccard similarity"},
    )
    ax.set(
        title="Per-syllable Jaccard similarity after flipping frames",
        xlabel="Original syllable ID (syllable i)",
        ylabel="Flipped syllable ID (syllable j)",
    )
    fig.savefig(flip_qc_folder / "per_syllable_similarity_heatmap.png")
    plt.close(fig)

    # plot syllable trajectories
    for _, row in correlation_df.iterrows():
        filtered_df = full_df[
            full_df["labels (original)"].isin([row["syll_i"], row["syll_j"]]) & (full_df["onset_aligned_time"] <= 0.4)
        ]
        fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
        for y, ax in zip(
            ["onset_aligned_angle", "velocity_3d_mm", "height_ave_mm", "length_mm"],
            axes.flatten(),
        ):
            ax = sns.lineplot(
                data=filtered_df,
                x="onset_aligned_time",
                y=y,
                hue="labels (original)",
                ax=ax,
                ci=95,
                n_boot=250,
                palette="colorblind",
            )
            ax.set(xlabel="Time (s)", ylabel=y.replace("_", " ").capitalize())
        sns.despine()
        merge_flag = (
            row["correlation"] > kinematic_similarity_threshold
            and row["confusion_rate"] > flip_similarity_threshold
        )
        fig.suptitle(f"Syllable trajectories for {row['syll_i']} and {row['syll_j']}\nWill merge: {merge_flag}")
        fig.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust top to make room for the title
        fig.savefig(flip_qc_folder / f"syllable_trajectories_{row['syll_i']}_{row['syll_j']}.png")
        plt.close(fig)


def configure_logging(flip_qc_folder: Path):
    """
    Configure logging to write to a file in the flip_qc_folder, and also print to stdout.

    Args:
        flip_qc_folder: Path to the folder where logs will be saved.
    """
    log_path = flip_qc_folder / "frame_flip_qc.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Also log to stdout
            logging.FileHandler(log_path, mode="w"),
        ]
    )


def _calculate_merge_directions(correlation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate merge directions based on syllable usage.
    The syllable with lower usage is merged into the one with higher usage.
    
    Args:
        correlation_df: DataFrame containing syllable correlation data
        
    Returns:
        DataFrame with additional merge direction columns
    """
    higher_usage = correlation_df["syll_i_usage"] > correlation_df["syll_j_usage"]
    
    correlation_df["merge_into"] = np.where(
        higher_usage,
        correlation_df["syll_i"],
        correlation_df["syll_j"],
    )
    correlation_df["to_merge"] = np.where(
        higher_usage,
        correlation_df["syll_j"],
        correlation_df["syll_i"],
    )
    
    return correlation_df


def _save_results(flip_qc_folder: Path, within_syllable_similarity: Dict, 
                 per_syllable_similarity: np.ndarray, filtered_correlation_df: pd.DataFrame):
    """Save analysis results to files."""
    # Save syllable similarity results
    with open(flip_qc_folder / "frame_flip_qc_results.txt", "w") as f:
        f.write("Syllable similarity results after flipping frames (Jaccard similarity):\n")
        for k, v in within_syllable_similarity.items():
            f.write(f"{k}: {v:.4f}\n")

    np.savetxt(flip_qc_folder / "per_syllable_similarity.txt", per_syllable_similarity)
    
    # Save kinematic similarity results
    filtered_correlation_df.to_csv(
        flip_qc_folder / "kinematic_similarity_results.csv", index=False
    )
    logging.info(f"Saved flip-based merging dataframe to {flip_qc_folder / 'kinematic_similarity_results.csv'}")

    # Save merge candidates
    with open(flip_qc_folder / "flip_merge_candidates.csv", "w") as f:
        f.write("labels (original) - old,labels (original) - new\n")
        for _, row in filtered_correlation_df.iterrows():
            f.write(f"{int(row['to_merge'])},{int(row['merge_into'])}\n")
    logging.info("Saved flip merge candidates to flip_merge_candidates.csv")


@click.command()
@click.argument("aggregate_folder", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--flip-qc-folder",
    default="flip_qc",
    type=click.Path(file_okay=False),
    help="Folder to save flip qc intermediate files",
)
@click.option(
    "--pca-folder",
    default="_pca",
    type=click.Path(file_okay=False),
    help="Folder containing PCA results",
)
@click.option(
    "--model-file",
    type=click.Path(dir_okay=False, exists=True),
    help="Path to trained MoSeq model file",
)
@click.option(
    "--dataframe-path",
    type=click.Path(dir_okay=False, exists=True),
    help="Path to dataframe containing the timeseries of extracted features",
)
@click.option(
    "--flip-similarity-threshold",
    default=0.35,
    type=float,
    help="Threshold to allow for merging syllables based on syllable similarity after a mirror flip. Merge-able syllables must have a similarity above this threshold.",
)
@click.option(
    "--kinematic-similarity-threshold",
    default=0.85,
    type=float,
    help="Threshold to allow for merging syllables based on kinematic similarity. Merge-able syllables must have a kinematic similarity above this threshold.",
)
def main(
    aggregate_folder,
    flip_qc_folder,
    pca_folder,
    model_file,
    dataframe_path,
    flip_similarity_threshold,
    kinematic_similarity_threshold,
):
    """
    Runs a quality control check to assess syllable similarity after flipping frames.

    Args:
        aggregate_folder: Path to folder containing aggregated extraction h5 files
        flip_qc_folder: Folder to save flip qc intermediate files
        pca_folder: Folder containing PCA results
        model_file: Path to trained MoSeq model file
    """
    flip_qc_folder = Path(flip_qc_folder)
    configure_logging(flip_qc_folder)

    files = sorted(Path(aggregate_folder).glob("*.h5"))

    # step 1: flip frames U-D
    logging.info("Flipping frames up-down")
    flip_frames(files, flip_qc_folder)

    flip_model_file = Path(model_file).parent / "flipped_frames_model_results.p"

    if not flip_model_file.exists():
        # step 2: apply PCA
        logging.info("Applying PCA")
        pca_command = [
            "moseq2-pca",
            "apply-pca",
            "-i",
            str(flip_qc_folder),
            "-o",
            pca_folder,
            "--pca-file",
            f"{pca_folder}/pca.h5",
            "--output-file",
            "flipped_pca_scores",
            "--cluster-type",
            "local",
            "--overwrite-pca-apply",
            "1",
            "--config-file",
            "config.yaml",
        ]
        _run_command(pca_command, "Error applying PCA")

        # step 3: apply moseq model
        logging.info("Applying MoSeq model")
        model_command = [
            "moseq2-model",
            "apply-model",
            model_file,
            str(Path(pca_folder, "flipped_pca_scores.h5")),
            str(flip_model_file),
        ]
        _run_command(model_command, "Error applying MoSeq model")

    # step 4: compare syllable similarity
    within_syllable_similarity, avg_similarity, per_syllable_similarity = (
        compare_syllable_similarity(
            model_file,
            str(flip_model_file),
        )
    )

    # step 5: save results
    with open(flip_qc_folder / "frame_flip_qc_results.txt", "w") as f:
        f.write(
            "Syllable similarity results after flipping frames (Jaccard similarity):\n"
        )
        for k, v in within_syllable_similarity.items():
            f.write(f"{k}: {v:.4f}\n")

    np.savetxt(
        flip_qc_folder / "per_syllable_similarity.txt",
        per_syllable_similarity,
    )

    logging.info("Performing kinematic similarity analysis step")

    # load dataframe
    logging.info(f"Loading dataframe from {dataframe_path}")
    df = pd.read_csv(dataframe_path, index_col=0)

    # compute syllable usage to merge later
    usage = df.query("onset")["labels (original)"].value_counts(normalize=True)

    # process dataframe - add columns for syllable duration and syllable onset alignment
    logging.info("Computing syllable-aligned features")
    df = compute_syllable_aligned_features(df)

    # filter for syllables within a specific duration range
    df = df.query(f"`syllable_duration (s)` > {MIN_SYLLABLE_DURATION} & `syllable_duration (s)` < {MAX_SYLLABLE_DURATION}")

    # compute kinematic trajectories for each syllable
    logging.info("Computing kinematic trajectories")
    trajectories = compute_kinematic_trajectories(df)

    # compute kinematic similarity between confused syllables
    upper = np.triu(per_syllable_similarity, k=1)
    most_similar_indices = np.argsort(upper.flatten())[::-1]
    most_similar_indices = most_similar_indices[
        upper.flatten()[most_similar_indices] > SIMILARITY_THRESHOLD
    ]
    most_similar_indices = np.array(
        list(zip(*np.unravel_index(most_similar_indices, upper.shape)))
    )

    logging.info("Computing kinematic similarity")
    correlation_df = compute_kinematic_similarity(
        trajectories, most_similar_indices, upper
    )
    correlation_df["syll_i_usage"] = correlation_df["syll_i"].map(usage)
    correlation_df["syll_j_usage"] = correlation_df["syll_j"].map(usage)

    # calculate merge directions for syllables
    correlation_df = _calculate_merge_directions(correlation_df)

    logging.info("Generating plots")
    generate_plots(
        df,
        correlation_df,
        per_syllable_similarity,
        flip_qc_folder,
        flip_similarity_threshold,
        kinematic_similarity_threshold,
    )

    filtered_correlation_df = correlation_df[
        (correlation_df["correlation"] > kinematic_similarity_threshold)
        & (correlation_df["confusion_rate"] > flip_similarity_threshold)
    ]

    # save the dataframe, and use results to create a merge map
    _save_results(flip_qc_folder, within_syllable_similarity, per_syllable_similarity, filtered_correlation_df)


if __name__ == "__main__":
    main()
