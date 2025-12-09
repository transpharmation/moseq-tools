#!/usr/bin/env python
"""
Computes features for each syllable in a given experiment.

Features include:
- average mouse height
- average mouse width
- average mouse length
- average mouse speed
- PC scores for a XX second window from the start of each syllable
- syllable duration
- dynamics characteristics from the autoregressive matrices
"""

import h5py
import click
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

"""
Columns that exist in the `moseq_df.csv` dataframe
'labels (original)', 'labels (usage sort)',
'labels (frames sort)', 'onset', 'frame index', 'syllable index',
'kappa', 'model_type', 'angle', 'area_mm', 'area_px', 'centroid_x_mm',
'centroid_x_px', 'centroid_y_mm', 'centroid_y_px', 'height_ave_mm',
'length_mm', 'length_px', 'velocity_2d_mm', 'velocity_2d_px',
'velocity_3d_mm', 'velocity_3d_px', 'velocity_theta', 'width_mm',
'width_px', 'dist_to_center_px', 'group', 'uuid', 'h5_path',
'timestamps', 'SessionName', 'StartTime', 'SubjectName'
"""


def compute_syllable_duration(df):
    """Compute syllable duration for each syllable."""
    arr = df["onset"].to_numpy()
    onsets = np.where(arr)[0]
    durations = np.diff(
        onsets, prepend=0
    )  # Duration is difference between consecutive onsets
    df.loc[df.index[onsets], "syllable_duration"] = durations
    df["syllable_duration"] = df["syllable_duration"].ffill()
    return df


def load_pcs(path: str, uuids):
    with h5py.File(path, "r") as f:
        pcs = []
        avail_uuids = []
        for u in uuids:
            if u in f["scores"]:
                avail_uuids.append(u)
                pcs.append(f["scores"][u][3:, :10])
    pcs = np.vstack(pcs)
    zpcs = (pcs - np.nanmean(pcs, axis=0, keepdims=True)) / np.nanstd(
        pcs, axis=0, keepdims=True
    )
    return zpcs, avail_uuids


def compute_syllable_features(df):
    """Compute features for each syllable."""
    # Filter for syllable onset frames only
    onset_df = df.query("onset").copy()

    features = []

    for syllable_id in onset_df["labels (original)"].unique():
        syll_data = onset_df[onset_df["labels (original)"] == syllable_id]

        # Basic morphological features
        avg_height = syll_data["height_ave_mm"].mean()
        std_height = syll_data["height_ave_mm"].std()
        avg_width = syll_data["width_mm"].mean()
        std_width = syll_data["width_mm"].std()
        avg_length = syll_data["length_mm"].mean()
        std_length = syll_data["length_mm"].std()

        # Speed feature
        avg_speed = syll_data["velocity_3d_mm"].mean()
        std_speed = syll_data["velocity_3d_mm"].std()

        # Syllable duration (convert from frames to seconds assuming 30 fps)
        avg_duration = syll_data["syllable_duration"].mean() / 30
        std_duration = (syll_data["syllable_duration"] / 30).std()

        # Normalized syllable frequency (usage)
        frequency = len(syll_data) / len(onset_df)

        # PC scores - take average of first 10 PCs during syllable onset
        pc_features = {}
        for i in range(10):  # First 10 PC components
            pc_col = f"pc_{i+1}"
            if pc_col in syll_data.columns:
                pc_features[f"avg_pc_{i+1}"] = syll_data[pc_col].mean()
                pc_features[f"std_pc_{i+1}"] = syll_data[pc_col].std()
            else:
                pc_features[f"avg_pc_{i+1}"] = np.nan
                pc_features[f"std_pc_{i+1}"] = np.nan

        feature_dict = {
            "syllable_id": syllable_id,
            "avg_height_mm": avg_height,
            "std_height_mm": std_height,
            "avg_width_mm": avg_width,
            "std_width_mm": std_width,
            "avg_length_mm": avg_length,
            "std_length_mm": std_length,
            "avg_speed_3d_mm_s": avg_speed,
            "std_speed_3d_mm_s": std_speed,
            "avg_duration_s": avg_duration,
            "std_duration_s": std_duration,
            "normalized_frequency": frequency,
            **pc_features,
        }

        features.append(feature_dict)

    return pd.DataFrame(features)


def load_ar_matrices(path):

    model_dict = joblib.load(path)

    ar_mat = model_dict["model_parameters"]["ar_mat"]
    ar_mat = [x[:, :-1] for x in ar_mat]

    return ar_mat


def get_ar_eigvals(ar_mat):

    eigvals = []

    for ar in ar_mat:
        n = ar.shape[0]
        lags = ar.shape[1] // n
        C = np.zeros((lags * n, lags * n))
        C[0:n, 0:] = ar
        C[n : 2 * n, 0:n] = np.eye(n)
        C[2 * n : 3 * n, n : 2 * n] = np.eye(n)
        ev, evec = np.linalg.eigh(C)
        eigvals.append(ev)

    return eigvals


def get_spectral_radius(eigvals):
    return np.max(np.abs(eigvals))


def dynamic_complexity(eigvals):
    """Measures the spread of dynamic modes across the eigenvalues."""

    # normalize the eignvalue magnitudes
    mags = np.abs(eigvals)
    mags = mags / np.sum(mags)

    return -np.sum(mags * np.log(mags + 1e-10))


@click.command()
@click.argument("moseq_df_path", type=click.Path(exists=True))
@click.argument("pcs_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--output-path",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for features CSV file. By default, saves to the same directory as the input for moseq_df_path.",
)
def main(moseq_df_path, pcs_path, model_path, output_path):
    """Generate meta-clustering features from MoSeq dataframe.

    Args:
        moseq_df_path: Path to the MoSeq dataframe CSV file (i.e., moseq_df.csv)
        pcs_path: Path to the PCs HDF5 file (i.e., _pca/pca_scores.h5)
        model_path: Path to the MoSeq model file (i.e., _model/best-model.p)
        output_path: Optional output path for the features CSV file
    """

    # TODO: extract the AR trajectories from this dict
    ar_mat = load_ar_matrices(model_path)
    eigvals = get_ar_eigvals(ar_mat)
    spectral_radii = {i: get_spectral_radius(e) for i, e in enumerate(eigvals)}
    complexities = {i: dynamic_complexity(e) for i, e in enumerate(eigvals)}
    eigval_dict = dict(enumerate(eigvals))

    df = pd.read_csv(moseq_df_path, index_col=0)
    pcs, avail_uuids = load_pcs(pcs_path, df["uuid"].unique())
    df = df[df["uuid"].isin(avail_uuids)]
    # filter out all syllables with label -5
    df = df[df["labels (original)"] != -5]

    if len(df) != len(pcs):
        raise ValueError(
            f"Number of rows in dataframe ({len(df)}) does not match number of PC rows ({len(pcs)}) after filtering for available UUIDs."
        )
    for i in range(pcs.shape[1]):
        df[f"pc_{i+1}"] = pcs[:, i]

    print(f"Loaded dataframe with {len(df)} rows")

    print("Computing syllable durations...")
    df = compute_syllable_duration(df)

    print(
        f"Computing features for {df['labels (original)'].nunique()} unique syllables"
    )

    # Compute syllable features
    features_df = compute_syllable_features(df)
    features_df["spectral_radii"] = features_df["syllable_id"].map(spectral_radii)
    features_df["dynamic_complexity"] = features_df["syllable_id"].map(complexities)
    for i in range(10):
        features_df[f"eigval_{i+1}"] = features_df["syllable_id"].map(
            lambda x: np.abs(eigval_dict[x][i]) if len(eigval_dict[x]) > i else np.nan
        )

    # Determine output path
    if output_path is None:
        input_path = Path(moseq_df_path)
        output_path = input_path.parent / "syllable_features.csv"

    # Save features
    features_df.to_csv(output_path, index=False)
    print(f"Saved syllable features to {output_path}")
    print(f"Generated features for {len(features_df)} syllables")


if __name__ == "__main__":
    main()
