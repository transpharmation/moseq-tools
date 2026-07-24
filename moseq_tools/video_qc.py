#!/usr/bin/env python
"""
Fast quality-control reports for raw MoSeq depth videos.
"""

import json
import logging
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage
from scipy.stats import rankdata
from tqdm.auto import tqdm

DEFAULT_VIDEO_NAME = "depth.avi"
EDGE_DISTANCE_THRESHOLD = 25
WALL_DISTANCE_MIN_THRESHOLD = 2.0
FPS_WARN_PCT = 5.0
PRESENCE_WARN_PCT = 95.0
DEFAULT_BG_ROI_DEPTH_RANGE = (650.0, 750.0)
BG_ROI_WEIGHTS = (1.0, 0.1, 1.0)
DEFAULT_ARENA_REFERENCE_AREA = 124275.0
DEFAULT_ARENA_AREA_TOLERANCE = 0.15
DEFAULT_ARENA_CIRCULARITY_THRESHOLD = 0.50
DEFAULT_ARENA_HOLE_DIAMETER_PX = 5.0
SAMPLE_INTERVAL_SECONDS = 2.0
FRAME_HEIGHT_DISPLAY_RANGE_MM = (0, 80)
FRAME_HEIGHT_COLORBAR_LABEL = "Height above arena floor (mm)"
STATUS_LABEL_COLORS = {
    "PASS": "#15803d",
    "WARN": "#b91c1c",
}
PDF_CONTENT_TOP = 0.88
PDF_CONTENT_BOTTOM = 0.08
PDF_CONTENT_LEFT = 0.04
PDF_CONTENT_RIGHT = 0.98
PDF_FRAME_CONTENT_RIGHT = 0.94
PDF_WALL_CONTENT_LEFT = 0.075
PDF_GRID_HSPACE = 0.55
PDF_GRID_WSPACE = 0.35
PDF_SUMMARY_WIDTH_RATIOS = (1.25, 1.0, 1.55)
PDF_SUMMARY_WSPACE = 0.25


@dataclass(frozen=True)
class FrameIOConfig:
    """FFmpeg/ffprobe parameters threaded through every frame-reading call."""

    pixel_format: str = "gray16le"
    threads: int = 6
    command_timeout: int = 60
    prefetch_sampled_frames: bool = False
    mask_workers: int = 1


def configure_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path), mode="w"),
        ],
    )


def _run_with_timeout(command, timeout=None):
    """Run command capturing stdout/stderr.

    Returns (out, err, returncode). If the command exceeds timeout, the process
    is killed and reaped before subprocess.TimeoutExpired is re-raised, so each
    caller can apply its own timeout policy.
    """
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    return out, err, proc.returncode


def run_command(command, error_msg, timeout=None):
    logging.debug("Running command: %s", " ".join(command))
    try:
        out, err, returncode = _run_with_timeout(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "{}: command timed out after {} seconds".format(error_msg, timeout)
        )
    if returncode != 0:
        err_text = err.decode("utf-8", errors="replace")
        raise RuntimeError("{}: {}".format(error_msg, err_text.strip()))
    return out


def parse_rate(rate):
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            den = float(den)
            if den == 0:
                return None
            return float(num) / den
        except ValueError:
            return None
    try:
        return float(rate)
    except ValueError:
        return None


def get_stream_names(filename, command_timeout=60):
    command = [
        "ffprobe",
        "-v",
        "fatal",
        "-show_entries",
        "stream_tags=title",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(filename),
    ]
    try:
        out, err, returncode = _run_with_timeout(command, timeout=command_timeout)
    except subprocess.TimeoutExpired:
        return {"DEPTH": 0}
    if returncode != 0 or err or len(out) == 0:
        return {"DEPTH": 0}
    names = out.decode("utf-8", errors="replace").rstrip("\n").split("\n")
    return {name: i for i, name in enumerate(names)}


def get_video_info(filename, config, mapping="DEPTH"):
    mapping_dict = get_stream_names(filename, command_timeout=config.command_timeout)
    if isinstance(mapping, str):
        mapping = mapping_dict.get(mapping, 0)

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:{}".format(mapping),
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        "-threads",
        str(config.threads),
        str(filename),
    ]
    out = run_command(
        command,
        "Error reading video metadata",
        timeout=config.command_timeout,
    )
    payload = json.loads(out.decode("utf-8"))
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("No video streams found in {}".format(filename))

    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(
        stream.get("r_frame_rate")
    )
    duration = stream.get("duration")
    duration = float(duration) if duration not in (None, "N/A") else None

    nframes = stream.get("nb_frames")
    try:
        nframes = int(nframes)
    except (TypeError, ValueError):
        nframes = None

    if nframes is None and fps and duration:
        nframes = int(round(duration * fps))

    return {
        "file": str(filename),
        "codec": stream.get("codec_name", "unknown"),
        "pixel_format": stream.get("pix_fmt", "unknown"),
        "dims": (width, height),
        "fps": fps,
        "duration": duration,
        "nframes": nframes,
        "mapping": mapping,
    }


def pixel_format_to_dtype(pixel_format):
    if pixel_format.endswith("be"):
        return np.dtype(">u2")
    return np.dtype("<u2")


def frame_index_to_seconds(frame_index, fps):
    if fps is None or fps <= 0:
        return 0.0
    return float(frame_index) / float(fps)


def read_frame_window(filename, start_frame, count, finfo, config):
    width, height = finfo["dims"]
    fps = finfo.get("fps") or 30.0
    start_time = frame_index_to_seconds(start_frame, fps)

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        "{:.6f}".format(start_time),
        "-i",
        str(filename),
        "-vframes",
        str(count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        config.pixel_format,
        "-threads",
        str(config.threads),
        "-vcodec",
        "rawvideo",
    ]
    if str(filename).lower().endswith((".avi", ".mkv")):
        command += ["-map", "0:{}".format(finfo.get("mapping", 0)), "-vsync", "0"]
    command += ["-"]

    out = run_command(
        command,
        "Error reading frames from {}".format(filename),
        timeout=config.command_timeout,
    )
    dtype = pixel_format_to_dtype(config.pixel_format)
    values_per_frame = width * height
    frame_count = int(len(out) / (values_per_frame * dtype.itemsize))
    if frame_count == 0:
        raise RuntimeError("FFmpeg returned no frames for {}".format(filename))
    if frame_count < count:
        logging.warning(
            "Requested %s frames from %s but only decoded %s",
            count,
            filename,
            frame_count,
        )
    data = np.frombuffer(out, dtype=dtype, count=frame_count * values_per_frame)
    data = data.reshape((frame_count, height, width))
    # np.frombuffer returns a read-only view; copy so callers can mutate freely.
    return data.astype("uint16")


def build_sample_indices(nframes, fps, sample_interval_seconds=SAMPLE_INTERVAL_SECONDS):
    if nframes is None or nframes <= 0:
        raise RuntimeError("Cannot sample video without a valid frame count")
    if fps is None or fps <= 0:
        raise RuntimeError("Cannot sample video without a valid frame rate")
    stride = max(1, int(round(float(fps) * float(sample_interval_seconds))))
    return np.arange(0, int(nframes), stride, dtype="int64")


def group_contiguous_indices(indices):
    if len(indices) == 0:
        return []
    groups = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            groups.append((start, prev))
            start = value
            prev = value
    groups.append((start, prev))
    return groups


def read_sampled_frames(
    filename,
    finfo,
    config,
    fallback_fps=None,
    sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
):
    fps = finfo.get("fps") or fallback_fps
    indices = build_sample_indices(
        finfo["nframes"],
        fps,
        sample_interval_seconds=sample_interval_seconds,
    )
    return read_indexed_frames(
        filename,
        finfo,
        indices,
        config,
        "Reading sampled frame windows",
    )


def read_indexed_frames(filename, finfo, indices, config, desc):
    frames = []
    actual_indices = []
    groups = group_contiguous_indices(np.asarray(indices, dtype="int64"))
    for start, end in tqdm(groups, desc=desc, leave=False):
        count = end - start + 1
        chunk = read_frame_window(filename, start, count, finfo, config)
        frames.append(chunk)
        actual_indices.extend(range(start, start + len(chunk)))
    if not frames:
        raise RuntimeError("No frames were read from {}".format(filename))
    return np.concatenate(frames, axis=0), np.array(actual_indices, dtype="int64")


def compute_background(
    filename, finfo, bg_frame_stride, background_median_blur, config
):
    if finfo["nframes"] is None or finfo["nframes"] <= 0:
        raise RuntimeError("Cannot compute background without a valid frame count")

    stride = max(1, int(bg_frame_stride))
    indices = np.arange(0, int(finfo["nframes"]), stride, dtype="int64")
    if indices[-1] != int(finfo["nframes"]) - 1:
        indices = np.append(indices, int(finfo["nframes"]) - 1)

    frames, _ = read_indexed_frames(
        filename,
        finfo,
        indices,
        config,
        desc="Reading background frames",
    )
    blur_size = int(background_median_blur)
    if blur_size > 1:
        if blur_size % 2 == 0:
            blur_size += 1
        frames = np.asarray(
            [cv2.medianBlur(frame, blur_size) for frame in frames],
            dtype="uint16",
        )
    return np.nanmedian(frames.astype("float32"), axis=0).astype("uint16")


def plane_fit3(points):
    a = points[1] - points[0]
    b = points[2] - points[0]
    normal = np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype="float64",
    )
    denom = np.sum(np.square(normal))
    if denom < np.spacing(1):
        return None
    normal /= np.sqrt(denom)
    return np.hstack((normal, np.dot(-points[0], normal)))


def plane_ransac(
    depth_image,
    bg_roi_depth_range,
    noise_tolerance=30,
    iters=1000,
    in_ratio=0.1,
):
    use_points = np.logical_and(
        depth_image > bg_roi_depth_range[0],
        depth_image < bg_roi_depth_range[1],
    )
    point_count = int(np.sum(use_points))
    if point_count <= 10:
        raise RuntimeError(
            "Too few pixels exist within bg ROI depth range {}: {}".format(
                tuple(bg_roi_depth_range), point_count
            )
        )

    xx, yy = np.meshgrid(
        np.arange(depth_image.shape[1]),
        np.arange(depth_image.shape[0]),
    )
    coords = np.vstack(
        (
            xx[use_points].ravel(),
            yy[use_points].ravel(),
            depth_image[use_points].ravel(),
        )
    ).T.astype("float64")

    rng = np.random.default_rng(0)

    best_plane = None
    best_dist = np.inf
    best_num = 0

    for _ in tqdm(range(int(iters)), desc="Finding arena plane", leave=False):
        sel = coords[rng.choice(coords.shape[0], 3, replace=True)]
        tmp_plane = plane_fit3(sel)
        if tmp_plane is None:
            continue

        dist = np.abs(np.dot(coords, tmp_plane[:3]) + tmp_plane[3])
        inliers = dist < noise_tolerance
        ninliers = int(np.sum(inliers))
        mean_dist = float(np.mean(dist))
        if (
            (ninliers / float(point_count)) > in_ratio
            and ninliers > best_num
            and mean_dist < best_dist
        ):
            best_dist = mean_dist
            best_num = ninliers
            best_plane = tmp_plane

    if best_plane is None:
        raise RuntimeError("Could not fit an arena plane from the background image")

    all_coords = np.vstack((xx.ravel(), yy.ravel(), depth_image.ravel())).T.astype(
        "float64"
    )
    return best_plane, np.abs(np.dot(all_coords, best_plane[:3]) + best_plane[3])


def get_bucket_center(image, threshold):
    _, thresh = cv2.threshold(
        image.astype("float32"),
        float(threshold),
        float(np.nanmax(image) or 1.0),
        cv2.THRESH_BINARY,
    )
    moments = cv2.moments(thresh)
    if moments["m00"] == 0:
        return image.shape[1] // 2, image.shape[0] // 2
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def auto_bg_roi_depth_range(background):
    finite = background[np.isfinite(background)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return DEFAULT_BG_ROI_DEPTH_RANGE

    threshold = float(np.nanmedian(finite) / 2.0)
    cx, cy = get_bucket_center(background, threshold)
    center_depth = float(background[cy, cx])
    if center_depth <= 0:
        center_depth = float(np.nanmedian(finite))
    return (center_depth - 50.0, center_depth + 50.0)


def compute_arena_roi_masks(
    background,
    bg_roi_depth_range,
    bg_roi_dilate=(10, 10),
    bg_roi_erode=(1, 1),
    dilate_iterations=1,
    erode_iterations=0,
    noise_tolerance=30,
):
    _, distances = plane_ransac(
        background,
        bg_roi_depth_range=bg_roi_depth_range,
        noise_tolerance=noise_tolerance,
    )
    inliers = distances.reshape(background.shape) < noise_tolerance
    labels = skimage.measure.label(inliers)
    regions = skimage.measure.regionprops(labels)
    if not regions:
        raise RuntimeError("No arena ROI candidates were found")

    areas = np.asarray([region.area for region in regions], dtype="float64")
    extents = np.asarray([region.extent for region in regions], dtype="float64")
    center = np.asarray(background.shape, dtype="float64") / 2.0
    dists = np.asarray(
        [
            np.sqrt(np.sum(np.square(region.coords - center), axis=1)).max()
            for region in regions
        ],
        dtype="float64",
    )
    ranks = np.vstack(
        (
            rankdata(-areas, method="max"),
            rankdata(-extents, method="max"),
            rankdata(dists, method="max"),
        )
    )
    weights = np.asarray(BG_ROI_WEIGHTS, dtype="float32")
    best_region_idx = int(
        np.mean(ranks.astype("float32") * weights[:, np.newaxis], axis=0).argmin()
    )

    roi = np.zeros(background.shape, dtype="uint8")
    coords = regions[best_region_idx].coords
    roi[coords[:, 0], coords[:, 1]] = 1

    strel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(bg_roi_dilate))
    strel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(bg_roi_erode))
    roi = cv2.dilate(roi, strel_dilate, iterations=int(dilate_iterations))
    if erode_iterations > 0:
        roi = cv2.erode(roi, strel_erode, iterations=int(erode_iterations))
    unfilled_roi = roi > 0
    filled_roi = ndimage.binary_fill_holes(unfilled_roi)
    return filled_roi, unfilled_roi


def compute_arena_roi(background, bg_roi_depth_range, **kwargs):
    filled_roi, _ = compute_arena_roi_masks(background, bg_roi_depth_range, **kwargs)
    return filled_roi


def equivalent_diameter(area):
    if area <= 0:
        return 0.0
    return float(np.sqrt(4.0 * float(area) / np.pi))


def compute_arena_mask_quality(
    arena_roi,
    unfilled_arena_roi,
    reference_area=DEFAULT_ARENA_REFERENCE_AREA,
    area_tolerance=DEFAULT_ARENA_AREA_TOLERANCE,
    circularity_threshold=DEFAULT_ARENA_CIRCULARITY_THRESHOLD,
    hole_diameter_px=DEFAULT_ARENA_HOLE_DIAMETER_PX,
):
    area = int(np.sum(arena_roi))
    perimeter = float(skimage.measure.perimeter(arena_roi, neighborhood=8))
    if area > 0 and perimeter > 0:
        circularity = float(4.0 * np.pi * float(area) / (perimeter * perimeter))
    else:
        circularity = 0.0

    holes = np.logical_and(arena_roi, ~unfilled_arena_roi)
    labels, label_count = ndimage.label(holes, structure=np.ones((3, 3), dtype=bool))
    if label_count:
        hole_areas = np.bincount(labels.ravel())[1:]
        large_hole_count = int(
            np.sum(
                [
                    equivalent_diameter(hole_area) >= float(hole_diameter_px)
                    for hole_area in hole_areas
                ]
            )
        )
    else:
        large_hole_count = 0

    reference_area = float(reference_area)
    area_tolerance = float(area_tolerance)
    min_area = reference_area * (1.0 - area_tolerance)
    max_area = reference_area * (1.0 + area_tolerance)
    area_ok = min_area <= float(area) <= max_area
    circularity_ok = circularity >= float(circularity_threshold)
    holes_ok = large_hole_count == 0

    return {
        "arena_area_px": area,
        "arena_reference_area_px": reference_area,
        "arena_area_min_px": float(min_area),
        "arena_area_max_px": float(max_area),
        "arena_circularity": circularity,
        "arena_large_hole_count": large_hole_count,
        "arena_area_ok": bool(area_ok),
        "arena_circularity_ok": bool(circularity_ok),
        "arena_holes_ok": bool(holes_ok),
        "arena_quality_ok": bool(area_ok and circularity_ok and holes_ok),
    }


def load_depth_timestamps(video_path, nframes):
    ts_path = video_path.with_name("depth_ts.txt")
    if not ts_path.exists():
        raise FileNotFoundError("depth_ts.txt not found")
    try:
        timestamps = np.loadtxt(str(ts_path), dtype="float64")
    except Exception as exc:
        return None, "Could not read depth_ts.txt: {}".format(exc)
    timestamps = np.atleast_1d(timestamps) / 1000.0
    if nframes is not None and len(timestamps) != int(nframes):
        raise ValueError(
            "depth_ts.txt line count ({}) did not match frame count ({})".format(
                len(timestamps), nframes
            )
        )
    if len(timestamps) < 2:
        raise ValueError("depth_ts.txt has fewer than 2 timestamps")
    return timestamps, "depth_ts.txt"


def summarize_timing(finfo, expected_fps, video_path):
    nframes = finfo.get("nframes")
    metadata_fps = finfo.get("fps")
    timestamps, source = load_depth_timestamps(video_path, nframes)
    intervals = None
    estimated_missing = None
    jittery_pct = None

    if timestamps is not None:
        duration = float(timestamps[-1] - timestamps[0])
        observed_fps = float(len(timestamps) - 1) / duration if duration > 0 else None
        intervals = np.diff(timestamps)
        expected_interval = 1.0 / float(expected_fps)
        estimated_missing = int(
            np.sum(np.maximum(np.rint(intervals * expected_fps).astype("int64") - 1, 0))
        )
        jittery = np.abs(intervals - expected_interval) > (expected_interval * 0.5)
        jittery_pct = float(np.mean(jittery) * 100.0)
    else:
        observed_fps = metadata_fps
        duration = finfo.get("duration")
        if duration is None and nframes and observed_fps:
            duration = float(nframes) / float(observed_fps)

    if observed_fps:
        fps_deviation_pct = (
            abs(observed_fps - expected_fps) / float(expected_fps) * 100.0
        )
    else:
        fps_deviation_pct = None

    return {
        "timestamp_source": source,
        "observed_fps": observed_fps,
        "expected_fps": float(expected_fps),
        "metadata_fps": metadata_fps,
        "timestamps": timestamps,
        "fps_deviation_pct": fps_deviation_pct,
        "duration": duration,
        "nframes": nframes,
        "intervals": intervals,
        "estimated_missing_frames": estimated_missing,
        "jittery_interval_pct": jittery_pct,
    }


def largest_component(mask, structure):
    cleaned = ndimage.binary_opening(mask, structure=structure)
    cleaned = ndimage.binary_closing(cleaned, structure=structure)
    labels, label_count = ndimage.label(cleaned, structure=structure)
    if label_count == 0:
        return {
            "mask": cleaned,
            "area": 0,
            "component_count": 0,
            "bbox": None,
            "edge_distance": np.nan,
        }

    areas = np.bincount(labels.ravel())
    areas[0] = 0
    largest_label = int(np.argmax(areas))
    largest_area = int(areas[largest_label])
    component_count = int(np.sum(areas > 25))
    component_mask = labels == largest_label
    ys, xs = np.where(component_mask)
    if len(xs) == 0:
        bbox = None
        edge_distance = np.nan
    else:
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        height, width = component_mask.shape
        bbox = (x0, y0, x1, y1)
        edge_distance = float(min(x0, y0, width - 1 - x1, height - 1 - y1))

    return {
        "mask": component_mask,
        "area": largest_area,
        "component_count": component_count,
        "bbox": bbox,
        "edge_distance": edge_distance,
    }


def median_or_nan(values):
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.nanmedian(values))


def compute_wall_reflection_metrics(raw_masks, component_masks, present, arena_roi):
    foreground_pixel_counts = np.sum(raw_masks, axis=(1, 2)).astype("float64")
    wall_distance_map = ndimage.distance_transform_edt(arena_roi)
    wall_distances = np.full(len(component_masks), np.nan, dtype="float64")

    for i, component_mask in enumerate(component_masks):
        if not present[i]:
            continue
        component_distances = wall_distance_map[component_mask]
        if component_distances.size:
            wall_distances[i] = float(np.nanmin(component_distances))

    valid_wall_distance = np.logical_and(
        present,
        np.logical_and(
            np.isfinite(wall_distances),
            wall_distances >= float(WALL_DISTANCE_MIN_THRESHOLD),
        ),
    )
    wall_adjacent = np.logical_and(
        valid_wall_distance,
        wall_distances < float(EDGE_DISTANCE_THRESHOLD),
    )
    wall_away = np.logical_and(
        valid_wall_distance,
        ~wall_adjacent,
    )
    excluded_too_close = np.logical_and(
        present,
        np.logical_and(
            np.isfinite(wall_distances),
            wall_distances < float(WALL_DISTANCE_MIN_THRESHOLD),
        ),
    )

    near_median = median_or_nan(foreground_pixel_counts[wall_adjacent])
    away_median = median_or_nan(foreground_pixel_counts[wall_away])
    if np.isfinite(near_median) and np.isfinite(away_median) and away_median > 0:
        ratio = float(near_median / away_median)
    else:
        ratio = np.nan

    return {
        "wall_distances": wall_distances,
        "wall_adjacent": wall_adjacent,
        "foreground_pixel_counts": foreground_pixel_counts,
        "wall_near_foreground_median": near_median,
        "wall_away_foreground_median": away_median,
        "wall_near_away_foreground_ratio": ratio,
        "wall_near_frame_count": int(np.sum(wall_adjacent)),
        "wall_away_frame_count": int(np.sum(wall_away)),
        "wall_excluded_too_close_count": int(np.sum(excluded_too_close)),
        "wall_distance_min_threshold": float(WALL_DISTANCE_MIN_THRESHOLD),
    }


def compute_wall_adjacent_height_image(
    heights, wall_adjacent, arena_roi, percentile=99
):
    if not np.any(wall_adjacent):
        image = np.full(arena_roi.shape, np.nan, dtype="float32")
    else:
        image = np.percentile(
            heights[wall_adjacent].astype("float32"),
            float(percentile),
            axis=0,
        ).astype("float32")
        image[~arena_roi] = np.nan
    return image


def analyze_masks(
    frames,
    frame_indices,
    background,
    arena_roi,
    min_height_mm,
    max_height_mm,
    min_mouse_area,
    bg_roi_depth_range,
    mask_workers=1,
):
    heights = background.astype("int32")[np.newaxis, :, :] - frames.astype("int32")
    heights[frames == 0] = 0
    heights[heights < 0] = 0
    raw_masks = np.logical_and(
        heights >= float(min_height_mm),
        heights <= float(max_height_mm),
    )
    raw_masks &= arena_roi[np.newaxis, :, :]
    structure = np.ones((3, 3), dtype=bool)

    component_masks = []
    areas = []
    component_counts = []
    edge_distances = []
    bboxes = []

    worker_count = max(1, int(mask_workers))
    if worker_count == 1:
        components = [
            largest_component(mask, structure)
            for mask in tqdm(raw_masks, desc="Analyzing masks", leave=False)
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            components = list(
                tqdm(
                    executor.map(
                        lambda mask: largest_component(mask, structure), raw_masks
                    ),
                    total=len(raw_masks),
                    desc="Analyzing masks",
                    leave=False,
                )
            )

    for component in components:
        component_masks.append(component["mask"])
        areas.append(component["area"])
        component_counts.append(component["component_count"])
        edge_distances.append(component["edge_distance"])
        bboxes.append(component["bbox"])

    component_masks = np.asarray(component_masks, dtype=bool)
    areas = np.asarray(areas, dtype="float64")
    component_counts = np.asarray(component_counts, dtype="float64")
    edge_distances = np.asarray(edge_distances, dtype="float64")
    present = areas >= float(min_mouse_area)
    wall_metrics = compute_wall_reflection_metrics(
        raw_masks,
        component_masks,
        present,
        arena_roi,
    )
    wall_height_image = compute_wall_adjacent_height_image(
        heights,
        wall_metrics["wall_adjacent"],
        arena_roi,
    )

    metrics = {
        "background": background,
        "heights": heights,
        "raw_masks": raw_masks,
        "component_masks": component_masks,
        "arena_roi": arena_roi,
        "areas": areas,
        "component_counts": component_counts,
        "edge_distances": edge_distances,
        "bboxes": bboxes,
        "present": present,
        "frame_indices": frame_indices,
        "presence_pct": float(np.mean(present) * 100.0),
        "median_area": float(np.nanmedian(areas)),
        "median_components": float(np.nanmedian(component_counts)),
        "edge_adjacent_pct": float(
            np.mean((edge_distances < EDGE_DISTANCE_THRESHOLD) & present) * 100.0
        ),
        "mouse_mask_coverage_pct": float(
            np.nanmedian(areas) / float(frames.shape[1] * frames.shape[2]) * 100.0
        ),
        "roi_coverage_pct": float(np.mean(arena_roi) * 100.0),
        "bg_roi_depth_range": tuple(float(v) for v in bg_roi_depth_range),
        "wall_adjacent_height_p99": wall_height_image,
        "wall_adjacent_height_percentile": 99.0,
    }
    metrics.update(wall_metrics)
    return metrics


def safe_nanarg(values, mode):
    values = np.asarray(values, dtype="float64")
    if np.all(np.isnan(values)):
        return 0
    if mode == "min":
        return int(np.nanargmin(values))
    if mode == "max":
        return int(np.nanargmax(values))
    raise ValueError("Unknown mode {}".format(mode))


def choose_representative_indices(mask_metrics):
    areas = mask_metrics["areas"]
    present = mask_metrics["present"]
    edge = mask_metrics["edge_distances"].copy()
    choices = {}

    if np.any(present):
        present_idx = np.where(present)[0]
        median_area = np.nanmedian(areas[present])
        typical_local = np.argmin(np.abs(areas[present] - median_area))
        choices["typical"] = int(present_idx[typical_local])
        choices["worst detection"] = int(present_idx[np.argmin(areas[present])])
        edge[~present] = np.nan
        choices["wall-adjacent"] = safe_nanarg(edge, "min")
    else:
        choices["typical"] = safe_nanarg(areas, "max")
        choices["worst detection"] = safe_nanarg(areas, "min")
        choices["wall-adjacent"] = choices["worst detection"]

    if np.any(~present):
        missing_idx = np.where(~present)[0]
        choices["worst detection"] = int(missing_idx[np.argmin(areas[~present])])

    choices["largest mask"] = safe_nanarg(areas, "max")

    ordered = []
    seen = set()
    for label, idx in choices.items():
        if idx not in seen:
            ordered.append((label, idx))
            seen.add(idx)
    return ordered


def pct_text(value, precision=1):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return "{:.{}f}%".format(value, precision)


def number_text(value, precision=1):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return "{:.{}f}".format(float(value), precision)


def seconds_text(value):
    if value is None:
        return "n/a"
    minutes = float(value) / 60.0
    return "{:.1f} min".format(minutes)


def pass_warn(condition):
    return "PASS" if condition else "WARN"


def style_status_cell(cell, value):
    color = STATUS_LABEL_COLORS.get(value)
    if color is None:
        return
    text = cell.get_text()
    text.set_color(color)
    text.set_fontweight("bold")


def summarize_status(analysis):
    timing = analysis["timing"]
    mask_metrics = analysis["mask_metrics"]
    fps_dev = timing["fps_deviation_pct"]
    fps_ok = fps_dev is not None and fps_dev <= FPS_WARN_PCT
    presence_ok = mask_metrics["presence_pct"] >= PRESENCE_WARN_PCT
    mask_ok = mask_metrics["median_area"] >= analysis["params"]["min_mouse_area"]
    arena_ok = bool(mask_metrics["arena_quality_ok"])
    return {
        "fps": pass_warn(fps_ok),
        "presence": pass_warn(presence_ok),
        "mask": pass_warn(mask_ok),
        "arena": pass_warn(arena_ok),
        "overall": pass_warn(fps_ok and presence_ok and mask_ok and arena_ok),
    }


def analyze_video(
    video_path,
    expected_fps,
    min_height_mm,
    max_height_mm,
    min_mouse_area,
    bg_frame_stride,
    background_median_blur,
    bg_roi_depth_range,
    manual_set_depth_range,
    config,
    arena_reference_area=DEFAULT_ARENA_REFERENCE_AREA,
    arena_area_tolerance=DEFAULT_ARENA_AREA_TOLERANCE,
    arena_circularity_threshold=DEFAULT_ARENA_CIRCULARITY_THRESHOLD,
    arena_hole_diameter_px=DEFAULT_ARENA_HOLE_DIAMETER_PX,
):
    logging.info("Analyzing %s", video_path)
    finfo = get_video_info(video_path, config)
    timing = summarize_timing(finfo, expected_fps, video_path)

    frames_future = None
    if config.prefetch_sampled_frames:
        executor = ThreadPoolExecutor(max_workers=1)
        frames_future = executor.submit(
            read_sampled_frames,
            video_path,
            finfo,
            config,
            expected_fps,
        )
    else:
        executor = None

    try:
        background = compute_background(
            video_path,
            finfo,
            bg_frame_stride=bg_frame_stride,
            background_median_blur=background_median_blur,
            config=config,
        )
        if frames_future is not None:
            frames, frame_indices = frames_future.result()
        else:
            frames, frame_indices = read_sampled_frames(
                video_path, finfo, config, fallback_fps=expected_fps
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if bg_roi_depth_range is None:
        if manual_set_depth_range:
            bg_roi_depth_range = DEFAULT_BG_ROI_DEPTH_RANGE
        else:
            bg_roi_depth_range = auto_bg_roi_depth_range(background)
    arena_roi, unfilled_arena_roi = compute_arena_roi_masks(
        background, bg_roi_depth_range
    )
    arena_quality = compute_arena_mask_quality(
        arena_roi,
        unfilled_arena_roi,
        reference_area=arena_reference_area,
        area_tolerance=arena_area_tolerance,
        circularity_threshold=arena_circularity_threshold,
        hole_diameter_px=arena_hole_diameter_px,
    )

    mask_metrics = analyze_masks(
        frames,
        frame_indices,
        background,
        arena_roi,
        min_height_mm,
        max_height_mm,
        min_mouse_area,
        bg_roi_depth_range,
        mask_workers=config.mask_workers,
    )
    mask_metrics.update(arena_quality)
    analysis = {
        "video_path": video_path,
        "session_dir": video_path.parent,
        "session_name": video_path.parent.name,
        "finfo": finfo,
        "timing": timing,
        "frames": frames,
        "frame_indices": frame_indices,
        "mask_metrics": mask_metrics,
        "representatives": choose_representative_indices(mask_metrics),
        "params": {
            "expected_fps": expected_fps,
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "min_height_mm": min_height_mm,
            "max_height_mm": max_height_mm,
            "min_mouse_area": min_mouse_area,
            "bg_frame_stride": bg_frame_stride,
            "background_median_blur": background_median_blur,
            "bg_roi_depth_range": tuple(float(v) for v in bg_roi_depth_range),
            "manual_set_depth_range": manual_set_depth_range,
            "arena_reference_area": arena_reference_area,
            "arena_area_tolerance": arena_area_tolerance,
            "arena_circularity_threshold": arena_circularity_threshold,
            "arena_hole_diameter_px": arena_hole_diameter_px,
            "threads": config.threads,
            "pixel_format": config.pixel_format,
            "command_timeout": config.command_timeout,
            "prefetch_sampled_frames": config.prefetch_sampled_frames,
            "mask_workers": config.mask_workers,
        },
    }
    analysis["status"] = summarize_status(analysis)
    return analysis


def image_limits(image):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0, 1
    lo, hi = np.percentile(finite, [1, 99])
    if lo == hi:
        hi = lo + 1
    return lo, hi


def floor_relative_height(frame, background):
    heights = background.astype("float32") - frame.astype("float32")
    heights[frame == 0] = np.nan
    heights[heights < 0] = 0
    return heights


def plot_depth_with_mask(
    ax,
    frame,
    mask=None,
    title="",
    background=None,
    height_limits=None,
    colorbar=False,
):
    if background is None:
        image = frame
        vmin, vmax = image_limits(frame)
        cmap = "gray"
        colorbar_label = "Depth from camera (mm)"
    else:
        image = floor_relative_height(frame, background)
        vmin, vmax = height_limits or image_limits(image)
        cmap = colormaps["viridis"].copy()
        cmap.set_bad("black")
        colorbar_label = FRAME_HEIGHT_COLORBAR_LABEL

    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    if mask is not None:
        overlay = np.ma.masked_where(~mask, mask)
        ax.imshow(overlay, cmap="spring", alpha=0.35)
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(colorbar_label, fontsize=7, labelpad=2)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_height_image(
    ax,
    image,
    title="",
    height_limits=None,
    colorbar=False,
    empty_message="n/a",
):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        add_na_plot(ax, title, empty_message)
        return

    vmin, vmax = height_limits or image_limits(image)
    cmap = colormaps["viridis"].copy()
    cmap.set_bad("black")
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(FRAME_HEIGHT_COLORBAR_LABEL, fontsize=7, labelpad=2)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def add_summary_table(ax, analysis):
    ax.axis("off")
    finfo = analysis["finfo"]
    timing = analysis["timing"]
    masks = analysis["mask_metrics"]
    status = analysis["status"]
    rows = [
        ("Overall", status["overall"]),
        ("Frame rate", status["fps"]),
        ("Mouse presence", status["presence"]),
        ("Mouse pixel area", status["mask"]),
        ("Arena mask quality", status["arena"]),
        ("Expected FPS", number_text(timing["expected_fps"], 1)),
        ("Observed FPS", number_text(timing["observed_fps"], 2)),
        ("FPS deviation", pct_text(timing["fps_deviation_pct"], 1)),
        ("Timestamp source", timing["timestamp_source"]),
        ("Frame count", number_text(timing["nframes"], 0)),
        ("Duration", seconds_text(timing["duration"])),
        ("Mouse present", pct_text(masks["presence_pct"], 1)),
        ("Mouse area (px^2)", number_text(masks["median_area"], 0)),
        (
            "Wall refl near/away",
            number_text(masks["wall_near_away_foreground_ratio"], 2),
        ),
        (
            "Wall samples near/away",
            "{} / {}".format(
                masks["wall_near_frame_count"],
                masks["wall_away_frame_count"],
            ),
        ),
        (
            "Wall ignored <{:.0f}px".format(masks["wall_distance_min_threshold"]),
            number_text(masks["wall_excluded_too_close_count"], 0),
        ),
        ("Arena ROI coverage", pct_text(masks["roi_coverage_pct"], 1)),
        ("Arena area (px^2)", number_text(masks["arena_area_px"], 0)),
        ("Arena circularity", number_text(masks["arena_circularity"], 3)),
        ("Arena large holes", number_text(masks["arena_large_hole_count"], 0)),
        (
            "Arena depth range",
            "{:.0f}-{:.0f} mm".format(*masks["bg_roi_depth_range"]),
        ),
        (
            "Video",
            "{}x{} {}".format(finfo["dims"][0], finfo["dims"][1], finfo["codec"]),
        ),
    ]
    table = ax.table(
        cellText=[[key, value] for key, value in rows],
        colLabels=["Metric", "Value"],
        colWidths=[0.58, 0.42],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.08)
    for _, cell in table.get_celld().items():
        cell.set_linewidth(0.25)
    for row_index, (_, value) in enumerate(rows, start=1):
        style_status_cell(table[(row_index, 1)], value)


def generated_at_text():
    return "Generated: {}".format(
        datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    )


def add_generated_at(fig, generated_at):
    if generated_at:
        fig.text(
            0.02,
            0.99,
            generated_at,
            fontsize=7,
            ha="left",
            va="top",
        )


def add_page_header(fig, analysis, suffix="", generated_at=None):
    add_generated_at(fig, generated_at)
    title = "MoSeq Video QC: {}".format(analysis["session_name"])
    if suffix:
        title = "{} - {}".format(title, suffix)
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, y=0.965, ha="left")


def sampled_frame_times_seconds(analysis):
    masks = analysis["mask_metrics"]
    frame_indices = masks["frame_indices"]
    timestamps = analysis["timing"].get("timestamps")
    if timestamps is not None and len(timestamps) > int(np.max(frame_indices)):
        return timestamps[frame_indices] - timestamps[0]

    fps = analysis["finfo"].get("fps") or analysis["params"]["expected_fps"]
    return frame_indices / float(fps)


def append_summary_page(pdf, analysis, generated_at=None):
    masks = analysis["mask_metrics"]
    timing = analysis["timing"]
    sample_times = sampled_frame_times_seconds(analysis)

    fig = plt.figure(figsize=(11, 8.5))
    add_page_header(fig, analysis, "summary", generated_at)
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.2, 1.0, 1.0],
        width_ratios=PDF_SUMMARY_WIDTH_RATIOS,
        bottom=PDF_CONTENT_BOTTOM,
        left=PDF_CONTENT_LEFT,
        right=PDF_CONTENT_RIGHT,
        top=PDF_CONTENT_TOP,
        hspace=PDF_GRID_HSPACE,
        wspace=PDF_SUMMARY_WSPACE,
    )
    ax_table = fig.add_subplot(gs[:, 0])
    add_summary_table(ax_table, analysis)

    ax_fps = fig.add_subplot(gs[0, 1:])
    intervals = timing["intervals"]
    if intervals is not None:
        ax_fps.plot(np.arange(len(intervals)), 1.0 / intervals, lw=0.8)
        ax_fps.axhline(timing["expected_fps"], color="black", lw=0.8, ls="--")
        ax_fps.set_ylabel("Instant FPS")
        ax_fps.set_xlabel("Timestamp interval")
    else:
        ax_fps.bar(
            ["expected", "observed"], [timing["expected_fps"], timing["observed_fps"]]
        )
        ax_fps.set_ylabel("FPS")
    ax_fps.set_title("Observed vs expected frame rate", fontsize=10)

    ax_area = fig.add_subplot(gs[1:, 1:])
    ax_area.plot(sample_times / 60.0, masks["areas"], marker=".", lw=0.8, ms=3)
    ax_area.axhline(
        analysis["params"]["min_mouse_area"],
        color="black",
        lw=0.8,
        ls="--",
        label="Mouse-present threshold",
    )
    ax_area.set_title("Mouse-associated pixel area", fontsize=10)
    ax_area.set_xlabel("Recording time (min)")
    ax_area.set_ylabel("Pixel area (px^2)")
    ax_area.legend(loc="best", fontsize=7, frameon=False)
    pdf.savefig(fig)
    plt.close(fig)


def add_na_plot(ax, title, message="n/a"):
    ax.set_title(title, fontsize=10)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def wall_reflection_groups(masks):
    counts = masks["foreground_pixel_counts"]
    distances = masks["wall_distances"]
    finite = np.logical_and(
        np.logical_and(np.isfinite(counts), np.isfinite(distances)),
        distances >= masks["wall_distance_min_threshold"],
    )
    near = np.logical_and(masks["wall_adjacent"], finite)
    away = np.logical_and(
        masks["present"],
        np.logical_and(finite, ~masks["wall_adjacent"]),
    )
    return counts, distances, near, away


def append_wall_reflections_page(pdf, analysis, generated_at=None):
    masks = analysis["mask_metrics"]
    sample_times = sampled_frame_times_seconds(analysis) / 60.0
    counts, distances, near, away = wall_reflection_groups(masks)
    valid = np.logical_or(near, away)

    fig = plt.figure(figsize=(11, 8.5))
    add_page_header(fig, analysis, "wall reflections", generated_at)
    gs = fig.add_gridspec(
        2,
        2,
        bottom=PDF_CONTENT_BOTTOM,
        left=PDF_WALL_CONTENT_LEFT,
        right=PDF_FRAME_CONTENT_RIGHT,
        top=PDF_CONTENT_TOP,
        hspace=0.4,
        wspace=0.28,
    )

    ax_hist = fig.add_subplot(gs[0, 0])
    if np.any(near) and np.any(away):
        combined = counts[valid]
        if np.nanmin(combined) == np.nanmax(combined):
            bins = 10
        else:
            bins = np.linspace(np.nanmin(combined), np.nanmax(combined), 24)
        ax_hist.hist(
            counts[away],
            bins=bins,
            density=True,
            alpha=0.65,
            label="away",
            color="#2563eb",
        )
        ax_hist.hist(
            counts[near],
            bins=bins,
            density=True,
            alpha=0.65,
            label="wall-adjacent",
            color="#dc2626",
        )
        ax_hist.set_xlabel("Foreground pixels")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(loc="best", fontsize=8, frameon=False)
        ax_hist.set_title("Foreground count distribution", fontsize=10)
    else:
        add_na_plot(
            ax_hist,
            "Foreground count distribution",
            "Need near-wall and away frames",
        )

    ax_time = fig.add_subplot(gs[0, 1])
    if np.any(valid):
        scatter = ax_time.scatter(
            sample_times[valid],
            counts[valid],
            c=distances[valid],
            cmap="viridis",
            s=10,
            alpha=0.85,
        )
        cbar = fig.colorbar(scatter, ax=ax_time, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Wall distance (px)", fontsize=7, labelpad=2)
        ax_time.set_xlabel("Recording time (min)")
        ax_time.set_ylabel("Foreground pixels")
        ax_time.set_title("Foreground count over time", fontsize=10)
    else:
        add_na_plot(ax_time, "Foreground count over time", "No detected mouse frames")

    ax_dist = fig.add_subplot(gs[1, 0])
    if np.any(valid):
        ax_dist.scatter(
            distances[away],
            counts[away],
            s=10,
            alpha=0.70,
            color="#2563eb",
            label="away",
            lw=0,
        )
        ax_dist.scatter(
            distances[near],
            counts[near],
            s=15,
            alpha=0.70,
            color="#dc2626",
            label="wall-adjacent",
            lw=0,
        )
        ax_dist.axvline(
            masks["wall_distance_min_threshold"],
            color="gray",
            lw=0.8,
            ls=":",
            label="min distance",
        )
        ax_dist.axvline(
            EDGE_DISTANCE_THRESHOLD,
            color="black",
            lw=0.8,
            ls="--",
            label="wall cutoff",
        )
        ax_dist.set_xlabel("Minimum arena-wall distance (px)")
        ax_dist.set_ylabel("Foreground pixels")
        ax_dist.legend(loc="best", fontsize=8, frameon=False)
        ax_dist.set_title("Foreground count vs wall distance", fontsize=10)
    else:
        add_na_plot(
            ax_dist,
            "Foreground count vs wall distance",
            "No detected mouse frames",
        )

    ax_summary = fig.add_subplot(gs[1, 1])
    ax_summary.axis("off")
    rows = [
        ("Near-wall frames", number_text(masks["wall_near_frame_count"], 0)),
        ("Away frames", number_text(masks["wall_away_frame_count"], 0)),
        (
            "Near median foreground",
            number_text(masks["wall_near_foreground_median"], 0),
        ),
        (
            "Away median foreground",
            number_text(masks["wall_away_foreground_median"], 0),
        ),
        (
            "Near/away ratio",
            number_text(masks["wall_near_away_foreground_ratio"], 2),
        ),
        (
            "Ignored <{:.0f}px".format(masks["wall_distance_min_threshold"]),
            number_text(masks["wall_excluded_too_close_count"], 0),
        ),
    ]
    table = ax_summary.table(
        cellText=[[key, value] for key, value in rows],
        colLabels=["Metric", "Value"],
        colWidths=[0.58, 0.42],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for _, cell in table.get_celld().items():
        cell.set_linewidth(0.25)
    ax_summary.set_title("Wall-reflection summary", fontsize=10)

    pdf.savefig(fig)
    plt.close(fig)


def frame_panel_title(masks, label, sample_idx):
    frame_no = int(masks["frame_indices"][sample_idx])
    return "{}\nframe {} area {}".format(
        label,
        frame_no,
        int(masks["areas"][sample_idx]),
    )


def wall_superposition_title(masks):
    return "wall-adjacent p{:.0f}\n{} frames".format(
        masks["wall_adjacent_height_percentile"],
        masks["wall_near_frame_count"],
    )


def frames_page_panels(analysis):
    panels = []
    wall_panel = ("wall-superposition", None)
    inserted_wall_panel = False

    for label, sample_idx in analysis["representatives"]:
        panels.append(("frame", (label, sample_idx)))
        if label == "largest mask":
            panels.append(wall_panel)
            inserted_wall_panel = True

    if not inserted_wall_panel:
        panels.append(wall_panel)
    return panels[:6]


def append_frames_page(pdf, analysis, generated_at=None):
    masks = analysis["mask_metrics"]

    fig = plt.figure(figsize=(11, 8.5))
    add_page_header(fig, analysis, "frames", generated_at)
    gs = fig.add_gridspec(
        2,
        4,
        bottom=PDF_CONTENT_BOTTOM,
        left=PDF_CONTENT_LEFT,
        right=PDF_FRAME_CONTENT_RIGHT,
        top=PDF_CONTENT_TOP,
        hspace=PDF_GRID_HSPACE,
        wspace=PDF_GRID_WSPACE,
    )

    ax_bg = fig.add_subplot(gs[0, 0])
    plot_depth_with_mask(ax_bg, masks["background"], None, "Median background")
    ax_mask = fig.add_subplot(gs[1, 0])
    ax_mask.imshow(masks["arena_roi"], cmap="gray")
    ax_mask.set_title("Arena ROI mask", fontsize=9)
    ax_mask.set_xticks([])
    ax_mask.set_yticks([])

    panels = frames_page_panels(analysis)
    for pos, (panel_type, payload) in enumerate(panels):
        row = pos // 3
        col = pos % 3 + 1
        ax = fig.add_subplot(gs[row, col])
        if panel_type == "wall-superposition":
            plot_height_image(
                ax,
                masks["wall_adjacent_height_p99"],
                wall_superposition_title(masks),
                height_limits=FRAME_HEIGHT_DISPLAY_RANGE_MM,
                colorbar=True,
                empty_message="No wall-adjacent frames",
            )
        else:
            label, sample_idx = payload
            plot_depth_with_mask(
                ax,
                analysis["frames"][sample_idx],
                masks["component_masks"][sample_idx],
                frame_panel_title(masks, label, sample_idx),
                background=masks["background"],
                height_limits=FRAME_HEIGHT_DISPLAY_RANGE_MM,
                colorbar=True,
            )

    pdf.savefig(fig)
    plt.close(fig)


def append_report_pages(pdf, analysis, generated_at=None):
    append_summary_page(pdf, analysis, generated_at)
    append_wall_reflections_page(pdf, analysis, generated_at)
    append_frames_page(pdf, analysis, generated_at)


def write_session_pdf(analysis, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at_text()
    with PdfPages(str(report_path)) as pdf:
        append_report_pages(pdf, analysis, generated_at)
    logging.info("Wrote %s", report_path)


def batch_summary_row(analysis):
    timing = analysis["timing"]
    masks = analysis["mask_metrics"]
    return [
        analysis["session_name"],
        analysis["status"]["overall"],
        analysis["status"]["fps"],
        analysis["status"]["presence"],
        analysis["status"]["arena"],
        number_text(timing["observed_fps"], 2),
        pct_text(masks["presence_pct"], 1),
        number_text(masks["median_area"], 0),
    ]


def write_batch_overview_pdf(summary_rows, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at_text()
    with PdfPages(str(report_path)) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        add_generated_at(fig, generated_at)
        gs = fig.add_gridspec(
            1,
            1,
            bottom=PDF_CONTENT_BOTTOM,
            left=PDF_CONTENT_LEFT,
            right=PDF_CONTENT_RIGHT,
            top=PDF_CONTENT_TOP,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax.axis("off")
        table = ax.table(
            cellText=summary_rows,
            colLabels=[
                "Session",
                "Overall",
                "FPS",
                "Presence",
                "Arena",
                "Obs FPS",
                "Present",
                "Med mouse area",
            ],
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        font_size = (
            8 if len(summary_rows) <= 24 else 7 if len(summary_rows) <= 36 else 6
        )
        table.set_fontsize(font_size)
        table.scale(1.0, 1.25)
        for row_index, row in enumerate(summary_rows, start=1):
            for col_index in (1, 2, 3, 4):
                style_status_cell(table[(row_index, col_index)], row[col_index])
        ax.set_title("MoSeq Video QC Batch Summary", fontsize=14, fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)


def write_batch_pdf(summary_rows, report_paths, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise RuntimeError(
            "Batch PDF concatenation requires pypdf. Install video-qc requirements "
            "or run `pip install pypdf`."
        ) from exc

    overview_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="video_qc_batch_overview_",
            suffix=".pdf",
            dir=str(report_path.parent),
            delete=False,
        ) as tmp:
            overview_path = Path(tmp.name)
        write_batch_overview_pdf(summary_rows, overview_path)

        writer = PdfWriter()
        # Keep source readers alive until write; pages can reference source objects.
        readers = []
        for source_pdf_path in [overview_path] + list(report_paths):
            reader = PdfReader(str(source_pdf_path))
            readers.append(reader)
            for page in reader.pages:
                writer.add_page(page)

        with open(str(report_path), "wb") as f:
            writer.write(f)
    finally:
        if overview_path is not None:
            try:
                overview_path.unlink()
            except OSError:
                logging.warning("Could not remove temporary %s", overview_path)
    logging.info("Wrote %s", report_path)


def sanitize_name(name):
    keep = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            keep.append(char)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "session"


def resolve_video_paths(raw_path):
    path = Path(raw_path).expanduser()
    direct_video = path / DEFAULT_VIDEO_NAME
    if direct_video.is_file():
        return [direct_video.resolve()]

    videos = sorted(
        video.resolve()
        for video in path.glob("*/{}".format(DEFAULT_VIDEO_NAME))
        if video.is_file()
    )

    if not videos:
        raise click.ClickException(
            "Input must be a recording folder containing {video} or a parent folder "
            "whose immediate child folders contain {video}: {path}".format(
                video=DEFAULT_VIDEO_NAME,
                path=path,
            )
        )
    return videos


def session_report_path(video_path, output_dir):
    report_dir = report_output_dir(output_dir)
    session_name = sanitize_name(video_path.parent.name)
    return report_dir / "{}_video_qc.pdf".format(session_name)


def report_output_dir(output_dir):
    output_dir = Path(output_dir)
    if output_dir.is_absolute():
        return output_dir
    return Path.cwd() / output_dir


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--expected-fps",
    default=30.0,
    type=float,
    show_default=True,
    help="Expected recording frame rate in Hz.",
)
@click.option(
    "--output-dir",
    default=".",
    type=click.Path(file_okay=False),
    show_default=True,
    help="Report output folder. Relative paths are resolved from the current working directory.",
)
@click.option(
    "--min-height-mm",
    default=10.0,
    type=float,
    show_default=True,
    help="Minimum mouse height above the arena floor included in foreground masks.",
)
@click.option(
    "--max-height-mm",
    default=120.0,
    type=float,
    show_default=True,
    help="Maximum mouse height above the arena floor included in foreground masks.",
)
@click.option(
    "--min-mouse-area",
    default=500,
    type=int,
    show_default=True,
    help="Minimum mouse-associated pixel area, in px^2, counted as mouse present.",
)
@click.option(
    "--bg-frame-stride",
    default=250,
    type=int,
    show_default=True,
    help="Frame stride used to sample video frames for the median background.",
)
@click.option(
    "--background-median-blur",
    default=5,
    type=int,
    show_default=True,
    help="Odd OpenCV median blur kernel size applied to background sample frames.",
)
@click.option(
    "--bg-roi-depth-range",
    default=None,
    type=(float, float),
    help="Manual low/high depth range, in mm, used to fit the arena floor plane.",
)
@click.option(
    "--arena-reference-area",
    default=DEFAULT_ARENA_REFERENCE_AREA,
    type=float,
    show_default=True,
    help="Expected arena mask pixel area for arena quality checks.",
)
@click.option(
    "--arena-area-tolerance",
    default=DEFAULT_ARENA_AREA_TOLERANCE,
    type=float,
    show_default=True,
    help="Allowed fractional deviation from the expected arena mask area.",
)
@click.option(
    "--arena-circularity-threshold",
    default=DEFAULT_ARENA_CIRCULARITY_THRESHOLD,
    type=float,
    show_default=True,
    help="Minimum arena mask circularity score for arena quality checks.",
)
@click.option(
    "--arena-hole-diameter-px",
    default=DEFAULT_ARENA_HOLE_DIAMETER_PX,
    type=float,
    show_default=True,
    help="Minimum enclosed hole diameter, in pixels, counted against arena quality.",
)
@click.option(
    "--manual-set-depth-range",
    is_flag=True,
    help="Use the default or supplied bg ROI depth range instead of auto-detecting it.",
)
@click.option(
    "--threads",
    default=6,
    type=int,
    show_default=True,
    help="Threads passed to FFmpeg/FFprobe.",
)
@click.option(
    "--pixel-format",
    default="gray16le",
    show_default=True,
    help="FFmpeg pixel format for depth frames.",
)
@click.option(
    "--command-timeout",
    default=60,
    type=int,
    show_default=True,
    help="Seconds to wait for each FFmpeg/FFprobe command before failing that video.",
)
@click.option(
    "--prefetch-sampled-frames",
    is_flag=True,
    help="Read sampled frames concurrently with background frame processing.",
)
@click.option(
    "--mask-workers",
    default=1,
    type=int,
    show_default=True,
    help="Worker threads used for per-frame mask component analysis.",
)
@click.option(
    "--batch-workers",
    default=1,
    type=int,
    show_default=True,
    help="Worker threads used to process multiple videos concurrently.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop the batch on the first video error.",
)
def main(
    path,
    expected_fps,
    output_dir,
    min_height_mm,
    max_height_mm,
    min_mouse_area,
    bg_frame_stride,
    background_median_blur,
    bg_roi_depth_range,
    arena_reference_area,
    arena_area_tolerance,
    arena_circularity_threshold,
    arena_hole_diameter_px,
    manual_set_depth_range,
    threads,
    pixel_format,
    command_timeout,
    prefetch_sampled_frames,
    mask_workers,
    batch_workers,
    fail_fast,
):
    """
    Generate fast PDF QC reports for one recording folder or one parent folder.
    """
    videos = resolve_video_paths(path)
    out_dir = report_output_dir(output_dir)
    configure_logging(out_dir / "video_qc.log")

    logging.info("Found %s video(s)", len(videos))
    config = FrameIOConfig(
        pixel_format=pixel_format,
        threads=threads,
        command_timeout=command_timeout,
        prefetch_sampled_frames=prefetch_sampled_frames,
        mask_workers=max(1, int(mask_workers)),
    )
    summary_rows = []
    report_paths = []
    failures = []

    def process_video(video_path):
        analysis = analyze_video(
            video_path,
            expected_fps=expected_fps,
            min_height_mm=min_height_mm,
            max_height_mm=max_height_mm,
            min_mouse_area=min_mouse_area,
            bg_frame_stride=bg_frame_stride,
            background_median_blur=background_median_blur,
            bg_roi_depth_range=bg_roi_depth_range,
            manual_set_depth_range=manual_set_depth_range,
            config=config,
            arena_reference_area=arena_reference_area,
            arena_area_tolerance=arena_area_tolerance,
            arena_circularity_threshold=arena_circularity_threshold,
            arena_hole_diameter_px=arena_hole_diameter_px,
        )
        report_path = session_report_path(video_path, output_dir)
        write_session_pdf(analysis, report_path)
        return {
            "summary_row": batch_summary_row(analysis),
            "report_path": report_path,
            "message": "{} -> {}".format(video_path, report_path),
        }

    worker_count = max(1, int(batch_workers))
    if worker_count == 1:
        work_items = []
        for video_path in tqdm(videos, desc="Analyzing videos"):
            try:
                work_items.append(process_video(video_path))
            except Exception as exc:
                logging.exception("Failed to analyze %s", video_path)
                failures.append((video_path, str(exc)))
                click.echo("FAILED {}: {}".format(video_path, exc), err=True)
                if fail_fast:
                    raise
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                (video_path, executor.submit(process_video, video_path))
                for video_path in videos
            ]
            work_items = []
            for video_path, future in tqdm(futures, desc="Analyzing videos"):
                try:
                    work_items.append(future.result())
                except Exception as exc:
                    logging.exception("Failed to analyze %s", video_path)
                    failures.append((video_path, str(exc)))
                    click.echo("FAILED {}: {}".format(video_path, exc), err=True)
                    if fail_fast:
                        raise

    for item in work_items:
        click.echo(item["message"])
        summary_rows.append(item["summary_row"])
        report_paths.append(item["report_path"])

    if len(report_paths) > 1:
        batch_path = out_dir / "video_qc_batch_report.pdf"
        write_batch_pdf(summary_rows, report_paths, batch_path)
        click.echo("Batch report -> {}".format(batch_path))

    if failures:
        failure_path = out_dir / "video_qc_failures.txt"
        with open(str(failure_path), "w") as f:
            for video_path, message in failures:
                f.write("{}\t{}\n".format(video_path, message))
        click.echo(
            "Completed with {} failed video(s). See {}".format(
                len(failures), failure_path
            ),
            err=True,
        )
        logging.warning("Completed with %s failed video(s)", len(failures))

    if not report_paths:
        raise click.ClickException("No videos were successfully analyzed.")

    logging.info("Done")


if __name__ == "__main__":
    main()  # pyright: ignore
