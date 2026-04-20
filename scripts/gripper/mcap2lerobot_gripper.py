#!/usr/bin/env python3
"""
Convert DAS-Gripper MCAP files to LeRobot v2.1 dataset format (LeRobot >= 0.3.3).

Each MCAP file becomes one episode. Syncs to the controller camera timestamps at ~30Hz.
State = controller eef_pose + gripper (8D), Action = controlled eef_pose + gripper (8D).

Usage:
    python mcap2lerobot.py file1.mcap -o output_dataset
    python mcap2lerobot.py recordings -o output_dataset
    python mcap2lerobot.py *.mcap -o output_dataset --fps 30

Requirements:
    pip install mcap mcap-protobuf-support pyarrow numpy av
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from fractions import Fraction

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

STATE_DIM = 8
STATE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
CHUNKS_SIZE = 1000


def default_log(message):
    print(message)


def parse_args():
    p = argparse.ArgumentParser(description="Convert DAS-Gripper MCAP to LeRobot v2.1")
    p.add_argument(
        "mcap_files",
        nargs="+",
        help="Input MCAP file(s) and/or directories containing .mcap files",
    )
    p.add_argument("-o", "--output", required=True, help="Output dataset directory")
    p.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    p.add_argument(
        "--description",
        "--task",
        dest="description",
        default="Teleoperation",
        help="Human-readable dataset description",
    )
    p.add_argument("--robot-type", default="das_gripper", help="Robot type")
    p.add_argument(
        "--controller-prefix",
        "--leader",
        dest="leader",
        default="robot0",
        help="Prefix for the controller robot in MCAP topics",
    )
    p.add_argument(
        "--controlled-prefix",
        "--follower",
        dest="follower",
        default="robot1",
        help="Prefix for the controlled robot in MCAP topics",
    )
    return p.parse_args()


def resolve_mcap_inputs(inputs):
    """Expand file and directory inputs into a sorted list of .mcap files."""
    resolved = []
    seen = set()

    for input_path in inputs:
        if os.path.isdir(input_path):
            dir_entries = sorted(
                os.path.join(input_path, name)
                for name in os.listdir(input_path)
                if name.lower().endswith(".mcap")
            )
            if not dir_entries:
                raise ValueError(f"No .mcap files found in directory: {input_path}")
            for file_path in dir_entries:
                norm_path = os.path.abspath(file_path)
                if norm_path not in seen:
                    seen.add(norm_path)
                    resolved.append(file_path)
            continue

        if not os.path.isfile(input_path):
            raise ValueError(f"Input path does not exist or is not a file: {input_path}")
        if not input_path.lower().endswith(".mcap"):
            raise ValueError(f"Input file is not an .mcap file: {input_path}")

        norm_path = os.path.abspath(input_path)
        if norm_path not in seen:
            seen.add(norm_path)
            resolved.append(input_path)

    if not resolved:
        raise ValueError("No .mcap files were provided")

    return sorted(resolved)


# ---------------------------------------------------------------------------
# MCAP reading
# ---------------------------------------------------------------------------

def read_mcap_messages(mcap_path):
    """Read all decoded messages from MCAP, grouped by topic and sorted by time."""
    topics = defaultdict(list)
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages():
            topics[channel.topic].append((message.log_time, decoded))
    for topic in topics:
        topics[topic].sort(key=lambda x: x[0])
    return topics


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def extract_pose(msg):
    """Extract [x,y,z,qx,qy,qz,qw] from PoseInFrame."""
    p = msg.pose
    return [
        p.position.x, p.position.y, p.position.z,
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
    ]


def nearest_lookup(target_ts, source_entries):
    """For each target timestamp (ns int64), return the nearest source value."""
    if not source_entries:
        return [None] * len(target_ts)
    src_ts = np.array([t for t, _ in source_entries], dtype=np.int64)
    src_vals = [v for _, v in source_entries]
    target_arr = np.array(target_ts, dtype=np.int64)
    idx = np.searchsorted(src_ts, target_arr, side="left")
    results = []
    for i, j in enumerate(idx):
        if j <= 0:
            results.append(src_vals[0])
        elif j >= len(src_ts):
            results.append(src_vals[-1])
        elif abs(src_ts[j] - target_arr[i]) <= abs(src_ts[j - 1] - target_arr[i]):
            results.append(src_vals[j])
        else:
            results.append(src_vals[j - 1])
    return results


def find_first_keyframe(frames):
    """Return index of first h264 keyframe (SPS/IDR) in CompressedImage list."""
    for i, (_ts, msg) in enumerate(frames):
        data = bytes(msg.data)
        if len(data) < 5:
            continue
        # Search for Annex-B start codes followed by SPS(0x67) or IDR(0x65)
        for sc in (b"\x00\x00\x00\x01", b"\x00\x00\x01"):
            for nal in (0x67, 0x65):
                if sc + bytes([nal]) in data[:100]:
                    return i
    return 0


# ---------------------------------------------------------------------------
# Video writing (h264 remux into mp4)
# ---------------------------------------------------------------------------

def write_h264_mp4(h264_data_list, output_path, fps):
    """Decode h264 and re-encode into mp4. Returns (frame_count, width, height)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write concatenated h264 bitstream to temp file
    tmp_path = output_path + ".tmp.h264"
    vid_w, vid_h = 640, 480
    try:
        with open(tmp_path, "wb") as f:
            for data in h264_data_list:
                f.write(data)

        inp = av.open(tmp_path, format="h264")
        in_stream = inp.streams.video[0]

        out = av.open(output_path, mode="w")
        vid_w = in_stream.codec_context.width or 640
        vid_h = in_stream.codec_context.height or 480
        out_stream = out.add_stream("libx264", rate=fps)
        out_stream.width = vid_w
        out_stream.height = vid_h
        out_stream.pix_fmt = "yuv420p"
        out_stream.time_base = Fraction(1, fps)

        frame_count = 0
        for frame in inp.decode(in_stream):
            frame.pts = frame_count
            frame.time_base = Fraction(1, fps)
            for packet in out_stream.encode(frame):
                out.mux(packet)
            frame_count += 1

        # Flush encoder
        for packet in out_stream.encode():
            out.mux(packet)

        out.close()
        inp.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return frame_count, vid_w, vid_h


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(values):
    """Compute per-dimension {min, max, mean, std} for a list of vectors."""
    arr = np.array(values, dtype=np.float64)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }


def _scalar_stats(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
    }


# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------

def process_episode(mcap_path, episode_idx, global_offset, output_dir, fps, leader, follower, task_idx, log_fn=default_log):
    """Convert one MCAP file → one episode. Returns (num_frames, episode_stats)."""
    log_fn(f"Episode {episode_idx}: {os.path.basename(mcap_path)}")
    topics = read_mcap_messages(mcap_path)

    # Camera resolution
    cam_info = topics.get(f"/{leader}/sensor/camera0/camera_info", [])
    width, height = 640, 480
    if cam_info:
        width, height = cam_info[0][1].width, cam_info[0][1].height

    # Camera frames – trim to first keyframe
    leader_cam = topics.get(f"/{leader}/sensor/camera0/compressed", [])
    follower_cam = topics.get(f"/{follower}/sensor/camera0/compressed", [])
    lc_kf = find_first_keyframe(leader_cam)
    fc_kf = find_first_keyframe(follower_cam)
    leader_cam = leader_cam[lc_kf:]
    follower_cam = follower_cam[fc_kf:]
    log_fn(f"  Cameras after keyframe trim: leader={len(leader_cam)}, follower={len(follower_cam)}")

    # EEF pose & gripper
    leader_eef = topics.get(f"/{leader}/vio/eef_pose", [])
    follower_eef = topics.get(f"/{follower}/vio/eef_pose", [])
    leader_grip = topics.get(f"/{leader}/sensor/magnetic_encoder", [])
    follower_grip = topics.get(f"/{follower}/sensor/magnetic_encoder", [])

    # Common time range across key streams
    streams = [s for s in [leader_cam, follower_cam, leader_eef, follower_eef] if s]
    if not streams:
        log_fn("  WARNING: no data streams found, skipping")
        return 0, None
    t_start = max(s[0][0] for s in streams)
    t_end = min(s[-1][0] for s in streams)

    # Master timeline = leader camera timestamps in common range
    master_ts = [ts for ts, _ in leader_cam if t_start <= ts <= t_end]
    N = len(master_ts)
    log_fn(f"  Time range: {(t_end - t_start) / 1e9:.2f}s, {N} master frames")

    if N == 0:
        log_fn("  WARNING: no overlapping frames, skipping")
        return 0, None

    # Sync all signals to master timestamps
    leader_eef_vals = nearest_lookup(master_ts, [(t, extract_pose(m)) for t, m in leader_eef])
    follower_eef_vals = nearest_lookup(master_ts, [(t, extract_pose(m)) for t, m in follower_eef])
    leader_grip_vals = nearest_lookup(master_ts, [(t, m.value) for t, m in leader_grip])
    follower_grip_vals = nearest_lookup(master_ts, [(t, m.value) for t, m in follower_grip])

    states = [eef + [g] for eef, g in zip(leader_eef_vals, leader_grip_vals)]
    actions = [eef + [g] for eef, g in zip(follower_eef_vals, follower_grip_vals)]

    # Match camera frames to master timestamps
    leader_cam_data = [bytes(m.data) for m in nearest_lookup(master_ts, leader_cam)]
    follower_cam_data = [bytes(m.data) for m in nearest_lookup(master_ts, follower_cam)]

    # Write videos
    chunk = f"chunk-{episode_idx // CHUNKS_SIZE:03d}"
    ep_str = f"episode_{episode_idx:06d}"

    lv_key = f"observation.images.{leader}_camera"
    fv_key = f"observation.images.{follower}_camera"

    vid_l = os.path.join(output_dir, "videos", chunk, lv_key, f"{ep_str}.mp4")
    vid_f = os.path.join(output_dir, "videos", chunk, fv_key, f"{ep_str}.mp4")

    n_l, vid_w, vid_h = write_h264_mp4(leader_cam_data, vid_l, fps)
    n_f, _, _ = write_h264_mp4(follower_cam_data, vid_f, fps)
    log_fn(f"  Video frames written: leader={n_l}, follower={n_f}, resolution={vid_w}x{vid_h}")

    # Align to minimum available frames
    N_final = min(N, n_l, n_f)
    if N_final < N:
        log_fn(f"  Trimming to {N_final} frames (video shorter than master)")
    master_ts = master_ts[:N_final]
    states = states[:N_final]
    actions = actions[:N_final]

    # Relative timestamps (seconds from episode start)
    t0 = master_ts[0]
    timestamps = [(t - t0) / 1e9 for t in master_ts]

    # Build parquet table
    global_indices = list(range(global_offset, global_offset + N_final))
    table = pa.table({
        "observation.state": pa.FixedSizeListArray.from_arrays(
            pa.array(np.array(states, dtype=np.float32).flatten(), type=pa.float32()), STATE_DIM,
        ),
        "action": pa.FixedSizeListArray.from_arrays(
            pa.array(np.array(actions, dtype=np.float32).flatten(), type=pa.float32()), STATE_DIM,
        ),
        lv_key: pa.array([True] * N_final, type=pa.bool_()),
        fv_key: pa.array([True] * N_final, type=pa.bool_()),
        "timestamp": pa.array(timestamps, type=pa.float32()),
        "frame_index": pa.array(list(range(N_final)), type=pa.int64()),
        "episode_index": pa.array([episode_idx] * N_final, type=pa.int64()),
        "index": pa.array(global_indices, type=pa.int64()),
        "task_index": pa.array([task_idx] * N_final, type=pa.int64()),
    })

    data_dir = os.path.join(output_dir, "data", chunk)
    os.makedirs(data_dir, exist_ok=True)
    pq.write_table(table, os.path.join(data_dir, f"{ep_str}.parquet"))

    # Per-episode stats
    ep_stats = {
        "episode_index": episode_idx,
        "stats": {
            "observation.state": compute_stats(states),
            "action": compute_stats(actions),
            "timestamp": _scalar_stats(timestamps),
            "frame_index": _scalar_stats(list(range(N_final))),
            "episode_index": _scalar_stats([episode_idx] * N_final),
            "index": _scalar_stats(global_indices),
            "task_index": _scalar_stats([task_idx] * N_final),
        },
    }

    return N_final, ep_stats, vid_w, vid_h


# ---------------------------------------------------------------------------
# Metadata writing
# ---------------------------------------------------------------------------

def write_metadata(output_dir, episodes_info, fps, robot_type, description, leader, follower, width, height):
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    total_frames = sum(e["length"] for e in episodes_info)
    total_episodes = len(episodes_info)

    lv_key = f"observation.images.{leader}_camera"
    fv_key = f"observation.images.{follower}_camera"

    video_info = {
        "video.fps": fps,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "fps": fps,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": STATE_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": STATE_NAMES,
            },
            lv_key: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "video_info": video_info,
            },
            fv_key: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "video_info": video_info,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * 2,
        "total_chunks": (total_episodes - 1) // CHUNKS_SIZE + 1 if total_episodes else 1,
        "chunks_size": CHUNKS_SIZE,
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "splits": {"train": f"0:{total_episodes}"},
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # tasks.jsonl
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": description}) + "\n")

    # episodes.jsonl
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep in episodes_info:
            f.write(json.dumps({
                "episode_index": ep["episode_index"],
                "tasks": [description],
                "length": ep["length"],
            }) + "\n")

    # episodes_stats.jsonl
    with open(os.path.join(meta_dir, "episodes_stats.jsonl"), "w") as f:
        for ep in episodes_info:
            f.write(json.dumps(ep["stats_entry"]) + "\n")


def convert_mcap_to_lerobot(
    mcap_inputs,
    output_dir,
    fps=30,
    description="Teleoperation",
    robot_type="das_gripper",
    leader="robot0",
    follower="robot1",
    log_fn=default_log,
    task=None,
):
    if task is not None:
        description = task

    mcap_files = resolve_mcap_inputs(mcap_inputs)
    os.makedirs(output_dir, exist_ok=True)

    episodes_info = []
    global_offset = 0
    width, height = 640, 480

    for episode_index, mcap_path in enumerate(mcap_files):
        result = process_episode(
            mcap_path,
            episode_index,
            global_offset,
            output_dir,
            fps,
            leader,
            follower,
            task_idx=0,
            log_fn=log_fn,
        )
        n_frames, ep_stats, width, height = result
        if n_frames == 0:
            log_fn("  Skipped (no frames)")
            continue
        episodes_info.append({
            "episode_index": episode_index,
            "length": n_frames,
            "stats_entry": ep_stats,
        })
        global_offset += n_frames

    if not episodes_info:
        raise RuntimeError("No episodes produced.")

    write_metadata(output_dir, episodes_info, fps, robot_type, description, leader, follower, width, height)

    total_frames = sum(episode["length"] for episode in episodes_info)
    log_fn(f"\nDone: {len(episodes_info)} episodes, {total_frames} total frames -> {output_dir}")
    return {
        "episodes": len(episodes_info),
        "frames": total_frames,
        "output_dir": output_dir,
        "files": mcap_files,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    try:
        convert_mcap_to_lerobot(
            mcap_inputs=args.mcap_files,
            output_dir=args.output,
            fps=args.fps,
            description=args.description,
            robot_type=args.robot_type,
            leader=args.leader,
            follower=args.follower,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
