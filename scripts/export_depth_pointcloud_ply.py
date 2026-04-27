import argparse
import os
import os.path as osp
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mcaploader import McapLoader


def write_ply_ascii(points: np.ndarray, output_path: str):
    """
    Write xyz point cloud to ascii ply file.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape should be [N, 3], got {points.shape}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def export_depth_point_cloud_ply(mcap_file: str, args):
    bag = McapLoader(mcap_file)
    topic_data = bag.get_topic_data(args.depth_topic)
    if topic_data is None or len(topic_data) == 0:
        print(f"No depth data found on topic: {args.depth_topic}")
        return

    output_dir = args.output_dir
    if output_dir == "":
        base_name = osp.splitext(osp.basename(mcap_file))[0]
        output_dir = osp.join(osp.dirname(mcap_file), f"{base_name}_depth_ply")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Find {len(topic_data)} depth frames, start export ply to: {output_dir}")
    saved_count = 0
    skip_count = 0
    for idx, d in enumerate(topic_data):
        depth_data = d.get("decode_data", None)
        if depth_data is None:
            skip_count += 1
            continue

        points = bag.convert_depth_to_point_cloud(
            depth_data=depth_data,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            pixel_stride=args.pixel_stride,
            stereo_calibration_topic=args.stereo_calibration_topic,
        )

        if points.shape[0] == 0:
            skip_count += 1
            continue

        if "data" in d and hasattr(d["data"], "header"):
            ts = d["data"].header.timestamp
        else:
            ts = idx
        output_ply = osp.join(output_dir, f"{ts}.ply")
        write_ply_ascii(points, output_ply)
        saved_count += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(topic_data):
            print(f"  Processed {idx + 1}/{len(topic_data)} frames, saved={saved_count}, skipped={skip_count}")

    print(f"Done. Saved {saved_count} ply files, skipped {skip_count} frames.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export depth topic in mcap to per-frame point cloud ply files."
    )
    parser.add_argument("mcap_file", type=str, help="Input mcap file path")
    parser.add_argument(
        "--depth-topic",
        type=str,
        default="/robot0/sensor/depth/compressed",
        help="Depth topic name",
    )
    parser.add_argument(
        "--stereo-calibration-topic",
        type=str,
        default="/robot0/sensor/depth/stereo_calibration",
        help="Stereo calibration topic name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for ply files. default: <mcap_name>_depth_ply",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.2,
        help="Minimum valid depth in meter",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=2.0,
        help="Maximum valid depth in meter, <=0 means disable upper bound",
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=1,
        help="Pixel stride when generating point cloud",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if (not osp.exists(args.mcap_file)) or (not osp.isfile(args.mcap_file)):
        raise FileNotFoundError(f"mcap file not found: {args.mcap_file}")
    export_depth_point_cloud_ply(args.mcap_file, args)
