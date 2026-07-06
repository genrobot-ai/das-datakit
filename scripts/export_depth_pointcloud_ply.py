import argparse
from bisect import bisect_left
import os
import os.path as osp
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mcaploader import McapLoader


def write_ply_ascii(points: np.ndarray, output_path: str, colors: np.ndarray = None):
    """
    Write xyz or xyzrgb point cloud to ascii ply file.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape should be [N, 3], got {points.shape}")
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape != points.shape:
            raise ValueError(f"colors shape should match points [N, 3], got {colors.shape}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def get_topic_timestamp(topic_item: dict, default: int = 0) -> int:
    proto = topic_item.get("data", None)
    if proto is not None and hasattr(proto, "header"):
        ts = int(getattr(proto.header, "timestamp", 0) or 0)
        if ts > 0:
            return ts
    return int(topic_item.get("log_time", default))


def load_rgb_frames(bag: McapLoader, rgb_topic: str):
    rgb_topic_data = bag.get_topic_data(rgb_topic)
    if rgb_topic_data is None or len(rgb_topic_data) == 0:
        return [], {}

    rgb_by_ts = {}
    for idx, d in enumerate(rgb_topic_data):
        img_bgr = d.get("decode_data", None)
        if img_bgr is None:
            continue
        img_bgr = np.asarray(img_bgr)
        if img_bgr.ndim != 3 or img_bgr.shape[-1] < 3:
            continue
        ts = get_topic_timestamp(d, idx)
        rgb_by_ts[ts] = img_bgr[:, :, :3][:, :, ::-1].copy()
    rgb_timestamps = sorted(rgb_by_ts)
    return rgb_timestamps, rgb_by_ts


def nearest_rgb(rgb_timestamps: list, rgb_by_ts: dict, ts: int, threshold_ns: int):
    if not rgb_timestamps:
        return None
    insert_idx = bisect_left(rgb_timestamps, ts)
    candidates = []
    if insert_idx < len(rgb_timestamps):
        candidates.append(rgb_timestamps[insert_idx])
    if insert_idx > 0:
        candidates.append(rgb_timestamps[insert_idx - 1])
    best = min(candidates, key=lambda x: abs(x - ts))
    if abs(best - ts) > threshold_ns:
        return None
    return rgb_by_ts[best]


def preprocess_rgb_for_depth(rgb_img: np.ndarray, depth_shape_hw: tuple, crop_ratio: float = 1.0) -> np.ndarray:
    h, w = depth_shape_hw
    crop_ratio = float(crop_ratio)
    if crop_ratio <= 0 or crop_ratio > 1:
        raise ValueError(f"rgb_crop_ratio should be in (0, 1], got {crop_ratio}")

    if crop_ratio < 0.999:
        src_h, src_w = rgb_img.shape[:2]
        crop_h = max(1, int(round(src_h * crop_ratio)))
        crop_w = max(1, int(round(src_w * crop_ratio)))
        y0 = max(0, (src_h - crop_h) // 2)
        x0 = max(0, (src_w - crop_w) // 2)
        rgb_img = rgb_img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(rgb_img, (w, h), interpolation=cv2.INTER_LINEAR)


def export_depth_point_cloud_ply(mcap_file: str, args):
    bag = McapLoader(mcap_file)
    use_camera2_depth = args.depth_topic == McapLoader.CAMERA2_DEPTH_TOPIC
    if args.export_rgb_ply and not use_camera2_depth:
        raise ValueError(
            f"--export-rgb-ply currently requires --depth-topic {McapLoader.CAMERA2_DEPTH_TOPIC}"
        )

    topic_data = bag.get_topic_data(args.depth_topic)
    if topic_data is None or len(topic_data) == 0:
        print(f"No depth data found on topic: {args.depth_topic}")
        return

    rgb_timestamps = []
    rgb_by_ts = {}
    if args.export_rgb_ply:
        rgb_timestamps, rgb_by_ts = load_rgb_frames(bag, args.rgb_topic)
        if not rgb_timestamps:
            print(f"No RGB data found on topic: {args.rgb_topic}, fallback to xyz ply.")

    output_dir = args.output_dir
    if output_dir == "":
        base_name = osp.splitext(osp.basename(mcap_file))[0]
        output_dir = osp.join(osp.dirname(mcap_file), f"{base_name}_depth_ply")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Find {len(topic_data)} depth frames, start export ply to: {output_dir}")
    saved_count = 0
    skip_count = 0
    no_color_count = 0
    for idx, d in enumerate(topic_data):
        if args.max_frames > 0 and idx >= args.max_frames:
            break
        depth_data = d.get("decode_data", None)
        if depth_data is None:
            skip_count += 1
            continue

        colors = None
        if use_camera2_depth:
            if args.export_rgb_ply:
                points, yy, xx = bag.convert_camera2_depth_to_point_cloud(
                    depth_data=depth_data,
                    min_ray_z=args.min_ray_z,
                    max_range=args.max_range,
                    pixel_stride=args.pixel_stride,
                    depth_info_topic=args.depth_info_topic,
                    return_pixel_indices=True,
                )
                ts = get_topic_timestamp(d, idx)
                rgb_img = nearest_rgb(
                    rgb_timestamps,
                    rgb_by_ts,
                    ts,
                    int(float(args.rgb_match_threshold_ms) * 1e6),
                )
                if rgb_img is not None:
                    rgb_depth = preprocess_rgb_for_depth(
                        rgb_img,
                        depth_data.shape[:2],
                        crop_ratio=args.rgb_crop_ratio,
                    )
                    colors = rgb_depth[yy, xx]
                else:
                    no_color_count += 1
            else:
                points = bag.convert_camera2_depth_to_point_cloud(
                    depth_data=depth_data,
                    min_ray_z=args.min_ray_z,
                    max_range=args.max_range,
                    pixel_stride=args.pixel_stride,
                    depth_info_topic=args.depth_info_topic,
                )
        else:
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

        ts = get_topic_timestamp(d, idx)
        output_ply = osp.join(output_dir, f"{ts}.ply")
        write_ply_ascii(points, output_ply, colors=colors)
        saved_count += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(topic_data):
            print(f"  Processed {idx + 1}/{len(topic_data)} frames, saved={saved_count}, skipped={skip_count}")

    print(f"Done. Saved {saved_count} ply files, skipped {skip_count} frames.")
    if args.export_rgb_ply and no_color_count > 0:
        print(f"Warning: {no_color_count} frames have no matched RGB image and were saved as xyz ply.")


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
        "--depth-info-topic",
        type=str,
        default=McapLoader.CAMERA2_DEPTH_INFO_TOPIC,
        help="DepthInfo topic name for camera2 depth",
    )
    parser.add_argument(
        "--min-ray-z",
        type=float,
        default=0.1,
        help="Minimum Double Sphere ray z for camera2 depth point cloud",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=5.0,
        help="Maximum camera2 point range in meter, <=0 means disable upper bound",
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=1,
        help="Pixel stride when generating point cloud",
    )
    parser.add_argument(
        "--export-rgb-ply",
        action="store_true",
        help="Export RGB ply for camera2 depth by sampling colors from original RGB images",
    )
    parser.add_argument(
        "--rgb-topic",
        type=str,
        default="/robot0/sensor/camera2/compressed",
        help="Original RGB image topic used by --export-rgb-ply",
    )
    parser.add_argument(
        "--rgb-match-threshold-ms",
        type=float,
        default=50.0,
        help="Maximum timestamp difference for matching RGB image to depth frame",
    )
    parser.add_argument(
        "--rgb-crop-ratio",
        type=float,
        default=1.0,
        help="Optional center crop ratio for RGB image before resizing to depth resolution",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to export, <=0 means export all frames",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if (not osp.exists(args.mcap_file)) or (not osp.isfile(args.mcap_file)):
        raise FileNotFoundError(f"mcap file not found: {args.mcap_file}")
    export_depth_point_cloud_ply(args.mcap_file, args)
