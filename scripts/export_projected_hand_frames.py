#!/usr/bin/env python3
"""
将 MCAP 中的左右手 handtracking（VIO 坐标系）投影到相机原始畸变图上，并导出为图片。

脚本逻辑：
1. 读取 `/robot0/handtracking/left` 和 `/robot0/handtracking/right`
2. 读取 `/robot0/vio/eef_pose`
3. 读取各相机 `/robot0/sensor/cameraN/camera_info` 中的内参与 `T_b_c`
4. 将 handtracking 从 VIO 坐标系变换到目标相机坐标系
5. 做针孔投影，并在 double sphere (`ds`) 模型下映射回原始畸变图像素
6. 将骨架画到 `/robot0/sensor/cameraN/compressed` 解码后的图像上，按相机分文件夹导出 JPG

依赖的 MCAP topic：
- `/robot0/handtracking/left`
- `/robot0/handtracking/right`
- `/robot0/vio/eef_pose`
- `/robot0/sensor/cameraN/compressed`
- `/robot0/sensor/cameraN/camera_info`

默认配置：导出六个相机（0..5），每 100 帧导出 1 帧。

用法示例：

1. 导出六个相机，每 10 帧保存 1 帧：
   python scripts/export_projected_hand_frames.py \
     --input-mcap test_data/example_labeled_world.mcap \
     --output-dir test_data/export_projected_hand_frames_stride10 \
     --camera-ids 0 1 2 3 4 5 \
     --frame-stride 10

2. 只导出 camera0：
   python scripts/export_projected_hand_frames.py \
     --input-mcap test_data/example_labeled_world.mcap \
     --output-dir test_data/export_projected_hand_frames_camera0 \
     --camera-ids 0

3. 导出六个相机，每 100 帧保存 1 帧：
   python scripts/export_projected_hand_frames.py \
     --input-mcap test_data/example_labeled_world.mcap \
     --output-dir test_data/export_projected_hand_frames_stride100 \
     --camera-ids 0 1 2 3 4 5 \
     --frame-stride 100

输出目录结构示例：
- output_dir/
  - camera0/
  - camera1/
  - camera2/
  - camera3/
  - camera4/
  - camera5/
  - summary.json
"""
from __future__ import annotations

import argparse
import os
import bisect
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from mcap.reader import NonSeekingReader

REPO_ROOT = Path(__file__).resolve().parent.parent
PB2_DIR = REPO_ROOT / "pb2"
if str(PB2_DIR) not in sys.path:
    sys.path.insert(0, str(PB2_DIR))

import CameraCalibration_pb2  # noqa: E402
import CompressedImage_pb2  # noqa: E402
import HandTrackInfo_pb2  # noqa: E402
import PoseInFrame_pb2  # noqa: E402


SCHEMA_COMPRESSED_IMAGE = "foxglove.CompressedImage"
SCHEMA_CAMERA_CALIBRATION = "foxglove.CameraCalibration"
SCHEMA_HAND = "foxglove.HandPointFrameInfo"

HAND_TOPIC_LEFT = "/robot0/handtracking/left"
HAND_TOPIC_RIGHT = "/robot0/handtracking/right"
DEFAULT_CAMERA_IDS = (0, 1, 2, 3, 4, 5)

def _pinhole_unproject(q: np.ndarray) -> np.ndarray:
    x, y = q[..., 0], q[..., 1]
    z = np.ones_like(x)
    rays = np.stack([x, y, z], axis=-1)
    norm = np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays / np.maximum(norm, 1e-12)


def _double_sphere_project(v: np.ndarray, xi: float, alpha: float, eps: float = 1e-8) -> np.ndarray:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    z1 = alpha * d1 + (1 - alpha) * z
    d2 = np.sqrt(x * x + y * y + z1 * z1)
    denom = np.maximum(xi * d2 + z1, eps)
    return np.stack([x / denom, y / denom], axis=-1)


_HAND_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _decode_pb2(cls: type[Any], payload: bytes) -> Any:
    msg = cls()
    msg.ParseFromString(payload)
    return msg


def _iter_mcap_records(
    mcap_path: Path,
    *,
    schema_whitelist: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    with open(mcap_path, "rb") as f:
        reader = NonSeekingReader(f)
        for schema, channel, message in reader.iter_messages():
            schema_name = getattr(schema, "name", "") or ""
            if schema_whitelist is not None and schema_name not in schema_whitelist:
                continue
            publish_time = int(getattr(message, "publish_time", message.log_time))
            yield {
                "schema_name": schema_name,
                "topic": getattr(channel, "topic", "") or "",
                "log_time": int(message.log_time),
                "publish_time": publish_time,
                "data": message.data,
            }


def default_topics_for_camera_id(camera_id: int) -> tuple[str, str]:
    base = f"/robot0/sensor/camera{camera_id}"
    return (f"{base}/compressed", f"{base}/camera_info")


def _k_to_fx_fy_cx_cy(K: list[float]) -> tuple[float, float, float, float]:
    if len(K) < 9:
        raise ValueError(f"CameraCalibration.K 长度不足: {len(K)}")
    return float(K[0]), float(K[4]), float(K[2]), float(K[5])


def _quat_pos_to_4x4(px: float, py: float, pz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        raise ValueError("Quaternion has zero norm")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw), px],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw), py],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy), pz],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _invert_rigid_transform(transform_4x4: np.ndarray) -> np.ndarray:
    rotation = transform_4x4[:3, :3]
    translation = transform_4x4[:3, 3]
    rotation_t = rotation.T
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rotation_t
    inv[:3, 3] = -(rotation_t @ translation)
    return inv


def _parse_calibration_intrinsics(msg: Any) -> tuple[float, float, float, float, float, float, int, int, str, np.ndarray]:
    distortion = list(msg.D)
    K = list(msg.K)
    if len(distortion) >= 6:
        fx, fy, cx, cy, xi, alpha = (float(v) for v in distortion[:6])
    elif len(K) >= 9:
        fx, fy, cx, cy = _k_to_fx_fy_cx_cy(K)
        xi, alpha = 0.0, 0.0
    else:
        raise ValueError(f"CameraCalibration D ({len(distortion)}) 与 K ({len(K)}) 不足以解析内参")
    t_b_c = list(msg.T_b_c)
    if len(t_b_c) >= 7:
        t_body_camera = _quat_pos_to_4x4(*[float(v) for v in t_b_c[:7]])
    else:
        t_body_camera = np.eye(4, dtype=np.float64)
    return fx, fy, cx, cy, xi, alpha, int(msg.width), int(msg.height), (msg.distortion_model or "").strip().lower(), t_body_camera


def _load_latest_calibration(
    mcap_path: Path,
    calib_topic: str,
) -> tuple[float, float, float, float, float, float, int, int, str, np.ndarray]:
    last: Any = None
    for rec in _iter_mcap_records(mcap_path, schema_whitelist={SCHEMA_CAMERA_CALIBRATION}):
        if rec["topic"] != calib_topic:
            continue
        last = _decode_pb2(CameraCalibration_pb2.CameraCalibration, rec["data"])
    if last is None:
        raise FileNotFoundError(f"MCAP 中未找到标定 topic: {calib_topic}")
    return _parse_calibration_intrinsics(last)


def _collect_hands_by_log_time(mcap_path: Path) -> dict[int, dict[str, Any]]:
    by_t: dict[int, dict[str, Any]] = {}
    for rec in _iter_mcap_records(mcap_path, schema_whitelist={SCHEMA_HAND}):
        topic = rec["topic"]
        if topic not in (HAND_TOPIC_LEFT, HAND_TOPIC_RIGHT):
            continue
        frame = _decode_pb2(HandTrackInfo_pb2.HandPointFrameInfo, rec["data"])
        if not frame.is_valid:
            continue
        slot = by_t.setdefault(int(rec["log_time"]), {"left": None, "right": None})
        side = "left" if topic == HAND_TOPIC_LEFT else "right"
        slot[side] = frame
    return by_t


def _collect_eef_pose_by_log_time(mcap_path: Path) -> dict[int, dict[str, list[float]]]:
    by_t: dict[int, dict[str, list[float]]] = {}
    for rec in _iter_mcap_records(mcap_path):
        if rec["topic"] != "/robot0/vio/eef_pose" or rec["schema_name"] != "foxglove.PoseInFrame":
            continue
        msg = _decode_pb2(PoseInFrame_pb2.PoseInFrame, rec["data"])
        by_t[int(rec["log_time"])] = {
            "position": [float(msg.pose.position.x), float(msg.pose.position.y), float(msg.pose.position.z)],
            "orientation": [float(msg.pose.orientation.x), float(msg.pose.orientation.y), float(msg.pose.orientation.z), float(msg.pose.orientation.w)],
        }
    return by_t


R_BODY = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def _normalize_quaternion(raw_value: list[float]) -> list[float]:
    if len(raw_value) != 4:
        raise ValueError(f"pose quaternion must have 4 values, got {len(raw_value)}")
    x, y, z, w = [float(item) for item in raw_value]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / norm, y / norm, z / norm, w / norm]


def _quat_to_rotation_matrix_xyzw(quat_xyzw: list[float]) -> np.ndarray:
    x, y, z, w = quat_xyzw
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _build_vio_from_camera_transform(position_xyz: list[float], orientation_xyzw: list[float], t_body_camera: np.ndarray) -> np.ndarray:
    quat_xyzw = _normalize_quaternion(orientation_xyzw)
    r_vio_body = _quat_to_rotation_matrix_xyzw(quat_xyzw)
    t_vio_body = np.eye(4, dtype=np.float64)
    t_vio_body[:3, :3] = r_vio_body
    t_vio_body[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    r_body_4x4 = np.eye(4, dtype=np.float64)
    r_body_4x4[:3, :3] = R_BODY
    return t_vio_body @ r_body_4x4 @ t_body_camera


def _nearest_log_time(sorted_ts: list[int], t: int, max_delta_ns: int) -> int | None:
    if not sorted_ts:
        return None
    idx = bisect.bisect_left(sorted_ts, t)
    best: int | None = None
    best_d = max_delta_ns + 1
    for j in (idx, idx - 1):
        if 0 <= j < len(sorted_ts):
            d = abs(sorted_ts[j] - t)
            if d < best_d:
                best = sorted_ts[j]
                best_d = d
    return best if best_d <= max_delta_ns else None


def _hand_positions_xyz(frame: Any) -> list[tuple[float, float, float]]:
    xyz: list[tuple[float, float, float]] = []
    for bone in frame.bone_data:
        g = bone.to_global
        xyz.append((float(g.position.x), float(g.position.y), float(g.position.z)))
    return xyz


def project_points_pinhole(
    xyz: list[tuple[float, float, float]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    pts = np.zeros((len(xyz), 2), dtype=np.float32)
    eps = 1e-6
    for i, (x, y, z) in enumerate(xyz):
        if z <= eps:
            pts[i] = np.nan
            continue
        pts[i, 0] = fx * (x / z) + cx
        pts[i, 1] = fy * (y / z) + cy
    return pts


def _is_double_sphere_model(model: str) -> bool:
    if not model:
        return True
    if model == "ds":
        return True
    return "double" in model and "sphere" in model


def pinhole_pixels_to_distorted_ds(
    uv: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    xi: float,
    alpha: float,
) -> np.ndarray:
    out = np.array(uv, dtype=np.float32, copy=True)
    valid = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1])
    if not np.any(valid):
        return out
    q = np.stack([(out[valid, 0] - cx) / fx, (out[valid, 1] - cy) / fy], axis=-1).astype(np.float64)
    rays = _pinhole_unproject(q)
    q_src = _double_sphere_project(rays, xi, alpha)
    out[valid, 0] = (q_src[:, 0] * fx + cx).astype(np.float32)
    out[valid, 1] = (q_src[:, 1] * fy + cy).astype(np.float32)
    out[~valid] = np.nan
    return out


def _draw_hand_overlay(bgr: np.ndarray, uv: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    h, w = bgr.shape[:2]
    for a, b in _HAND_EDGES:
        pa = uv[a]
        pb = uv[b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        ia = (int(round(pa[0])), int(round(pa[1])))
        ib = (int(round(pb[0])), int(round(pb[1])))
        if not (0 <= ia[0] < w and 0 <= ia[1] < h and 0 <= ib[0] < w and 0 <= ib[1] < h):
            continue
        cv2.line(bgr, ia, ib, color_bgr, 2, lineType=cv2.LINE_AA)
    for p in uv:
        if np.any(np.isnan(p)):
            continue
        c = (int(round(p[0])), int(round(p[1])))
        if 0 <= c[0] < w and 0 <= c[1] < h:
            cv2.circle(bgr, c, 3, color_bgr, -1, lineType=cv2.LINE_AA)


class HandOverlayContext:
    def __init__(self, mcap_path: Path, calib_topic: str, *, max_delta_ns: int = 2_000_000) -> None:
        self._max_delta_ns = int(max_delta_ns)
        self._fx, self._fy, self._cx, self._cy, self._xi, self._alpha, self._calib_w, self._calib_h, self._dist_model, self._t_body_camera = _load_latest_calibration(
            mcap_path, calib_topic
        )
        self._ds_forward = _is_double_sphere_model(self._dist_model)
        self._hands_by_t = _collect_hands_by_log_time(mcap_path)
        self._sorted_hand_ts = sorted(self._hands_by_t.keys())
        self._poses_by_t = _collect_eef_pose_by_log_time(mcap_path)
        self._sorted_pose_ts = sorted(self._poses_by_t.keys())

    def overlay_bgr(self, bgr: np.ndarray, image_log_time_ns: int) -> np.ndarray:
        nh, nw = bgr.shape[:2]
        scale_x = nw / float(self._calib_w) if self._calib_w > 0 else 1.0
        scale_y = nh / float(self._calib_h) if self._calib_h > 0 else 1.0
        hit_t = _nearest_log_time(self._sorted_hand_ts, int(image_log_time_ns), self._max_delta_ns)
        pose_t = _nearest_log_time(self._sorted_pose_ts, int(image_log_time_ns), self._max_delta_ns)
        overlay = bgr.copy()
        if hit_t is None or pose_t is None:
            return overlay
        pair = self._hands_by_t[hit_t]
        pose = self._poses_by_t[pose_t]
        t_vio_camera = _build_vio_from_camera_transform(pose["position"], pose["orientation"], self._t_body_camera)
        t_camera_vio = _invert_rigid_transform(t_vio_camera)
        for side, color in (("left", (0, 255, 255)), ("right", (255, 128, 0))):
            frame = pair.get(side)
            if frame is None:
                continue
            xyz = _hand_positions_xyz(frame)
            if len(xyz) < 21:
                continue
            camera_xyz: list[tuple[float, float, float]] = []
            for x, y, z in xyz[:21]:
                p_vio = np.array([x, y, z, 1.0], dtype=np.float64)
                p_cam = t_camera_vio @ p_vio
                camera_xyz.append((float(p_cam[0]), float(p_cam[1]), float(p_cam[2])))
            uv_calib = project_points_pinhole(camera_xyz, self._fx, self._fy, self._cx, self._cy)
            if self._ds_forward:
                uv_calib = pinhole_pixels_to_distorted_ds(
                    uv_calib, self._fx, self._fy, self._cx, self._cy, self._xi, self._alpha
                )
            uv = uv_calib.copy()
            uv[:, 0] *= scale_x
            uv[:, 1] *= scale_y
            _draw_hand_overlay(overlay, uv, color)
        return overlay


def _decode_jpeg(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _normalize_h264_bytestream(data: bytes) -> bytes:
    if not data:
        return data
    if data.startswith(b"\x00\x00\x01") or data.startswith(b"\x00\x00\x00\x01"):
        return data
    out = bytearray()
    i = 0
    n = len(data)
    while i + 4 <= n:
        nal_len = int.from_bytes(data[i : i + 4], "big")
        i += 4
        if nal_len <= 0 or i + nal_len > n:
            out.clear()
            break
        out.extend(b"\x00\x00\x00\x01")
        out.extend(data[i : i + nal_len])
        i += nal_len
    if out and i == n:
        return bytes(out)
    hdr = data[0]
    nal_type = hdr & 0x1F
    if nal_type in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12):
        return b"\x00\x00\x00\x01" + data
    return data


def _decode_h264_packet(codec: Any, data: bytes) -> list[np.ndarray]:
    import av

    data = _normalize_h264_bytestream(data)
    if not data:
        return []
    packet = av.packet.Packet(data)
    try:
        return [frame.to_ndarray(format="bgr24") for frame in codec.decode(packet)]
    except av.error.InvalidDataError:
        return []


def _flush_h264_decoder(codec: Any) -> list[np.ndarray]:
    import av

    try:
        return [frame.to_ndarray(format="bgr24") for frame in codec.decode(None)]
    except (EOFError, av.error.InvalidDataError, av.error.EOFError):
        return []


def _detect_format(data: bytes, format_hint: str = "") -> str:
    hint = format_hint.lower()
    if "jpeg" in hint or "jpg" in hint:
        return "jpeg"
    if "h264" in hint or "avc" in hint:
        return "h264"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if len(data) >= 4 and (data[:4] == b"\x00\x00\x00\x01" or data[:3] == b"\x00\x00\x01"):
        return "h264"
    return "h264"


class _CompressedPacketDecoder:
    def __init__(self) -> None:
        self._codec: Any = None
        self._last_h264_log_time = 0

    def decode(self, raw: bytes, fmt_hint: str, log_time: int) -> list[tuple[int, np.ndarray]]:
        detected = _detect_format(raw, fmt_hint)
        if detected == "jpeg":
            image = _decode_jpeg(raw)
            return [] if image is None else [(log_time, image)]
        self._last_h264_log_time = log_time
        if self._codec is None:
            try:
                import av
            except ImportError as exc:
                raise RuntimeError("源图像为 H.264 时需安装 PyAV: pip install av") from exc
            self._codec = av.CodecContext.create("h264", "r")
        return [(log_time, img) for img in _decode_h264_packet(self._codec, raw)]

    def flush(self) -> list[tuple[int, np.ndarray]]:
        if self._codec is None:
            return []
        return [(self._last_h264_log_time, img) for img in _flush_h264_decoder(self._codec)]


def export_projected_hand_frames(
    input_mcap: Path,
    output_dir: Path,
    *,
    camera_ids: tuple[int, ...] = DEFAULT_CAMERA_IDS,
    max_delta_ns: int = 2_000_000,
    jpeg_quality: int = 95,
    frame_stride: int = 1,
) -> dict[str, Any]:
    src = input_mcap.expanduser().resolve()
    out_root = output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    camera_states: dict[int, dict[str, Any]] = {}
    for camera_id in camera_ids:
        image_topic, calib_topic = default_topics_for_camera_id(camera_id)
        overlay_ctx = HandOverlayContext(src, calib_topic, max_delta_ns=max_delta_ns)
        camera_dir = out_root / f"camera{camera_id}"
        camera_dir.mkdir(parents=True, exist_ok=True)
        camera_states[camera_id] = {
            "camera_id": camera_id,
            "image_topic": image_topic,
            "calib_topic": calib_topic,
            "overlay_ctx": overlay_ctx,
            "decoder": _CompressedPacketDecoder(),
            "camera_dir": camera_dir,
            "frame_index": 0,
            "summary": {
                "camera_id": camera_id,
                "image_topic": image_topic,
                "calib_topic": calib_topic,
                "output_dir": str(camera_dir),
                "written_images": 0,
                "skipped_decode_empty": 0,
                "skipped_imwrite_fail": 0,
                "skipped_by_stride": 0,
                "flushed_images": 0,
            },
        }

    frame_stride = max(1, int(frame_stride))

    summary: dict[str, Any] = {
        "input_mcap": str(src),
        "output_dir": str(out_root),
        "camera_ids": list(camera_ids),
        "max_delta_ns": int(max_delta_ns),
        "jpeg_quality": int(jpeg_quality),
        "frame_stride": frame_stride,
        "cameras": {str(cid): state["summary"] for cid, state in camera_states.items()},
    }

    def _write_frame(state: dict[str, Any], bgr: np.ndarray, log_time: int, publish_time: int, source_format: str) -> None:
        idx = int(state["frame_index"])
        filename = f"{idx:06d}_{int(log_time)}_{int(publish_time)}.jpg"
        path = state["camera_dir"] / filename
        ok = cv2.imwrite(
            str(path),
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            state["summary"]["skipped_imwrite_fail"] += 1
            return
        state["frame_index"] += 1
        state["summary"]["written_images"] += 1
        state["summary"]["last_source_format"] = source_format

    for rec in _iter_mcap_records(src, schema_whitelist={SCHEMA_COMPRESSED_IMAGE}):
        topic = rec["topic"]
        matching_state: dict[str, Any] | None = None
        for state in camera_states.values():
            if topic == state["image_topic"]:
                matching_state = state
                break
        if matching_state is None:
            continue

        msg = _decode_pb2(CompressedImage_pb2.CompressedImage, rec["data"])
        raw = bytes(msg.data)
        if not raw:
            matching_state["summary"]["skipped_decode_empty"] += 1
            continue
        fmt_hint = msg.format or ""
        detected = _detect_format(raw, fmt_hint)
        frames = matching_state["decoder"].decode(raw, fmt_hint, int(rec["log_time"]))
        if not frames:
            matching_state["summary"]["skipped_decode_empty"] += 1
            continue

        for image_log_time, bgr in frames:
            frame_index_before_write = int(matching_state["frame_index"]) + int(matching_state["summary"]["skipped_by_stride"])
            if frame_index_before_write % frame_stride != 0:
                matching_state["summary"]["skipped_by_stride"] += 1
                continue
            overlay_bgr = matching_state["overlay_ctx"].overlay_bgr(bgr, image_log_time)
            _write_frame(
                matching_state,
                overlay_bgr,
                image_log_time,
                int(rec["publish_time"]),
                detected,
            )

    for state in camera_states.values():
        flush_frames = state["decoder"].flush()
        for image_log_time, bgr in flush_frames:
            overlay_bgr = state["overlay_ctx"].overlay_bgr(bgr, image_log_time)
            before = state["summary"]["written_images"]
            _write_frame(state, overlay_bgr, image_log_time, image_log_time, "h264_flush")
            if state["summary"]["written_images"] > before:
                state["summary"]["flushed_images"] += 1

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="将 labeled_world MCAP 中左右手投影到 6 路相机原始畸变图，并导出为图片。")
    parser.add_argument("--input-mcap", type=Path, required=True, help="输入 labeled_world.mcap")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录，内部将创建 camera0..camera5 六个文件夹")
    parser.add_argument("--camera-ids", type=int, nargs="*", default=list(DEFAULT_CAMERA_IDS), help="要导出的相机 ID，默认 0 1 2 3 4 5")
    parser.add_argument("--max-delta-ns", type=int, default=2_000_000, help="图像与手消息最近邻对齐阈值")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="输出 JPEG 质量")
    parser.add_argument("--frame-stride", type=int, default=100, help="每隔多少帧导出一帧；10 表示每 10 帧导 1 帧")
    args = parser.parse_args()

    summary = export_projected_hand_frames(
        args.input_mcap,
        args.output_dir,
        camera_ids=tuple(args.camera_ids),
        max_delta_ns=args.max_delta_ns,
        jpeg_quality=args.jpeg_quality,
        frame_stride=args.frame_stride,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
