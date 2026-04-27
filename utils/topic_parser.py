import av
import numpy as np
from typing import Optional, Callable
from fractions import Fraction
import io
import re
import pdb
import huecodec


# pyav doc
# https://pyav.basswood-io.com/docs/stable/api/video.html
class VideoDecoder:
    def __init__(self):
        # 设置 PyAV 日志级别为详细模式
        av.logging.set_level(av.logging.ERROR)
        self.decoder_codec = av.CodecContext.create('h264', 'r')

    def decode_frame_container_v2(self, compressed_data: bytes) -> Optional[np.ndarray]:
        if not compressed_data or len(compressed_data) == 0:
            return None
        packet = av.packet.Packet(compressed_data)
        for j, decoded_frame in enumerate(self.decoder_codec.decode(packet)):
            img = decoded_frame.to_ndarray(channel_last=True, format="bgr24")
            return img
        return None


_HUE_Z_RE = re.compile(r"z_min_m=([^;]+);.*?z_max_m=([^;]+)", re.DOTALL)


def parse_hue_z_range(fmt: str) -> tuple:
    if not fmt:
        return (0.1, 1.5)
    m = _HUE_Z_RE.search(fmt.replace(" ", ""))
    if not m:
        return (0.1, 1.5)
    try:
        return (float(m.group(1)), float(m.group(2)))
    except ValueError:
        return (0.1, 1.5)

def img_parser(data: list) -> list:
    if not isinstance(data, list):
        assert ValueError("you should pass a list of item when decompressing image")
    all_decode_data = []
    video_decoder = VideoDecoder()
    has_find_first_kf = False
    for msg in data:
        nal_bytes = msg.data
        start_offset = 4 if nal_bytes.startswith(b"\x00\x00\x00\x01") else 3 if nal_bytes.startswith(b"\x00\x00\x01") else 0
        nal_unit_type = nal_bytes[start_offset] & 0x1F
        if nal_unit_type != 7 and (not has_find_first_kf):  # 关键帧
            decoded_data = None
        else:
            has_find_first_kf = True
            try:
                decoded_data = video_decoder.decode_frame_container_v2(msg.data)
            except Exception as e:
                decoded_data = None

        if decoded_data is None:
            all_decode_data.append(None)
        else:
            if isinstance(decoded_data, list):
                for _d in decoded_data:
                    all_decode_data.append(_d)
            else:
                all_decode_data.append(decoded_data)
    
    return all_decode_data


def depth_img_parser(data: list) -> list:
    """
    Decode H264 depth stream frame-by-frame.
    Output per-frame depth map with shape [H, W, 1], unit meter.
    """
    if huecodec is None:
        raise ImportError(
            "huecodec is required for depth H264 decoding. "
            "Please install hue-depth-encoding package first."
        )
    if not isinstance(data, list):
        assert ValueError("you should pass a list of item when decompressing image")
    all_decode_data = []
    video_decoder = VideoDecoder()
    has_find_first_kf = False
    for msg in data:
        nal_bytes = msg.data
        start_offset = 4 if nal_bytes.startswith(b"\x00\x00\x00\x01") else 3 if nal_bytes.startswith(b"\x00\x00\x01") else 0
        nal_unit_type = nal_bytes[start_offset] & 0x1F
        if nal_unit_type != 7 and (not has_find_first_kf):  # 关键帧
            decoded_data = None
        else:
            has_find_first_kf = True
            try:
                decoded_data = video_decoder.decode_frame_container_v2(msg.data)
            except Exception:
                decoded_data = None

        if decoded_data is None:
            all_decode_data.append(None)
            continue

        try:
            z_min, z_max = parse_hue_z_range(getattr(msg, "format", ""))
            # decoded_data is BGR, convert to RGB without opencv dependency
            rgb_data = decoded_data[:, :, ::-1].copy()
            depth_data = huecodec.rgb2depth(rgb_data, (z_min, z_max), inv_depth=False)
            depth_data = np.asarray(depth_data, dtype=np.float32)
            if depth_data.ndim == 2:
                depth_data = depth_data[..., None]
            all_decode_data.append(depth_data)
        except Exception:
            all_decode_data.append(None)
    return all_decode_data

def imu_parser(data: list) -> list:
    all_decoder_data = []
    for msg in data:
        all_decoder_data.append(np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ]))
    return all_decoder_data

def tactile_parser(data: list) -> list:
    all_decoder_data = []
    for msg in data:
        all_decoder_data.append(np.array(
            msg.pressure_data
        ))
    return all_decoder_data

def pose_parser(data: list) -> list:
    all_decoder_data = []
    for msg in data:
        all_decoder_data.append(np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]))
    return all_decoder_data

def magnetic_encoder_parser(data: list) -> list:
    all_decoder_data = []
    for msg in data:
        all_decoder_data.append(np.array(
            [msg.value]
        ))
    return all_decoder_data