import av
import numpy as np
from typing import Optional, Callable
from fractions import Fraction
import io
import cv2
import pdb

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

def img_parser(data: list) -> list:
    if not isinstance(data, list):
        assert ValueError("you should pass a list of item when decompressing image")
    all_decode_data = []
    video_decoder = VideoDecoder()
    for msg in data:
        decoded_data = video_decoder.decode_frame_container_v2(msg.data)
        if decoded_data is None:
            all_decode_data.append(None)
        else:
            if isinstance(decoded_data, list):
                for _d in decoded_data:
                    all_decode_data.append(_d)
            else:
                all_decode_data.append(decoded_data)
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