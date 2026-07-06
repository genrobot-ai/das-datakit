import numpy as np
import os.path as osp
import json
from mcap.reader import make_reader, McapReader, NonSeekingReader
from mcap.writer import Writer
from mcap_protobuf.decoder import DecoderFactory
from mcap_protobuf.writer import Writer
try:
    import cv2
except Exception:
    cv2 = None
from .topic_parser import (
    img_parser,
    depth_img_parser,
    predz_depth_img_parser,
    imu_parser,
    tactile_parser,
    pose_parser,
    magnetic_encoder_parser
)
from collections import defaultdict
from .sync_graph import RelationGraph
import io
import sys
sys.path.append(osp.join(osp.dirname(__file__), "../pb2"))
from pb2.IMUMeasurement_pb2 import IMUMeasurement
from pb2.CameraCalibration_pb2 import CameraCalibration
from pb2.SystemInfo_pb2 import SystemInfo
from pb2.TactileMeasurement_pb2 import TactileMeasurement
from pb2.MagneticEncoder_pb2 import MagneticEncoderMeasurement
from pb2.CompressedImage_pb2 import CompressedImage
from pb2.DepthInfo_pb2 import DepthInfo
from pb2.PoseInFrame_pb2 import PoseInFrame
from pb2.RobotInfo_pb2 import RobotInfo
import pdb

PROTO_MAPPING = {
    "/robot0/sensor/camera0/compressed": CompressedImage,
    "/robot0/sensor/camera1/compressed": CompressedImage,
    "/robot0/sensor/camera2/compressed": CompressedImage,
    "/robot0/sensor/camera2/depth": CompressedImage,
    "/robot0/sensor/camera2/depth_info": DepthInfo,
    "/robot0/sensor/imu": IMUMeasurement,
    "/robot0/sensor/camera0/camera_info": CameraCalibration,
    "/robot0/sensor/camera1/camera_info": CameraCalibration,
    "/robot0/sensor/camera2/camera_info": CameraCalibration,
    "/robot0/system_info": SystemInfo,
    "/robot0/sensor/tactile_left": TactileMeasurement,
    "/robot0/sensor/tactile_right": TactileMeasurement,
    "/robot0/sensor/magnetic_encoder": MagneticEncoderMeasurement,
    "/robot0/vio/eef_pose": PoseInFrame,
    "/robot0/sim/robot_info": RobotInfo,

    "/robot1/sensor/camera0/compressed": CompressedImage,
    "/robot1/sensor/camera1/compressed": CompressedImage,
    "/robot1/sensor/camera2/compressed": CompressedImage,
    "/robot1/sensor/imu": IMUMeasurement,
    "/robot1/sensor/camera0/camera_info": CameraCalibration,
    "/robot1/sensor/camera1/camera_info": CameraCalibration,
    "/robot1/sensor/camera2/camera_info": CameraCalibration,
    "/robot1/system_info": SystemInfo,
    "/robot1/sensor/tactile_left": TactileMeasurement,
    "/robot1/sensor/tactile_right": TactileMeasurement,
    "/robot1/sensor/magnetic_encoder": MagneticEncoderMeasurement,
    "/robot1/vio/eef_pose": PoseInFrame,
    "/robot1/sim/robot_info": RobotInfo,

    "/robot0/sensor/camera3/compressed": CompressedImage,
    "/robot0/sensor/camera4/compressed": CompressedImage,
    "/robot0/sensor/camera5/compressed": CompressedImage,
    "/robot0/sensor/depth/compressed": CompressedImage,
    "/robot0/sensor/depth/rectify_compressed": CompressedImage,
    "/robot0/sensor/camera3/camera_info": CameraCalibration,
    "/robot0/sensor/camera4/camera_info": CameraCalibration,
    "/robot0/sensor/camera5/camera_info": CameraCalibration,
}

def ns_to_s(ns):
    return float(ns) / 1e9

def parse_topic_data(reader: McapReader, topic: str):
    if topic not in PROTO_MAPPING:
        print(f"topic {topic} is not in PROTO_MAPPING.")
        return []
    proto_msg_class = PROTO_MAPPING[topic]
    topic_msgs = []
    for schema, channel, message in reader.iter_messages(topics=[topic]):
        proto_msg = proto_msg_class()
        proto_msg.ParseFromString(message.data)
        topic_msgs.append({
            "data": proto_msg,
            "log_time": message.log_time,
            "publish_time": message.publish_time,
        })
    return topic_msgs

class McapLoader:
    DEFAULT_STEREO_CALIBRATION_TOPIC = "/robot0/sensor/depth/stereo_calibration"
    CAMERA2_DEPTH_TOPIC = "/robot0/sensor/camera2/depth"
    CAMERA2_DEPTH_INFO_TOPIC = "/robot0/sensor/camera2/depth_info"

    TOPIC_AUTO_DECOMPRESS_MAP = {
        "/robot0/sensor/depth/compressed": depth_img_parser,
        "/robot0/sensor/camera2/depth": predz_depth_img_parser,
    }

    AUTO_DECOMPRESS_MAP = {
        "CompressedImage": img_parser,
        "CompressedVideo": img_parser,
        "IMUMeasurement": imu_parser,
        "MagneticEncoderMeasurement": magnetic_encoder_parser,
        "TactileMeasurement": tactile_parser,
        "PoseInFrame": pose_parser
    }
    
    def __init__(
        self,
        bag_path:str,
    ):
        self._bag_path = bag_path
        self.init_reader()
        self._bag_data = {}
        self.get_statistic_info()

        self.topic_sync_info = defaultdict(dict)
        self.seq2idx = defaultdict(dict)
        self.sync_graph = RelationGraph()
        self._stereo_calibration_cache = {}
        self._camera2_depth_info_cache = None

    def init_reader(self):
        self._stream = io.BytesIO(open(self._bag_path, "rb").read())
        self._mcap_reader = NonSeekingReader(self._stream, decoder_factories=[DecoderFactory()])
    
    def _reset_stream(self):
        self._stream.seek(0)
        self._mcap_reader = NonSeekingReader(self._stream, decoder_factories=[DecoderFactory()])
    
    def get_statistic_info(self):
        self._reset_stream()
        topic_summary = self._mcap_reader.get_summary()
        self._reset_stream()
        header = self._mcap_reader.get_header()
        self._topic_header = header
        # parse iter info
        self._reset_stream()
        meta_data = self._mcap_reader.iter_metadata()
        self._topic_meta = [meta for meta in meta_data]
        self._reset_stream()
        attachments = self._mcap_reader.iter_attachments()
        self._topic_attachments = [att for att in attachments]

        # read topic->id mapping
        topic_channels = topic_summary.channels
        topic2id = {}
        topic2schema_id = {}
        for v in topic_channels.values():
            if v.message_encoding != "protobuf":
                print(f"Unsupported message encoding: {v.message_encoding}")
            topic2id[v.topic] = v.id
            topic2schema_id[v.topic] = v.schema_id

        all_topic_names = list(topic2id.keys())

        topic_statistics = topic_summary.statistics
        topic_msg_count = topic_statistics.channel_message_counts
        msg_start_time = topic_statistics.message_start_time
        msg_end_time = topic_statistics.message_end_time

        # frequency
        bag_time_length = (msg_end_time - msg_start_time) / 1e9  # seconds
        topic_frequency_info = {}
        for topic_name in all_topic_names:
            msg_count = topic_msg_count.get(topic2id[topic_name], 0)
            if msg_count == 0:
                topic_frequency_info[topic_name] = 0
            else:
                topic_frequency_info[topic_name] = round(topic_msg_count[topic2id[topic_name]] / bag_time_length, 1)

        self.topic_schemas = {
            tn: topic_summary.schemas[topic2schema_id[tn]].name
            for tn in all_topic_names
        }
        self.all_topic_names = all_topic_names  
        self.topic_statistics = topic_statistics
        self.msg_start_time = msg_start_time
        self.msg_end_time = msg_end_time
        self.topic_frequency_info = topic_frequency_info

    def _update_seq2idx(self, topic_name: str, topic_data: list):
        for idx, sdata in enumerate(topic_data):
            if not hasattr(sdata["data"], "header"):
                continue
            header = sdata["data"].header
            sequence_num = header.sequence_num
            self.seq2idx[topic_name][sequence_num] = idx

    def _update_sync_info(self, topic_name: str, topic_data: list):
        for idx, sdata in enumerate(topic_data):
            if not hasattr(sdata["data"], "header"):
                continue
            header = sdata["data"].header
            sequence_num = header.sequence_num
            for input in header.inputs:
                self.sync_graph.add_relation(topic_name, sequence_num, {input.topic_name: input.sequence_num})

    def register_sync_relation_with_time(self, topic_name_1: str, topic_name_2: str, overwrite: bool = False) -> bool:
        if topic_name_1 == topic_name_2:
            print(f"two topics are same.")
            return True
        self.load_topics([topic_name_1, topic_name_2])
        
        topic_data_1 = self.get_topic_data(topic_name_1)
        if (topic_data_1 is None) or len(topic_data_1) == 0:
            print(f"topic {topic_name_1} is not in bag.")
            return False
        topic_data_2 = self.get_topic_data(topic_name_2)
        if (topic_data_2 is None) or len(topic_data_2) == 0:
            print(f"topic {topic_name_2} is not in bag.")
            return False
        
        seq_for_topic2 = []
        ts_for_topic2 = []
        for idx, sdata in enumerate(topic_data_2):
            # 有些topic没有header
            if not hasattr(sdata["data"], "header"):
                continue
            header = sdata["data"].header
            seq_for_topic2.append(header.sequence_num)
            ts_for_topic2.append(header.timestamp)
        
        if not ts_for_topic2:
            print(f"No valid timestamps found for {topic_name_2}")
            return False
        
        ts_for_topic2 = np.array(ts_for_topic2, dtype=np.int64)
        
        # start register sync relation
        valid_register_count = 0
        for idx, sdata in enumerate(topic_data_1):
            # 有些topic没有header
            if not hasattr(sdata["data"], "header"):
                continue
            header = sdata["data"].header
            sequence_num = header.sequence_num
            ts = header.timestamp

            # 找到对应的topic2的序号
            diff = np.abs(ts - ts_for_topic2)
            min_diff_idx = np.argmin(diff)

            # 注册同步关系
            self.sync_graph.add_relation(topic_name_1, sequence_num, {topic_name_2: seq_for_topic2[min_diff_idx]}, overwrite=overwrite)
            valid_register_count += 1
        
        self.sync_graph.deduce_relations()
        
        if valid_register_count == 0:
            print(f"no valid register relation between {topic_name_1} and {topic_name_2}")
        else:
            print(f"register [{valid_register_count}] sync relation between {topic_name_1}({len(topic_data_1)}), {topic_name_2}")
        return True
    
    def load_topics(self, topics: list = [], auto_decompress: bool = True, auto_sync: bool = False):
        """
        Load specified topics from a bag file.

        This method loads the specified topics from the bag file located at
        self._bag_path. If a topic is already loaded, it will not be loaded again.
        The method ensures that there are no duplicate topics being loaded.

        Args:
            topics (list): A list of topic names to be loaded. If a single topic
                   name is provided as a string, it will be converted to
                   a list containing that single topic.

        Returns:
            None
        """
        if not isinstance(topics, list):
            topics = [topics]
        # 防止重复解析
        not_loaded_topics = []
        for topic in topics:
            if topic not in self.all_topic_names:
                print(f" >> {topic} << is not protobug topic or not in bag, so it will not be loaded")
                continue
            if topic not in self._bag_data:
                not_loaded_topics.append(topic)
        # 去重
        not_loaded_topics = list(set(not_loaded_topics))
        if len(not_loaded_topics) == 0:
            return
        bag_data = {}
        for topic_name in not_loaded_topics:
            self._reset_stream()
            topic_msgs = parse_topic_data(self._mcap_reader, topic_name)
            # auto decompress
            if auto_decompress:
                try:
                    topic_msgs = self._auto_decompress(topic_msgs, topic_name=topic_name)
                except Exception as e:
                    print(f"auto decompress failed for topic {topic_name}: {e}")

            self._update_seq2idx(topic_name, topic_msgs)
            # update sync info
            if auto_sync:
                self._update_sync_info(topic_name, topic_msgs)
                self.sync_graph.deduce_relations()

            bag_data[topic_name] = topic_msgs

        if auto_sync:
            self.sync_graph.deduce_relations()
        self._bag_data.update(bag_data)
        
    def _auto_decompress(self, topic_data: list, topic_name: str = ""):
        def match_func(proto_desc):
            if topic_name in self.TOPIC_AUTO_DECOMPRESS_MAP:
                return self.TOPIC_AUTO_DECOMPRESS_MAP[topic_name]
            for k, v in self.AUTO_DECOMPRESS_MAP.items():
                if k in proto_desc:
                    return v
            return None
        if len(topic_data) > 0:
            proto_data = [d["data"] for d in topic_data]
            proto_desc = proto_data[0].DESCRIPTOR.name
            func = match_func(proto_desc)
            if func is not None:
                if topic_name == self.CAMERA2_DEPTH_TOPIC:
                    depth_info = self.get_camera2_depth_info()
                    if depth_info is None:
                        raise ValueError(
                            f"missing {self.CAMERA2_DEPTH_INFO_TOPIC} required to decode camera2 depth"
                        )
                    shape = (int(depth_info.height), int(depth_info.width))
                    decompressed_data = func(proto_data, shape)
                else:
                    decompressed_data = func(proto_data)
                for idx, d in enumerate(decompressed_data):
                    topic_data[idx]["decode_data"] = d
                topic_data = [d for d in topic_data if d["decode_data"] is not None]
        return topic_data
    
    def get_bag_data(self):
        return self._bag_data

    def get_topic_schema(self, topic_name) -> str:
        if topic_name not in self.topic_schemas:
            print(f">> {topic_name} << is not in bag.")
            return ""
        return self.topic_schemas[topic_name]

    def get_topic_data(self, topic_name):
        self.load_topics(topic_name)
        if topic_name not in self._bag_data:
            return None
        return self._bag_data[topic_name]

    def get_stereo_calibration(self, topic_name: str = DEFAULT_STEREO_CALIBRATION_TOPIC):
        if topic_name in self._stereo_calibration_cache:
            return self._stereo_calibration_cache[topic_name]
        if topic_name not in self.all_topic_names:
            self._stereo_calibration_cache[topic_name] = None
            return None

        last_msg = None
        with open(self._bag_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, _channel, _message, decoded in reader.iter_decoded_messages(topics=[topic_name]):
                last_msg = decoded
        self._stereo_calibration_cache[topic_name] = last_msg
        return last_msg

    def get_stereo_calibration_K_and_size(self, topic_name: str = DEFAULT_STEREO_CALIBRATION_TOPIC):
        stereo_cal = self.get_stereo_calibration(topic_name)
        if stereo_cal is None:
            raise ValueError(f"No StereoCalibration message in bag for topic {topic_name}.")
        k_list = list(getattr(stereo_cal, "K", []) or [])
        if len(k_list) != 9:
            raise ValueError(f"StereoCalibration.K has length {len(k_list)}, expected 9.")
        K = np.asarray(k_list, dtype=np.float64).reshape(3, 3)
        w = int(getattr(stereo_cal, "width", 0) or 0)
        h = int(getattr(stereo_cal, "height", 0) or 0)
        if w <= 0 or h <= 0:
            raise ValueError("StereoCalibration width/height invalid.")
        return K, w, h

    @staticmethod
    def depth_to_point_cloud(
        depth_data: np.ndarray,
        K: np.ndarray,
        min_depth: float = 0.0,
        max_depth: float = -1.0,
        pixel_stride: int = 1,
    ) -> np.ndarray:
        """
        Convert one depth frame to point cloud in camera coordinate.

        Args:
            depth_data (np.ndarray): [H, W] or [H, W, 1], unit meter.
            K (np.ndarray): Camera intrinsic matrix (3x3).
            min_depth (float): Keep points with depth > min_depth.
            max_depth (float): Keep points with depth < max_depth, <=0 means disable.
            pixel_stride (int): Point cloud stride, 1 means full resolution.

        Returns:
            np.ndarray: [N, 3], xyz in camera coordinate.
        """
        if depth_data is None:
            return np.empty((0, 3), dtype=np.float32)
        if not isinstance(depth_data, np.ndarray):
            depth_data = np.asarray(depth_data)
        if depth_data.ndim == 3:
            if depth_data.shape[-1] != 1:
                raise ValueError(f"depth_data last dim should be 1, got {depth_data.shape}")
            depth_data = depth_data[..., 0]
        if depth_data.ndim != 2:
            raise ValueError(f"depth_data should be [H, W] or [H, W, 1], got {depth_data.shape}")
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K shape should be (3, 3), got {K.shape}")

        stride = max(1, int(pixel_stride))
        z = np.asarray(depth_data[::stride, ::stride], dtype=np.float64)
        valid = np.isfinite(z) & (z > float(min_depth))
        if max_depth is not None and max_depth > 0:
            valid &= z < float(max_depth)
        vv, uu = np.where(valid)
        if len(vv) == 0:
            return np.empty((0, 3), dtype=np.float32)

        zv = z[vv, uu]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # Strided pixel coordinates map back to original image space.
        uu_full = uu.astype(np.float64) * float(stride)
        vv_full = vv.astype(np.float64) * float(stride)
        xv = (uu_full - cx) * zv / fx
        yv = (vv_full - cy) * zv / fy
        points = np.column_stack((xv, yv, zv)).astype(np.float32)
        return points

    def convert_depth_to_point_cloud(
        self,
        depth_data: np.ndarray,
        min_depth: float = 0.0,
        max_depth: float = -1.0,
        pixel_stride: int = 1,
        stereo_calibration_topic: str = DEFAULT_STEREO_CALIBRATION_TOPIC,
    ) -> np.ndarray:
        """
        Convert one depth frame to point cloud with StereoCalibration intrinsics.

        Args:
            depth_data (np.ndarray): [H, W] or [H, W, 1] depth map in meters.
            min_depth (float): Keep points with depth > min_depth.
            max_depth (float): Keep points with depth < max_depth, <=0 means disable.
            pixel_stride (int): Point cloud stride, 1 means full resolution.
            stereo_calibration_topic (str): Topic to load StereoCalibration(K/width/height).

        Returns:
            np.ndarray: [N, 3], xyz in camera coordinate.
        """
        K, cal_w, cal_h = self.get_stereo_calibration_K_and_size(stereo_calibration_topic)
        if depth_data is None:
            return np.empty((0, 3), dtype=np.float32)

        if depth_data.ndim == 3:
            depth_hw = depth_data[..., 0]
        else:
            depth_hw = depth_data
        if depth_hw.shape[:2] != (cal_h, cal_w):
            if cv2 is None:
                raise ImportError(
                    "opencv-python is required when depth resolution differs from StereoCalibration size."
                )
            depth_hw = cv2.resize(depth_hw, (cal_w, cal_h), interpolation=cv2.INTER_NEAREST)
            depth_data = depth_hw[..., None]

        return self.depth_to_point_cloud(
            depth_data=depth_data,
            K=K,
            min_depth=min_depth,
            max_depth=max_depth,
            pixel_stride=pixel_stride,
        )

    def get_camera2_depth_info(self, topic_name: str = CAMERA2_DEPTH_INFO_TOPIC):
        if self._camera2_depth_info_cache is not None:
            return self._camera2_depth_info_cache
        if topic_name not in self.all_topic_names:
            self._camera2_depth_info_cache = None
            return None

        last_msg = None
        with open(self._bag_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, _channel, _message, decoded in reader.iter_decoded_messages(topics=[topic_name]):
                last_msg = decoded
        self._camera2_depth_info_cache = last_msg
        return last_msg

    def get_camera2_depth_info_K_xi_alpha(self, topic_name: str = CAMERA2_DEPTH_INFO_TOPIC):
        depth_info = self.get_camera2_depth_info(topic_name)
        if depth_info is None:
            raise ValueError(f"No DepthInfo message in bag for topic {topic_name}.")
        k_list = list(getattr(depth_info, "K", []) or [])
        if len(k_list) != 9:
            raise ValueError(f"DepthInfo.K has length {len(k_list)}, expected 9.")
        K = np.asarray(k_list, dtype=np.float64).reshape(3, 3)
        xi = float(getattr(depth_info, "xi", 0.0) or 0.0)
        alpha = float(getattr(depth_info, "alpha", 0.0) or 0.0)
        h = int(getattr(depth_info, "height", 0) or 0)
        w = int(getattr(depth_info, "width", 0) or 0)
        if h <= 0 or w <= 0:
            raise ValueError("DepthInfo height/width invalid.")
        return K, xi, alpha, h, w

    @staticmethod
    def double_sphere_rays(shape_hw: tuple, K: np.ndarray, xi: float, alpha: float) -> np.ndarray:
        h, w = shape_hw
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        mx = (xx - cx) / fx
        my = (yy - cy) / fy
        r2 = mx * mx + my * my
        sqrt_arg = 1.0 - (2.0 * alpha - 1.0) * r2
        valid = sqrt_arg >= 0
        sqrt_arg = np.maximum(sqrt_arg, 0.0)
        mz = (1.0 - alpha * alpha * r2) / (alpha * np.sqrt(sqrt_arg) + (1.0 - alpha))
        k_arg = mz * mz + (1.0 - xi * xi) * r2
        valid &= k_arg >= 0
        k_arg = np.maximum(k_arg, 0.0)
        scale = (mz * xi + np.sqrt(k_arg)) / np.maximum(mz * mz + r2, 1e-12)
        rays = np.stack([scale * mx, scale * my, scale * mz - xi], axis=-1).astype(np.float32)
        rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
        rays[~valid] = np.nan
        return rays

    @staticmethod
    def depth_to_point_cloud_double_sphere(
        depth_data: np.ndarray,
        K: np.ndarray,
        xi: float,
        alpha: float,
        min_ray_z: float = 0.1,
        max_range: float = -1.0,
        pixel_stride: int = 1,
    ) -> np.ndarray:
        """
        Convert one depth frame to point cloud using Double Sphere camera model.

        Args:
            depth_data (np.ndarray): [H, W] or [H, W, 1], unit meter.
            K (np.ndarray): Camera intrinsic matrix (3x3).
            xi (float): Double Sphere xi parameter.
            alpha (float): Double Sphere alpha parameter.
            min_ray_z (float): Keep points whose ray z > min_ray_z.
            max_range (float): Keep points within this range, <=0 means disable.
            pixel_stride (int): Point cloud stride, 1 means full resolution.

        Returns:
            np.ndarray: [N, 3], xyz in camera coordinate.
        """
        if depth_data is None:
            return np.empty((0, 3), dtype=np.float32)
        if not isinstance(depth_data, np.ndarray):
            depth_data = np.asarray(depth_data)
        if depth_data.ndim == 3:
            if depth_data.shape[-1] != 1:
                raise ValueError(f"depth_data last dim should be 1, got {depth_data.shape}")
            depth_data = depth_data[..., 0]
        if depth_data.ndim != 2:
            raise ValueError(f"depth_data should be [H, W] or [H, W, 1], got {depth_data.shape}")

        stride = max(1, int(pixel_stride))
        depth = np.asarray(depth_data[::stride, ::stride], dtype=np.float32)
        rays = McapLoader.double_sphere_rays(depth.shape, K, xi, alpha)
        valid = (
            np.isfinite(depth)
            & (depth > 0)
            & np.isfinite(rays).all(axis=-1)
            & (rays[..., 2] > float(min_ray_z))
        )
        yy, xx = np.where(valid)
        if len(yy) == 0:
            return np.empty((0, 3), dtype=np.float32)

        rv = rays[yy, xx]
        d = depth[yy, xx]
        pts = rv * (d / np.maximum(rv[:, 2], 1e-6))[:, None]
        if max_range is not None and max_range > 0:
            keep = np.linalg.norm(pts, axis=1) <= float(max_range)
            pts = pts[keep]
        return pts.astype(np.float32)

    def convert_camera2_depth_to_point_cloud(
        self,
        depth_data: np.ndarray,
        min_ray_z: float = 0.1,
        max_range: float = -1.0,
        pixel_stride: int = 1,
        depth_info_topic: str = CAMERA2_DEPTH_INFO_TOPIC,
    ) -> np.ndarray:
        """
        Convert one camera2 depth frame to point cloud with DepthInfo intrinsics.

        Args:
            depth_data (np.ndarray): [H, W] or [H, W, 1] depth map in meters.
            min_ray_z (float): Keep points whose ray z > min_ray_z.
            max_range (float): Keep points within this range, <=0 means disable.
            pixel_stride (int): Point cloud stride, 1 means full resolution.
            depth_info_topic (str): Topic to load DepthInfo(K/xi/alpha/height/width).

        Returns:
            np.ndarray: [N, 3], xyz in camera coordinate.
        """
        K, xi, alpha, cal_h, cal_w = self.get_camera2_depth_info_K_xi_alpha(depth_info_topic)
        if depth_data is None:
            return np.empty((0, 3), dtype=np.float32)

        if depth_data.ndim == 3:
            depth_hw = depth_data[..., 0]
        else:
            depth_hw = depth_data
        if depth_hw.shape[:2] != (cal_h, cal_w):
            if cv2 is None:
                raise ImportError(
                    "opencv-python is required when depth resolution differs from DepthInfo size."
                )
            depth_hw = cv2.resize(depth_hw, (cal_w, cal_h), interpolation=cv2.INTER_NEAREST)
            depth_data = depth_hw[..., None]

        return self.depth_to_point_cloud_double_sphere(
            depth_data=depth_data,
            K=K,
            xi=xi,
            alpha=alpha,
            min_ray_z=min_ray_z,
            max_range=max_range,
            pixel_stride=pixel_stride,
        )

    # Get a frame of data for a topic based on seq num
    def get_topic_data_by_seq_num(self, topic_name, seq_id, sync_topics=[]):
        """
        Retrieve data for a specific topic and sequence number, along with synchronized data for other topics.

        Args:
            topic_name (str): The name of the primary topic to retrieve data for.
            seq_id (int): The sequence number of the data to retrieve.
            sync_topics (list, optional): A list of topic names to retrieve synchronized data for. Defaults to an empty list.

        Returns:
            dict: A dictionary containing the data for the primary topic and any synchronized topics. 
              If the primary topic or any synchronized topic data is not found, the corresponding value will be None.
        """
        # load topic data
        self.load_topics(topic_name)
        self.load_topics(sync_topics)
        if topic_name not in self._bag_data:
            print(f"{topic_name} is noe in bag_data.")
            return None
        hit_data = {}
        list_idx = self.seq2idx[topic_name].get(seq_id, None)
        if list_idx is None:
            return None
        hit_data[topic_name] = self._bag_data[topic_name][list_idx]
        # get sync data
        if isinstance(sync_topics, str):
            sync_topics = [sync_topics]
        # deduplication
        sync_topics = [i for i in sync_topics if i!=topic_name]
        if len(sync_topics) > 0:
            cur_sync_info = self.sync_graph.get_relations(topic_name, seq_id)
            for sname in sync_topics:
                sync_topic_seq = cur_sync_info.get(sname, None)
                if sync_topic_seq is None:
                    hit_data[sname] = None
                    continue
                data_idx = self.seq2idx[sname].get(sync_topic_seq, None)
                if data_idx is None:
                    hit_data[sname] = None
                    continue
                sync_data = self._bag_data[sname][data_idx]
                hit_data[sname] = sync_data

        return hit_data

    def get_header(self):
        return self._topic_header
    def get_attachments(self):
        return self._topic_attachments
    def get_meta(self):
        return self._topic_meta

    def get_topic_seq_num(self, topic_name: str) -> list:
        self.load_topics(topic_name)
        return list(self.seq2idx[topic_name].keys())

    def get_valid_topic_names(self):
        return list(self._bag_data.keys())

    def get_all_topic_names(self):
        return self.all_topic_names

    def get_topic_frequency(self, topic_name) -> float:
        return self.topic_frequency_info[topic_name]

    def get_topic_msg_count(self, topic_name) -> int:
        return len(self._bag_data[topic_name])

    def get_bag_name(self)->str:
        return osp.basename(self._bag_path)

    def get_bag_path(self) -> str:
        return self._bag_path
    
    def __del__(self):
        self.close()
    
    def close(self):
        if not self._stream.closed:
            self._stream.close()

    def __repr__(self):
        msg_count = {}
        for tn in self._bag_data:
            msg_count[tn] = len(self._bag_data[tn])
        return (
            f"bag name: {osp.basename(self._bag_path)}\n"
            f"timestamp_range: [{self.msg_start_time}, {self.msg_end_time}]\n"
            f"bag time length: {ns_to_s(self.msg_end_time - self.msg_start_time):.1f} s\n"
            f"all topic names: {self.all_topic_names}\n"
            f"loaded topic names: {self.get_valid_topic_names()}\n"
            f"topic frequncy:\n"
            f"{json.dumps(self.topic_frequency_info, indent=4)}\n"
            f"topic msg count:\n"
            f"{json.dumps(msg_count, indent=4)}\n"
        )