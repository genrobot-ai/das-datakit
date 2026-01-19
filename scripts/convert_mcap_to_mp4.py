import os.path as osp
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pandas as pd
import numpy as np
import argparse
import glob

from utils.mcaploader import McapLoader
from utils.interpolate import get_inter_data
from utils.io import load_json


def convert_dual_arm_mcap_to_mp4_csv(mcap_file, output_base_dir):
    """Converts a dual-arm MCAP file to MP4 video and CSV files for both arms."""
    print(f'Processing dual-arm: {mcap_file}')
    
    bag = McapLoader(mcap_file)
    print(f"Topics found: {len(bag.all_topic_names)}")
    
    # Define topics for both arms
    left_arm_topics = [
        "/robot0/sensor/camera0/compressed",   # Left camera
        "/robot0/sensor/magnetic_encoder",     # Left gripper
        "/robot0/vio/eef_pose"                 # Left pose
    ]
    
    right_arm_topics = [
        "/robot1/sensor/camera0/compressed",   # Right camera
        "/robot1/sensor/magnetic_encoder",     # Right gripper
        "/robot1/vio/eef_pose"                 # Right pose
    ]
    
    all_topics = left_arm_topics + right_arm_topics
    
    # Filter topics that exist
    existing_topics = [topic for topic in all_topics if topic in bag.all_topic_names]
    print(f"Loading topics: {existing_topics}")
    
    if len(existing_topics) == 0:
        print("No required topics found")
        return False
    
    bag.load_topics(existing_topics, auto_sync=False)
    
    # Get reference topic (left camera)
    REF_TOPIC = "/robot0/sensor/camera0/compressed"
    ref_topic_data = bag.get_topic_data(REF_TOPIC)
    if ref_topic_data is None or len(ref_topic_data) == 0:
        print(f"Error: No data found for reference topic {REF_TOPIC}")
        return False
    
    print(f'Found {len(ref_topic_data)} left camera frames')
    
    # Prepare left camera data
    left_sync_data = []
    left_timestamps = []
    left_seq_nums = []
    
    for d in ref_topic_data:
        if d.get("decode_data") is not None:
            left_sync_data.append(d["decode_data"])
            
            if "data" in d and hasattr(d["data"], "header"):
                header = d["data"].header
                timestamp = getattr(header, 'timestamp', None)
                left_timestamps.append(timestamp)
                
                seq_num = getattr(header, 'sequence_num', None)
                if seq_num is None:
                    seq_num = getattr(header, 'seq', None)
                left_seq_nums.append(seq_num)
            else:
                left_timestamps.append(None)
                left_seq_nums.append(None)
    
    if all(data is None for data in left_sync_data):
        print(f"Left camera data is all None")
        return False
    
    left_valid_frames = len([d for d in left_sync_data if d is not None])
    print(f"Valid left camera images: {left_valid_frames}")
    
    # Get right camera data
    right_camera_topic = "/robot1/sensor/camera0/compressed"
    right_topic_data = bag.get_topic_data(right_camera_topic) if right_camera_topic in existing_topics else None
    
    # Register sync relation for right camera
    if right_topic_data is not None:
        bag.register_sync_relation_with_time(REF_TOPIC, right_camera_topic)
        print(f"Registered sync relation: {REF_TOPIC} -> {right_camera_topic}")
    
    # Prepare right camera data with sync
    right_sync_data = []
    right_timestamps = []
    right_seq_nums = []
    aligned_timestamps = []  # Timestamps aligned to left camera
    
    if right_topic_data is not None:
        for i, left_ts in enumerate(left_timestamps):
            if i < len(left_sync_data) and left_sync_data[i] is not None and left_ts is not None:

                left_seq_num = left_seq_nums[i] if i < len(left_seq_nums) else None
                if left_seq_num is None:
                    right_sync_data.append(None)
                    right_timestamps.append(None)
                    right_seq_nums.append(None)
                    aligned_timestamps.append(None)
                    continue
                # Get synchronized right camera data
                sync_data = bag.get_topic_data_by_seq_num(
                    REF_TOPIC,
                    left_seq_num,  # Using index as seq_num
                    sync_topics=[right_camera_topic]
                )
                
                if sync_data and right_camera_topic in sync_data:
                    right_data = sync_data[right_camera_topic]
                    if "decode_data" in right_data and right_data["decode_data"] is not None:
                        right_sync_data.append(right_data["decode_data"])
                        
                        # Get original right camera timestamp
                        if "data" in right_data and hasattr(right_data["data"], "header"):
                            header = right_data["data"].header
                            right_ts = getattr(header, 'timestamp', None)
                            right_timestamps.append(right_ts)
                            
                            seq_num = getattr(header, 'sequence_num', None)
                            if seq_num is None:
                                seq_num = getattr(header, 'seq', None)
                            right_seq_nums.append(seq_num)
                        else:
                            right_timestamps.append(None)
                            right_seq_nums.append(None)
                        
                        # Use left timestamp as aligned timestamp
                        aligned_timestamps.append(left_ts)
                    else:
                        #print('append none 1')
                        right_sync_data.append(None)
                        right_timestamps.append(None)
                        right_seq_nums.append(None)
                        aligned_timestamps.append(None)
                else:
                    #print('append none 2')
                    right_sync_data.append(None)
                    right_timestamps.append(None)
                    right_seq_nums.append(None)
                    aligned_timestamps.append(None)
            else:
                #print('append none 3')
                right_sync_data.append(None)
                right_timestamps.append(None)
                right_seq_nums.append(None)
                aligned_timestamps.append(None)

    if right_sync_data is None:
        print('right_sync_data is None empty')
    
    # Get base name for folder
    base_name = osp.splitext(osp.basename(mcap_file))[0]
    
    # Create main output folder
    output_folder = osp.join(output_base_dir, base_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Create left and right subfolders
    left_folder = osp.join(output_folder, "left_hand")
    right_folder = osp.join(output_folder, "right_hand")
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)
    
    # ===== Process LEFT arm =====
    print(f"\n{'='*40}")
    print(f"Processing LEFT arm")
    print(f"{'='*40}")
    
    # Save left MP4 video
    left_mp4_path = osp.join(left_folder, "video.mp4")
    print(f"Saving left video...")
    
    left_video_writer = None
    fps = 30
    left_saved_frames = 0
    
    for i, img_data in enumerate(left_sync_data):
        if img_data is None:
            continue
            
        if left_video_writer is None:
            h, w, c = img_data.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            left_video_writer = cv2.VideoWriter(left_mp4_path, fourcc, fps, (w, h))
        
        left_video_writer.write(img_data)
        left_saved_frames += 1
    
    if left_video_writer is not None:
        left_video_writer.release()
        print(f"Left video saved: {left_mp4_path} ({left_saved_frames} frames)")
    
    # Save left timestamps
    left_timestamps_csv_path = osp.join(left_folder, "timestamps.csv")
    print(f"Saving left timestamps...")
    
    left_timestamps_data = []
    for i in range(len(left_sync_data)):
        if left_sync_data[i] is not None:
            left_timestamps_data.append({
                "frame_index": i,
                "seq": left_seq_nums[i] if i < len(left_seq_nums) and left_seq_nums[i] is not None else i,
                "header_stamp": left_timestamps[i] if i < len(left_timestamps) and left_timestamps[i] is not None else 0,
                "aligned_stamp": left_timestamps[i] if i < len(left_timestamps) and left_timestamps[i] is not None else 0
            })
    
    if left_timestamps_data:
        left_timestamps_df = pd.DataFrame(left_timestamps_data)
        left_timestamps_df.to_csv(left_timestamps_csv_path, index=False)
        print(f"Left timestamps saved: {left_timestamps_csv_path}")
    
    # Get left sensor data
    left_pose_data = None
    left_gripper_data = None
    
    if "/robot0/vio/eef_pose" in existing_topics:
        left_pose_data = get_inter_data(
            bag, "/robot0/vio/eef_pose", left_timestamps, inter_type="pose"
        )
    
    if "/robot0/sensor/magnetic_encoder" in existing_topics:
        left_gripper_data = get_inter_data(
            bag, "/robot0/sensor/magnetic_encoder", left_timestamps, inter_type="linear"
        )
    
    # Save left sensor data
    left_data_csv_path = osp.join(left_folder, "data.csv")
    print(f"Saving left sensor data...")
    
    left_sensor_data = []
    for i, ts in enumerate(left_timestamps):
        if i >= len(left_sync_data) or left_sync_data[i] is None:
            continue
            
        # Get pose data
        if left_pose_data is not None and i < len(left_pose_data):
            pose = left_pose_data[i]
            if len(pose) >= 7:
                x, y, z, qx, qy, qz, qw = pose[:7]
            else:
                x, y, z, qx, qy, qz, qw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        else:
            x, y, z, qx, qy, qz, qw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        
        # Get gripper data
        if left_gripper_data is not None and i < len(left_gripper_data):
            if left_gripper_data[i].size > 0:
                distance = float(left_gripper_data[i][0]) if isinstance(left_gripper_data[i], np.ndarray) else float(left_gripper_data[i])
            else:
                distance = 0.0
        else:
            distance = 0.0
        
        left_sensor_data.append({
            "timestamp": ts,
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "qx": float(qx),
            "qy": float(qy),
            "qz": float(qz),
            "qw": float(qw),
            "distance": float(distance)
        })
    
    if left_sensor_data:
        left_sensor_df = pd.DataFrame(left_sensor_data)
        left_sensor_df.to_csv(left_data_csv_path, index=False)
        print(f"Left sensor data saved: {left_data_csv_path}")
    
    # ===== Process RIGHT arm =====
    print(f"\n{'='*40}")
    print(f"Processing RIGHT arm")
    print(f"{'='*40}")
    
    if right_sync_data and any(d is not None for d in right_sync_data):
        right_valid_frames = len([d for d in right_sync_data if d is not None])
        print(f"Valid right camera images: {right_valid_frames}")
        
        # Save right MP4 video
        right_mp4_path = osp.join(right_folder, "video.mp4")
        print(f"Saving right video...")
        
        right_video_writer = None
        right_saved_frames = 0
        
        for i, img_data in enumerate(right_sync_data):
            if img_data is None:
                continue
                
            if right_video_writer is None:
                h, w, c = img_data.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                right_video_writer = cv2.VideoWriter(right_mp4_path, fourcc, fps, (w, h))
            
            right_video_writer.write(img_data)
            right_saved_frames += 1
        
        if right_video_writer is not None:
            right_video_writer.release()
            print(f"Right video saved: {right_mp4_path} ({right_saved_frames} frames)")
        
        # Save right timestamps
        right_timestamps_csv_path = osp.join(right_folder, "timestamps.csv")
        print(f"Saving right timestamps...")
        
        right_timestamps_data = []
        for i in range(len(right_sync_data)):
            if right_sync_data[i] is not None:
                right_timestamps_data.append({
                    "frame_index": i,
                    "seq": right_seq_nums[i] if i < len(right_seq_nums) and right_seq_nums[i] is not None else i,
                    "header_stamp": right_timestamps[i] if i < len(right_timestamps) and right_timestamps[i] is not None else 0,
                    "aligned_stamp": aligned_timestamps[i] if i < len(aligned_timestamps) and aligned_timestamps[i] is not None else 0
                })
        
        if right_timestamps_data:
            right_timestamps_df = pd.DataFrame(right_timestamps_data)
            right_timestamps_df.to_csv(right_timestamps_csv_path, index=False)
            print(f"Right timestamps saved: {right_timestamps_csv_path}")
        
        # Get right sensor data
        right_pose_data = None
        right_gripper_data = None
        
        if "/robot1/vio/eef_pose" in existing_topics:
            right_pose_data = get_inter_data(
                bag, "/robot1/vio/eef_pose", left_timestamps, inter_type="pose"  # Use left timestamps for alignment
            )
        
        if "/robot1/sensor/magnetic_encoder" in existing_topics:
            right_gripper_data = get_inter_data(
                bag, "/robot1/sensor/magnetic_encoder", left_timestamps, inter_type="linear"  # Use left timestamps for alignment
            )
        
        # Save right sensor data
        right_data_csv_path = osp.join(right_folder, "data.csv")
        print(f"Saving right sensor data...")
        
        right_sensor_data = []
        for i, ts in enumerate(left_timestamps):  # Use left timestamps for indexing
            if i >= len(right_sync_data) or right_sync_data[i] is None:
                continue
                
            # Get pose data
            if right_pose_data is not None and i < len(right_pose_data):
                pose = right_pose_data[i]
                if len(pose) >= 7:
                    x, y, z, qx, qy, qz, qw = pose[:7]
                else:
                    x, y, z, qx, qy, qz, qw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
            else:
                x, y, z, qx, qy, qz, qw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
            
            # Get gripper data
            if right_gripper_data is not None and i < len(right_gripper_data):
                if right_gripper_data[i].size > 0:
                    distance = float(right_gripper_data[i][0]) if isinstance(right_gripper_data[i], np.ndarray) else float(right_gripper_data[i])
                else:
                    distance = 0.0
            else:
                distance = 0.0
            
            right_sensor_data.append({
                "timestamp": aligned_timestamps[i] if i < len(aligned_timestamps) and aligned_timestamps[i] is not None else ts,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "qw": float(qw),
                "distance": float(distance)
            })
        
        if right_sensor_data:
            right_sensor_df = pd.DataFrame(right_sensor_data)
            right_sensor_df.to_csv(right_data_csv_path, index=False)
            print(f"Right sensor data saved: {right_data_csv_path}")
    else:
        print("No valid right camera data found")
    
    print(f"\nDual-arm processing completed: {output_folder}")
    return True

def convert_single_arm_mcap_to_mp4_csv(mcap_file, output_base_dir):
    """Converts a single-arm MCAP file to MP4 video and CSV files."""
    print(f'Processing single-arm: {mcap_file}')
    
    bag = McapLoader(mcap_file)
    print(f"Topics found: {len(bag.all_topic_names)}")

    # Load necessary topics
    REF_TOPIC = "/robot0/sensor/camera0/compressed"
    topics_to_load = [
        REF_TOPIC,
        "/robot0/vio/eef_pose",
        "/robot0/sensor/magnetic_encoder"
    ]
    
    # Filter topics that exist
    existing_topics = [topic for topic in topics_to_load if topic in bag.all_topic_names]
    print(f"Loading topics: {existing_topics}")
    
    if len(existing_topics) == 0:
        print("No required topics found")
        return False
    
    bag.load_topics(existing_topics, auto_sync=False)

    # Get camera data
    ref_topic_data = bag.get_topic_data(REF_TOPIC)
    if ref_topic_data is None or len(ref_topic_data) == 0:
        print(f"Error: No data found for {REF_TOPIC}")
        return False
    
    print(f'Found {len(ref_topic_data)} camera frames')
    
    # Prepare data lists
    sync_ref_data = []
    timestamps = []
    seq_nums = []
    
    for d in ref_topic_data:
        if d.get("decode_data") is not None:
            sync_ref_data.append(d["decode_data"])
            
            # Get header information
            if "data" in d and hasattr(d["data"], "header"):
                header = d["data"].header
                
                # Get timestamp
                timestamp = getattr(header, 'timestamp', None)
                timestamps.append(timestamp)
                
                # Get sequence number
                seq_num = getattr(header, 'sequence_num', None)
                if seq_num is None:
                    seq_num = getattr(header, 'seq', None)
                seq_nums.append(seq_num)
            else:
                timestamps.append(None)
                seq_nums.append(None)
    
    # Check if all camera data is None
    if all(data is None for data in sync_ref_data):
        print(f"Camera data is all None")
        return False
    
    valid_frames = len([d for d in sync_ref_data if d is not None])
    print(f"Valid camera images: {valid_frames}")
    
    # Get base name for folder
    base_name = osp.splitext(osp.basename(mcap_file))[0]
    
    # Create output folder
    output_folder = osp.join(output_base_dir, base_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Output file paths
    output_mp4_path = osp.join(output_folder, "video.mp4")
    output_timestamps_csv_path = osp.join(output_folder, "timestamps.csv")
    output_data_csv_path = osp.join(output_folder, "data.csv")
    
    # Save MP4 video
    print(f"Saving video...")
    
    video_writer = None
    fps = 30
    saved_frame_count = 0
    
    for i, img_data in enumerate(sync_ref_data):
        if img_data is None:
            continue
            
        if video_writer is None:
            h, w, c = img_data.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_mp4_path, fourcc, fps, (w, h))
        
        video_writer.write(img_data)
        saved_frame_count += 1
        
        if (i + 1) % 100 == 0 or (i + 1) == len(sync_ref_data):
            print(f"  Frames: {i + 1}/{len(sync_ref_data)}")
    
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved: {output_mp4_path} ({saved_frame_count} frames)")
    
    # Save timestamps to CSV
    print(f"Saving timestamps...")
    
    timestamps_data = []
    for i in range(len(sync_ref_data)):
        if sync_ref_data[i] is not None:
            frame_index = i
            seq_num = seq_nums[i] if i < len(seq_nums) and seq_nums[i] is not None else i
            header_stamp = timestamps[i] if i < len(timestamps) and timestamps[i] is not None else 0
            
            timestamps_data.append({
                "frame_index": frame_index,
                "seq": seq_num,
                "header_stamp": header_stamp,
                "aligned_stamp": header_stamp
            })
    
    timestamps_df = pd.DataFrame(timestamps_data)
    timestamps_df.to_csv(output_timestamps_csv_path, index=False)
    print(f"Timestamps saved: {output_timestamps_csv_path}")
    
    # Save sensor data to CSV
    print(f"Saving sensor data...")
    
    # Get interpolated data
    eef_pose_data = None
    gripper_data = None
    
    if "/robot0/vio/eef_pose" in existing_topics:
        eef_pose_data = get_inter_data(
            bag, "/robot0/vio/eef_pose", timestamps, inter_type="pose"
        )
    
    if "/robot0/sensor/magnetic_encoder" in existing_topics:
        gripper_data = get_inter_data(
            bag, "/robot0/sensor/magnetic_encoder", timestamps, inter_type="linear"
        )
    
    # Create sensor data
    sensor_data = []
    
    for i, ts in enumerate(timestamps):
        if i >= len(sync_ref_data) or sync_ref_data[i] is None:
            continue
            
        # Get pose data
        if eef_pose_data is not None and i < len(eef_pose_data):
            pose = eef_pose_data[i]
            if len(pose) >= 7:
                x, y, z, qx, qy, qz, qw = pose[:7]
            else:
                x, y, z, qx, qy, qz, qw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        else:
            x, y, z, qx, qy, qz, qw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        
        # Get gripper data
        if gripper_data is not None and i < len(gripper_data):
            if gripper_data[i].size > 0:
                distance = float(gripper_data[i][0]) if isinstance(gripper_data[i], np.ndarray) else float(gripper_data[i])
            else:
                distance = 0.0
        else:
            distance = 0.0
        
        sensor_data.append({
            "timestamp": ts,
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "qx": float(qx),
            "qy": float(qy),
            "qz": float(qz),
            "qw": float(qw),
            "distance": float(distance)
        })
    
    # Save to CSV
    if sensor_data:
        sensor_df = pd.DataFrame(sensor_data)
        sensor_df.to_csv(output_data_csv_path, index=False)
        print(f"Sensor data saved: {output_data_csv_path}")
    else:
        print("No sensor data to save")
    
    print(f"Processing completed: {output_folder}")
    return True

def detect_mcap_type(mcap_file):
    """Detect if MCAP file is single-arm or dual-arm."""
    try:
        bag = McapLoader(mcap_file)
        all_topics = bag.all_topic_names
        
        # Check for dual-arm topics
        dual_arm_required = [
            "/robot0/sensor/camera0/compressed",
            "/robot1/sensor/camera0/compressed"
        ]
        
        has_dual_arm = all(topic in all_topics for topic in dual_arm_required)
        
        if has_dual_arm:
            return "dual_arm"
        elif "/robot0/sensor/camera0/compressed" in all_topics:
            return "single_arm"
        else:
            return "unknown"
    except Exception as e:
        print(f"Error detecting MCAP type: {e}")
        return "error"

def convert_single_mcap_to_mp4_csv(mcap_file, output_base_dir):
    """Main conversion function that detects MCAP type and calls appropriate handler."""
    print(f"Analyzing MCAP type: {mcap_file}")
    
    mcap_type = detect_mcap_type(mcap_file)
    print(f"Detected MCAP type: {mcap_type}")
    
    if mcap_type == "dual_arm":
        return convert_dual_arm_mcap_to_mp4_csv(mcap_file, output_base_dir)
    elif mcap_type == "single_arm":
        return convert_single_arm_mcap_to_mp4_csv(mcap_file, output_base_dir)
    else:
        print(f"Unsupported MCAP type: {mcap_type}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Result MCAP(with groundtruth) to MP4 and CSV files")
    parser.add_argument("--mcap-file", type=str, default="", help="Single MCAP file path")
    parser.add_argument("--task-dir", type=str, default="", help="Task directory for batch processing")
    parser.add_argument("--out-path", type=str, default="", help="Output directory path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mcap_file = args.mcap_file
    task_dir = args.task_dir
    
    if not mcap_file and not task_dir:
        print("Error: Either --mcap-file or --task-dir must be specified")
        exit(1)
    
    # Determine output directory
    if args.out_path:
        output_base_dir = args.out_path
    else:
        if mcap_file:
            base_dir = osp.dirname(mcap_file) if osp.dirname(mcap_file) else os.getcwd()
        else:
            base_dir = task_dir
        output_base_dir = osp.join(base_dir, "mp4_result")
    os.makedirs(output_base_dir, exist_ok=True)
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    process_list = []
    
    if task_dir:
        # Batch processing mode
        if not osp.exists(task_dir) or not osp.isdir(task_dir):
            print(f"Error: Task directory not found: {task_dir}")
            exit(1)
        
        mcap_list_file = osp.join(task_dir, "vio_result.json")
        
        if osp.exists(mcap_list_file):
            print(f"Found vio_result.json, loading file list from it")
            try:
                mcap_info = load_json(mcap_list_file)
                mcap_list = mcap_info.get("success_mcap_files", [])
                
                if not mcap_list:
                    print("vio_result.json is empty, will scan directory for MCAP files")
                else:
                    print(f"Found {len(mcap_list)} files in vio_result.json")
                    
            except Exception as e:
                print(f"Error reading vio_result.json: {e}")
                print("Will scan directory for MCAP files")
                mcap_list = []
        else:
            print("No vio_result.json found, will scan directory for MCAP files")
            mcap_list = []
        
        if not mcap_list:
            mcap_files = glob.glob(osp.join(task_dir, "**/*.mcap"), recursive=True)
            mcap_files.extend(glob.glob(osp.join(task_dir, "*.mcap")))
            
            # 去重
            mcap_list = list(set(mcap_files))
            print(f"Found {len(mcap_list)} MCAP files in directory")
        
        if not mcap_list:
            print("No MCAP files to process")
            exit(0)
        
        for mcap_f in mcap_list:
            if osp.exists(mcap_f):
                process_list.append(mcap_f)
            else:
                print(f"Warning: File not found, skipping: {mcap_f}")
        
        print(f"Total files to process: {len(process_list)}")
    
    else:
        # Single file processing
        if not osp.exists(mcap_file) or not osp.isfile(mcap_file):
            print(f"Error: MCAP file not found: {mcap_file}")
            exit(1)
        
        process_list.append(mcap_file)
    
    # Process files
    print(f"Processing {len(process_list)} files")
    print(f"Output directory: {output_base_dir}")
    
    success_count = 0
    for idx, mcap_file in enumerate(process_list):
        print(f"\n{'='*60}")
        print(f"Processing [{idx+1}/{len(process_list)}]: {osp.basename(mcap_file)}")
        print(f"{'='*60}")
        
        try:
            if convert_single_mcap_to_mp4_csv(mcap_file, output_base_dir):
                success_count += 1
        except Exception as e:
            print(f"Error processing {mcap_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nProcessing complete. Success: {success_count}/{len(process_list)}")