import h5py
import os.path as osp
import os
from utils.mcaploader import McapLoader
from utils.interpolate import get_inter_data
from utils.h5_tools import write_array, write_img
from utils.io import load_json
import numpy as np
import argparse

# reference topic
REF_TOPIC = "/robot0/sensor/camera0/compressed"

PROCESS_TOPIC = [
    # mid fisheye camera
    "/robot0/sensor/camera0/compressed",
    # left stereo camera, sync with timestamp
    "/robot0/sensor/camera1/compressed",
    # right stereo camera, sync with timestamp
    "/robot0/sensor/camera2/compressed",
    # imu, sync with interpolation
    "/robot0/sensor/imu", 
    # tactile sensor, sync with interpolation
    "/robot0/sensor/tactile_left",
    "/robot0/sensor/tactile_right",
    # gripper width, sync with interpolation
    "/robot0/sensor/magnetic_encoder",
    # gripper pose, sync with interpolation
    "/robot0/vio/eef_pose",
]

def convert_single_mcap_to_h5(mcap_file, output_file, args):
    """Converts a single MCAP file to an HDF5 format.
    
    Args:
        mcap_file (str): The path to the input MCAP file.
        output_file (str): The path to the output HDF5 file.
        args (Namespace): Command-line arguments containing options for stereo camera, tactile data, and IMU data.
    
    Raises:
        SystemExit: Exits the program if stereo camera image retrieval fails.
    """
    bag = McapLoader(mcap_file)
    # print(bag)

    # step 1
    # parse all data we need
    bag.load_topics(PROCESS_TOPIC, auto_sync=False)

    # step2
    ref_topic_data = bag.get_topic_data(REF_TOPIC)
    ref_timestamp = [d["data"].header.timestamp for d in ref_topic_data]
    ref_topic_seq_num = bag.get_topic_seq_num(REF_TOPIC)

    # step 2
    # write all data to h5df file
    with h5py.File(output_file, "w") as f:
        observations = f.create_group("observations")
        camera_group = observations.create_group("cameras")

        # export mid camera data
        print(f"Start write mid camera data...")
        sync_data = [d["decode_data"] for d in ref_topic_data]
        write_img(camera_group, "mid_fisheye_color", sync_data)
        
        # export stereo camera data
        if args.stereo_camera:
            print(f"Start write stereo camera data...")
            left_wide_sync_data = []
            right_wide_sync_data = []
            for _idx in ref_topic_seq_num:
                data = bag.get_topic_data_by_seq_num(
                    REF_TOPIC,
                    _idx,
                    sync_topics=[
                        "/robot0/sensor/camera1/compressed",
                        "/robot0/sensor/camera2/compressed",
                    ],
                )
                if data is None:
                    print(f"Get stereo camera image failed at seq_num={_idx}")
                    os._exit(0)
                left_wide_sync_data.append(
                    data["/robot0/sensor/camera1/compressed"]["decode_data"]
                )
                right_wide_sync_data.append(
                    data["/robot0/sensor/camera2/compressed"]["decode_data"]
                )
            write_img(camera_group, "left_wide_color", left_wide_sync_data)
            write_img(camera_group, "right_wide_color", right_wide_sync_data)

        # export tactile data
        if args.tactile:
            print(f"Start write tactile data...")
            tactile_group = observations.create_group("tactile")
            tactile_left_data = get_inter_data(
                bag, "/robot0/sensor/tactile_left", ref_timestamp, inter_type="linear"
            )
            write_array(tactile_group, "left", np.array(tactile_left_data))
            tactile_right_data = get_inter_data(
                bag, "/robot0/sensor/tactile_right", ref_timestamp, inter_type="linear"
            )
            write_array(tactile_group, "right", np.array(tactile_right_data))

        # export pose data
        print(f"Start write pose data...")
        eef_pose_data = get_inter_data(
            bag, "/robot0/vio/eef_pose", ref_timestamp, inter_type="pose"
        )
        gripper_data = get_inter_data(
            bag, "/robot0/sensor/magnetic_encoder", ref_timestamp, inter_type="linear"
        )
        action_array = np.concatenate([eef_pose_data, gripper_data], axis=1)
        write_array(observations, "eef_pos", action_array)
        write_array(f, "action", action_array)

        if args.imu:
            print(f"Start write imu data...")
            imu_data = get_inter_data(
                bag, "/robot0/sensor/imu", ref_timestamp, inter_type="linear"
            )
            write_array(observations, "imu", imu_data)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert DAS mcap to h5 file")
    parser.add_argument("--mcap-file", type=str, default="", help="input mcap file path")
    parser.add_argument("--task-dir", type=str, default="", help="the task dir in matrix studio")
    parser.add_argument("--out-path", type=str, default="", help="output h5 file path")
    parser.add_argument("--imu", action="store_true", help="export imu data")
    parser.add_argument("--stereo-camera", action="store_true", help="export stereo camera data")
    parser.add_argument("--tactile", action="store_true", help="export tactile sensor data")
    parser.add_argument("--resume", action="store_true", help="skip processed H5 files")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    mcap_file = args.mcap_file
    task_dir = args.task_dir
    
    process_list = []
    if not mcap_file:
        assert task_dir != ""
        if (not osp.exists(task_dir)) or (not osp.isdir(task_dir)):
            raise FileNotFoundError
        mcap_list_file = osp.join(task_dir, "vio_result.json")
        if not osp.exists(mcap_list_file):
            raise FileNotFoundError

        output_path = osp.join(task_dir, "h5")
        os.makedirs(output_path, exist_ok=True)

        mcap_info = load_json(mcap_list_file)
        mcap_list = mcap_info.get("success_mcap_files", [])
        mcap_list = set(mcap_list)
        if len(mcap_list) == 0:
            print(f"No mcap files need to be processed, success_mcap_files is empty")
        for mcap_f in mcap_list:
            output_file = osp.join(output_path, osp.basename(mcap_f).replace(".mcap", ".h5"))
            process_list.append([mcap_f, output_file])
    else:
        if (not osp.exists(mcap_file)) or (not osp.isfile(mcap_file)):
            raise FileNotFoundError
        
        output_path = args.out_path
        if not output_path:
            mcap_dir = osp.dirname(mcap_file)
            output_path = mcap_dir if mcap_dir else os.getcwd()
        os.makedirs(output_path, exist_ok=True)
        
        output_file = osp.join(output_path, osp.basename(mcap_file).replace(".mcap", ".h5"))
        process_list.append([mcap_file, output_file])
        
    for idx, (f_mcap, f_h5) in enumerate(process_list):
        print(F"######################## Process {idx+1}/{len(process_list)} ########################")
        print(f"mcap: {f_mcap}")
        print(f"h5: {f_h5}")
        if args.resume:
            if osp.exists(f_h5):
                print(f"enable resume and {f_h5} has been generated")
                continue
        convert_single_mcap_to_h5(f_mcap, f_h5, args)
