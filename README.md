# **DAS: Data Acquisition System**

Enable embodied intelligence data acquisition to be as simple and natural as shooting a video.

# 📋 Contents

- [📦 Overview](#📦-overview)
- [🧹 Data Collection](#🧹-data-collection)
- [📚 Data Format](#📚-data-format)
  - [Mcap](#mcap)
  - [H5](#h5)
- [📖 Tutorials](#📖-tutorials)
  - [Installation](#installation)
  - [Quick Start](#quick-start)

# 📦 Overview

The DAS dataset (pronounced /dʌs/) is a public dataset for embodied intelligence developed based on data collected by the DAS device. As embodied intelligence continues to evolve, there is an urgent need for high-quality, comprehensive datasets to support technological research and application development. By releasing this dataset to the public, we aim to provide solid data support for the advancement of research in the embodied intelligence industry.

For this purpose, the DAS dataset is constructed using data collected by the DAS device, which captures a wealth of diverse sensor data. These data cover various scenarios and task contexts applicable to embodied intelligence, carefully selected to reflect the diverse environmental interactions, task execution processes and complex perception requirements faced by embodied intelligent systems. The rich diversity and comprehensiveness of the DAS dataset will encourage the development of methods that enable embodied intelligent systems to adapt to different real-world scenarios and complete complex tasks reliably.

A key feature of the DAS dataset is its data format and storage optimization. The entire dataset is stored in MCAP format files, a format well-suited for handling multi-sensor data streams in embedded and intelligent systems. Moreover, we adopt efficient compression methods during the data storage process, which significantly reduces the storage space occupied by individual data packets. This optimization not only facilitates convenient storage and management of the dataset but also enhances the efficiency of data transmission and loading, providing great convenience for researchers in data usage and algorithm training.

About DAS product: [genrobot.ai](https://www.genrobot.ai/)

# 🧹 Data Collection

TBD

# 📚 Data Format

## Mcap

This section describes the data format used in mcap

### camera sensor data

Related topics:

```shell
# mid fisheye camera
/robot0/sensor/camera0/compressed
# left stereo camera
/robot0/sensor/camera1/compressed
# right stereo camera
/robot0/sensor/camera2/compressed
```

How to read

```python
bag = McapLoader(mcap_file)
topic_data = bag.get_topic_data("/robot0/sensor/camera0/compressed")
print(topic_data["decode_data"])
"""
[
    [H, W, C]: np.ndarray, bgr
    ...
]
"""

```

### stereo depth sensor data

Related topics:

```shell
# hue encoded depth stream (H264)
/robot0/sensor/depth/compressed
# stereo calibration
/robot0/sensor/depth/stereo_calibration
```

How to read

```python
bag = McapLoader(mcap_file)
topic_data = bag.get_topic_data("/robot0/sensor/depth/compressed")
print(topic_data[0]["decode_data"])
"""
[
    [H, W, 1]: np.ndarray, depth map in meters in camera2 coordinate(float32)
    ...
]
"""
```

Notes:

- `topic_data` is a list, each item is one frame dict.
- Use `topic_data[i]["decode_data"]` to access the i-th depth frame.
- `decode_data` shape is `[H, W, 1]`, dtype is `float32`, and the unit is meter.

How to convert depth to point cloud in camera coordinate

```python
bag = McapLoader(mcap_file)
topic_data = bag.get_topic_data("/robot0/sensor/depth/compressed")
depth_data = topic_data[0]["decode_data"]
point_cloud = bag.convert_depth_to_point_cloud(
    depth_data=depth_data,
    min_depth=0.2,
    max_depth=2.0,
    pixel_stride=1,
    stereo_calibration_topic="/robot0/sensor/depth/stereo_calibration",
)
print(point_cloud)
"""
[
    [N, 3]: np.ndarray, xyz in camera coordinate
    ...
]
"""
```

More details:

- `convert_depth_to_point_cloud(...)` takes a single depth frame as input.
- Intrinsics `K` and image size are read automatically from `StereoCalibration`.
- If depth resolution differs from calibration resolution, depth will be resized with nearest interpolation before projection.
- `min_depth`, `max_depth`, and `pixel_stride` are used to filter/subsample points.

### camera2 depth sensor data

Related topics:

```shell
# PredZ+zstd encoded depth stream (float16)
/robot0/sensor/camera2/depth
# depth codec metadata (Double Sphere intrinsics)
/robot0/sensor/camera2/depth_info
```

How to read

```python
bag = McapLoader(mcap_file)
topic_data = bag.get_topic_data("/robot0/sensor/camera2/depth")
print(topic_data[0]["decode_data"])
"""
[H, W, 1]: np.ndarray, depth map in meters (float32)
"""
```

Notes:

- `topic_data` is a list, each item is one frame dict.
- Use `topic_data[i]["decode_data"]` to access the i-th depth frame.
- `decode_data` shape is `[H, W, 1]`, dtype is `float32`, and the unit is meter.
- Decoding requires `/robot0/sensor/camera2/depth_info` for image size metadata.

How to convert camera2 depth to point cloud in camera coordinate

```python
bag = McapLoader(mcap_file)
topic_data = bag.get_topic_data("/robot0/sensor/camera2/depth")
depth_data = topic_data[0]["decode_data"]
point_cloud = bag.convert_camera2_depth_to_point_cloud(
    depth_data=depth_data,
    min_ray_z=0.1,
    max_range=5.0,
    pixel_stride=1,
    depth_info_topic="/robot0/sensor/camera2/depth_info",
)
print(point_cloud)
"""
[N, 3]: np.ndarray, xyz in camera coordinate
"""
```

More details:

- `convert_camera2_depth_to_point_cloud(...)` takes a single depth frame as input.
- Intrinsics `K`, `xi`, and `alpha` are read automatically from `DepthInfo`.
- Uses Double Sphere camera model instead of pinhole projection.
- If depth resolution differs from `DepthInfo` size, depth will be resized with nearest interpolation before projection.
- `min_ray_z`: keep pixels whose Double Sphere ray direction has `ray.z > min_ray_z`. This filters unstable edge pixels whose viewing direction is nearly parallel to the image plane.
- `max_range`: keep 3D points whose distance to the camera origin is `<= max_range` (meters). Set `<= 0` to disable the upper bound.
- `pixel_stride` is used to subsample points when generating the point cloud.

### imu

Related topics:

```shell
/robot0/sensor/imu
```

How to read

```python
bag = McapLoader(mcap_file)
topic_data = bag.get_topic_data("/robot0/sensor/imu")
print(topic_data["decode_data"])
"""
[
    [6, ]: np.ndarray, (AngularVel_X, AngularVel_Y, AngularVel_Z, LinearAcc_X, LinearAcc_Y, LinearAcc_Z)
    ...
]
"""
```

### tactile sensor data

Related topics:

```shell
/robot0/sensor/tactile_left
/robot0/sensor/tactile_right
```

How to read

```python
bag = McapLoader(mcap_file)
topic_data = bag.get_topic_data("/robot0/sensor/tactile_left")
print(topic_data["decode_data"])
"""
[
    [N, ]: np.ndarray, 
    ...
]
"""
```

### vio pose

Related topics:

```shell
/robot0/vio/eef_pose
```

How to read

```python
bag = McapLoader(mcap_file)
topic_data = bag.get_topic_data("/robot0/vio/eef_pose")
print(topic_data["decode_data"])
"""
[
    [7, ]: np.ndarray, [Pos_X, Pos_Y, Pos_Z, Q_X, Q_Y, Q_Z, Q_W], The first three values are translations, and the last four values are quaternions
    ...
]
"""
```

### magnetic encoder data

Related topics:

```shell
/robot0/sensor/magnetic_encoder
```

How to read

```python
bag = McapLoader(mcap_file)
topic_data = bag.get_topic_data("/robot0/sensor/magnetic_encoder")
print(topic_data["decode_data"])
"""
[
    [1, ]: np.ndarray, measurement values of magnetic encoder
    ...
]
"""
```

## H5

Each HDF5 file corresponds to a single episode and encapsulates both observational data and actions. Below is the hierarchical structure of the HDF5 file:

```shell
xxx.h5
├── observations/
│   ├── cameras/
│   │   └── <camera_name_x> (Dataset)
│   ├── tactile/
│   │   └── <left or right> (Dataset)
│   ├── eef_pos (Dataset)
│   └── imu (Dataset)
└── action (Dataset) (actions mirror the eef_pos data)
```

Groups and Datasets:

observations/

- cameras/
  - Description: Image data from camera.
  - Datasets:
    - Type: Dataset
    - Shape: (num_frames, height=xxx, width=xxx, channels=3) for mid fisheye camera, (num_frames, height=xxx, width=xxx, channels=3) for side wide camera
    - Data Type: uint8
    - Compression: gzip with compression level 4.
- tactile/
  - Description: Pressure data from tactile sensor.
  - Datasets:
    - Type: Dataset.
    - Shape: (num_frames, NEED-TO-BE-CONFIRM), row=12, col=8 
    - Data Type: float32
    - Compression: gzip with compression level 4.
- eef_pos/
  - Type: Dataset.
  - Shape: (num_frames, 8)
  - Data Type: float32
  - Description: Position and orientation data for each timestep. We obtain high-precision positioning information based on SLAM technology.
  - Columns: [Pos_X, Pos_Y, Pos_Z, Q_X, Q_Y, Q_Z, Q_W, Gripper_width]
  - Compression: gzip with compression level 4.
- imu/
  - Type: Dataset.
  - Shape: (num_frames, 6)
  - Data Type: float32
  - Description: Angular Velocity and Linear Acceleration data from IMU sensor for each timestep. We align IMU and image data based on timestamp.
  - Columns: [AngularVel_X, AngularVel_Y, AngularVel_Z, LinearAcc_X, LinearAcc_Y, LinearAcc_Z]
  - Compression: gzip with compression level 4.

action/

- Type: Dataset
- Shape: (num_frames, 8)
- Data Type: float32
- Description: Stores action data corresponding to each timestep. Same to eef_pos.
- Columns: [Pos_X, Pos_Y, Pos_Z, Q_X, Q_Y, Q_Z, Q_W, Gripper_width]
- Compression: gzip with compression level 4.

# 📖 Tutorials

## Installation

```shell
pip install -r requirements.txt
```

`huecodec` is installed from GitHub source (`cheind/hue-depth-encoding`), not from PyPI.

## Quick Start

### 1. decode mcap file demo

```shell
python mcap_decoder.py YOUR_MCAP_FILE_PATH
```

The purpose of the script is to parse the required topic data from the MCAP, the core code is as follows:

```python
# decode images
camera0_img_data = bag.get_topic_data("/robot0/sensor/camera0/compressed")
if camera0_img_data is not None:
    for d in camera0_img_data:
        single_frame_img = dict(
            data=d["decode_data"], # [h, w, c], bgr
            timestamp=d["data"].header.timestamp
        )
# decode vio pose
vio_pose_data = bag.get_topic_data("/robot0/vio/eef_pose")
if vio_pose_data is not None:
    for d in vio_pose_data:
        single_frame_pose = dict(
            data=d["decode_data"], # [Pos_X, Pos_Y, Pos_Z, Q_X, Q_Y, Q_Z, Q_W], detailed information can be found in README.md
            timestamp=d["data"].header.timestamp
        )
```

The decoded data of each topic is stored in the `decode_data` field, please refer to the section for details [Mcap](#mcap). After executing the script, the h264 video in the specified camera topic will be decoded into images and then saved as an mp4 file.

### 2. convert mcap to h5

`By default, H5 files only store mid fisheye camera data, vio pose, and action`

```shell
# Generate the H5 file with the same name in the MCAP file directory
python mcap_to_h5.py --mcap-file YOUR_MCAP_FILE_PATH

# Means to resize the img proportionally to a new width(640)
python mcap_to_h5.py --mcap-file YOUR_MCAP_FILE_PATH --img-new-width 640

# Generate the H5 file in the given directory
python mcap_to_h5.py --mcap-file YOUR_MCAP_FILE_PATH --out_path H5_FILE_PATH

# Generate the H5 file in the matrix studio task directory, all h5 files will be saved in the TASK_DIR/h5
python mcap_to_h5.py --task-dir TASK_DIR_IN_MATRIX_STUDIO

# Generate the H5 file in the matrix studio task directory, skip the generated h5 file in the task dir
python mcap_to_h5.py --task-dir TASK_DIR_IN_MATRIX_STUDIO --resume

# enable export more sensor data to h5 file
# python mcap_to_h5.py --help for more details
python mcap_to_h5.py --mcap-file YOUR_MCAP_FILE_PATH --imu --stereo-camera --tactile 
```

