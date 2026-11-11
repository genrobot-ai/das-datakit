人体和手部处理结果 MCAP 转六路同步反投影 MP4

入口
  mcap_to_mp4.py       单文件或递归批量转换，默认每个 MCAP 输出一个六路同步视频。
  requirements.txt    独立的 Python 依赖，不依赖模型权重或仓库 pb2 生成文件。

安装与运行
  要求 Python 3.10+。依赖均通过公开 PyPI 安装，使用 CPU。
  PyAV 负责视频编解码和元数据读取；imageio-ffmpeg 提供 FFmpeg 可执行文件，
  用于音频封装和完整解码校验。常规安装无需另行安装 ffmpeg 或 ffprobe。
  以下命令在仓库根目录执行：

  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r scripts/bodytrack/requirements.txt

  # 默认输出一个视频，内含 camera0..camera5 的同步六宫格画面
  python scripts/bodytrack/mcap_to_mp4.py \
    /path/to/processed.mcap --output-dir /path/to/mp4

  # 单相机功能保留：只导出 camera2
  python scripts/bodytrack/mcap_to_mp4.py \
    /path/to/processed.mcap --output-dir /path/to/mp4 --cameras 2

  # 需要六个独立视频时，显式使用 separate 布局
  python scripts/bodytrack/mcap_to_mp4.py \
    /path/to/processed.mcap --output-dir /path/to/mp4 --layout separate

  # 只在同一视频中展示指定的三路相机
  python scripts/bodytrack/mcap_to_mp4.py \
    /path/to/processed.mcap --output-dir /path/to/mp4 --cameras 0 2 5

  # 递归处理目录，并保留原相对分组目录
  python scripts/bodytrack/mcap_to_mp4.py \
    /path/to/processed_results --output-dir /path/to/mp4 \
    --pattern '*infer_result.mcap'

  # 只输出视频、调整宽度或编码质量
  python scripts/bodytrack/mcap_to_mp4.py \
    /path/to/processed.mcap --output-dir /path/to/mp4 \
    --no-audio --width 960 --crf 20

  python scripts/bodytrack/mcap_to_mp4.py --help

  Debian/Ubuntu 创建虚拟环境若提示 ensurepip 不可用，可先安装 python3-venv：
    sudo apt-get install python3-venv
  Windows 可用 py -m venv .venv 创建环境、.venv\Scripts\activate 激活环境。
  若某个平台没有 imageio-ffmpeg 随包提供的 FFmpeg，可安装该平台的 FFmpeg，
  或通过 IMAGEIO_FFMPEG_EXE 环境变量指定公开 FFmpeg 可执行文件的路径。

输出效果
  camera0..camera5 各自的原始畸变画面，分别叠加融合平滑后的 BODY22 人体
  骨架与左右手 21 点骨架。每路使用自身内参、畸变模型和外参进行反投影。
  默认仅输出一个 MP4，三列两行排列，顺序为：
    camera0 | camera1 | camera2
    camera3 | camera4 | camera5
  每格标明 Camera 编号和录制时间。音频只保留一份，与合成视频起点对齐。
  左侧绿色、右侧橙色，画面顶部显示相对录制时间。
  默认合成视频总宽 1920；六路标定均为 1600 x 1300 时，每格 640 x 520，
  最终视频为 1920 x 1040。每路保持自身宽高比，尺寸不同的画面补黑边。
  单相机和 --layout separate 模式默认宽 1280，标定 1600 x 1300 时为 1280 x 1040。
  视频 H.264 / yuv420p，音频 AAC，MP4 开启 faststart。
  合成模式按 MCAP log_time 对齐，使用 camera2 的原始时间轴；未选 camera2
  时使用 --cameras 中第一路的时间轴。每格按时间查找最近的实际相机帧，
  不按帧序号直接拼接，不强制重采样成固定 30 fps。
  默认相机间最大允许偏差 20 ms；视频首尾取各路共同覆盖的时间范围，
  容差内的首尾时差允许保留。中途没有可匹配相机帧时，该格显示黑底
  NO SYNC FRAME，不长时间冻结旧画面。JSON 记录每路最大匹配偏差、重复使用、
  未使用及缺失帧数，以及从参考时间轴裁掉的帧数。
  单相机/逐路模式保留对应相机的全部帧和原始时间轴。
  时间戳写入 MP4 时四舍五入到微秒；最大量化误差 0.5 微秒。

MCAP 输入要求
  每个选中相机 N（默认 0..5）的必需 topic：
    /robot0/sensor/cameraN/compressed
      foxglove.CompressedImage，format=h264，Annex B 视频包。
    /robot0/sensor/cameraN/camera_info
      foxglove.CameraCalibration，包含 K、D、T_b_c、width、height。

  人体反投影的共同参考：
    /robot0/sensor/camera2/camera_info
      camera2 到刚性相机机架的外参，用于将原生人体点转换到其他相机。
    robotx/motioncapture/human_pose_fuse_smooth_concat_cam2
      foxglove.PosesInFrame，frame_id=camera2，poses 固定 22 项。
      注意这个 topic 开头没有 /。

  手部 topic（缺少时在报告中警告，仅绘制现有手部）：
    /robot0/handtracking/left
    /robot0/handtracking/right
      foxglove.HandPointFrameInfo，bone_data 为 21 个世界坐标关节点。
      手部世界坐标需要 /robot0/vio/eef_pose，以及相机 T_b_c。
      支持 mano_params.coordinate_frame=world_eef_pose / world。

骨架与投影约定
  人体读取原生 _cam2 topic，不重新计算人体姿态。六路相机属于同一刚性机架：
    cameraN_from_camera2 = inverse(T_b_cN) @ T_b_c2
  将 BODY22 点完整地旋转、平移到 cameraN 坐标，再使用 cameraN 的内参与
  畸变模型投影；camera2 的人体坐标保持原样。
  手部 world_from_cameraN = VIO_pose @ R_BODY @ T_b_cN，逆变换后投影。
  R_BODY 将 (x,y,z) 映射为 (-z,x,-y)，与 DAS handtracking 的坐标约定一致。
  相机支持 ds（D 长度 6）和 ds-radtan（D 长度 11 / 12）。
  ds-radtan 的 xi、alpha 取 D[4:6]，11 维版本补齐保留的 k4=0。
  使用 Double Sphere 有效域，允许其视场内部分 z<0 的点，不使用针孔 z>0 筛选。
  默认人体/手部与相机最大匹配差 1 ms；VIO 最大匹配差 100 ms。
  超出容差的姿态不叠加，统计写入 JSON，不冻结上一帧的骨架。
  原生 hand writer 有些版本不填写 is_valid；脚本同时检查 MANO 参数、
  非零且有限的 21 点坐标，跳过“无 MANO 参数 + 全零关节点”的无效手帧。
  同时间戳重复的人体消息仅在可视化时取最后一条，输入 MCAP 保持不变。

输出文件
  默认六宫格：
    <output-dir>/<原相对分组>/<MCAP文件名去扩展名>_6cam_body_hand.mp4
    <output-dir>/<原相对分组>/<MCAP文件名去扩展名>_6cam_body_hand.json
    每个 MCAP 只输出一段视频，六路直接解码、叠加、同步合成，不生成六段中间视频。
  单相机或 --layout separate：
    <output-dir>/<原相对分组>/<MCAP文件名去扩展名>_camN_body_hand.mp4
    <output-dir>/<原相对分组>/<MCAP文件名去扩展名>_camN_body_hand.json
  自定义多路合成的文件名包括所选编号，例如 --cameras 0 2 5：
    <MCAP文件名去扩展名>_cam0_cam2_cam5_grid_body_hand.mp4
  <output-dir>/mp4_report.json      汇总、逐视频状态与失败原因。
  <output-dir>/mp4_manifest.csv     UTF-8 BOM 清单，包含布局、相机编号、时长、帧数等。

  每段 MP4 在交付前会核对编码、分辨率、帧数、时长、全部显示时间戳，
  并通过 FFmpeg 完整解码视频和音频；JSON 记录输入/输出 SHA-256、骨架匹配
  统计、相机编号、相机间变换矩阵、坐标转换误差及音频对齐信息。
  临时 WAV、中间 MP4 自动清理，只在完整验证后将最终视频放到目标路径。
  输入文件始终只读；不上传文件，不删除输入或其他已有结果。
  默认拒绝覆盖已有 MP4/逐视频 JSON；明确指定 --overwrite 才会替换。
  合成模式缺少所选相机时，该输入文件明确失败并继续其他文件；逐路模式某路
  失败后继续其他相机和文件。清单记录失败原因，有失败时最终退出码为 1。
  缺少某一路视频或标定会明确报错，不会用其他相机画面代替。

常用参数
  --pattern GLOB              目录输入的文件名过滤，默认 *.mcap。
  --layout grid|separate      默认 grid，一个同步合成视频；separate 为逐路视频。
  --cameras N [N ...]          相机编号 0..5，默认全部六路；只指定一路时输出单相机视频。
  --width N                   合成视频总宽（默认 1920），或单路视频宽（默认 1280）。
                              偶数，每列至少 320 像素；六宫格至少 960。
  --crf N                     0..51，默认 20，越小画质越高。
  --threads N                 编码线程，默认 4。
  --sync-tolerance-ms N        人体/手部与画面的匹配容差，默认 1。
  --camera-sync-tolerance-ms N 合成视频中各路相机与参考时间的最大偏差，默认 20。
  --vio-tolerance-ms N         VIO 匹配容差，默认 100。
  --no-audio                  仅输出视频。
  --overwrite                 允许替换对应的已有视频及 JSON。
