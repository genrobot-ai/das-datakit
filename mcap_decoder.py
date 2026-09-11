import argparse
import cv2
from pathlib import Path
from utils.mcaploader import McapLoader

def parse_mcap(mcap_file, output_dir=None, camera_topic="/robot0/sensor/camera0/compressed"):
    bag = McapLoader(mcap_file)
    print("Topics:")
    for topic in bag.all_topic_names:
        print(f"  {topic}: {bag.topic_schemas.get(topic)} ({bag.topic_statistics.channel_message_counts.get(next((c.id for c in bag._mcap_reader.get_summary().channels.values() if c.topic == topic), 0), 0)} messages)")
    # Because loading all topics requires a significant amount of memory, it is recommended to specify the list of specific topics to be loaded as needed.
    # bag.load_topics(bag.all_topic_names, auto_sync=False)

    # decode images
    camera0_img_data = bag.get_topic_data(camera_topic)
    if camera0_img_data is not None:
        video_writer = None
        fps = 30
        out_dir = Path(output_dir or Path(mcap_file).parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = str(out_dir / (Path(mcap_file).stem + "_camera0.mp4"))
        print(f"Find {len(camera0_img_data)} camera0 images, start to save video...")
        for d in camera0_img_data:
            single_frame_img = dict(
                data=d["decode_data"],  # [h, w, c], bgr
                timestamp=d["data"].header.timestamp,
            )
            if video_writer is None:
                h, w, c = single_frame_img["data"].shape
                h = max(1, h // 6)
                w = max(1, w // 6)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

            img = cv2.resize(single_frame_img["data"], (w, h))
            video_writer.write(img)

        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {output_video_path}")
    else:
        print("No camera0 images found")

    # decode imu data
    vio_pose_data = bag.get_topic_data("/robot0/vio/eef_pose")
    if vio_pose_data is not None:
        print(f"Find {len(vio_pose_data)} vio pose data")
        for d in vio_pose_data:
            single_frame_pose = dict(
                data=d[
                    "decode_data"
                ],  # [Pos_X, Pos_Y, Pos_Z, Q_X, Q_Y, Q_Z, Q_W], detailed information can be found in README.md
                timestamp=d["data"].header.timestamp,
            )
    else:
        print("No vio pose data found")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DAS mcap to h5 file")
    parser.add_argument("mcap_file", type=str, default="", help="input mcap file path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--camera-topic", default="/robot0/sensor/camera0/compressed")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    mcap_file = args.mcap_file
    parse_mcap(mcap_file, output_dir=args.output_dir, camera_topic=args.camera_topic)
