#!/usr/bin/env python3
"""Export a synchronized six-camera BODY22/hand projection grid, or individual camera videos."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import wave
from collections import Counter
from contextlib import ExitStack, closing
from fractions import Fraction
from pathlib import Path

BODY = "robotx/motioncapture/human_pose_fuse_smooth_concat"
BODY_CAM2 = BODY + "_cam2"
CAMERA = "/robot0/sensor/camera2/compressed"
CALIBRATION = "/robot0/sensor/camera2/camera_info"
VIO = "/robot0/vio/eef_pose"
AUDIO = "/robot0/sensor/audio"
HANDS = ("/robot0/handtracking/left", "/robot0/handtracking/right")
BODY_EDGES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11), (3, 6), (6, 9), (9, 12),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21), (12, 15),
]
HAND_EDGES = [(0, i) for i in (1, 5, 9, 13, 17)] + [
    (i + j, i + j + 1) for i in (1, 5, 9, 13, 17) for j in range(3)
]
LEFT = {1, 4, 7, 10, 13, 16, 18, 20}
RIGHT = {2, 5, 8, 11, 14, 17, 19, 21}
GREEN, ORANGE, YELLOW = (70, 235, 90), (35, 150, 255), (50, 230, 245)
US, NS = Fraction(1, 1_000_000), Fraction(1, 1_000_000_000)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed: {result.stderr.strip()[-4000:]}")
    return result.stdout


def iter_decoded(path, topics):
    # NonSeekingReader also handles MCAPs without chunk/message indexes. Decode
    # selected topics in physical order; never buffer the entire video for sorting.
    with path.open("rb") as stream:
        reader = NonSeekingReader(stream, decoder_factories=[DecoderFactory()])
        yield from reader.iter_decoded_messages(topics=topics, log_time_order=False)


def xyz(point):
    return [point.x, point.y, point.z]


def transform(translation, quaternion):
    q = np.asarray(quaternion, dtype=float)
    require(np.isfinite(q).all() and np.linalg.norm(q) > 1e-12, "Invalid quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ]
    out[:3, 3] = translation
    require(np.isfinite(out).all(), "Invalid camera/VIO transform")
    return out


def nearest_time(times, timestamp, tolerance_ns):
    i = bisect.bisect_left(times, timestamp)
    candidates = [times[j] for j in (i - 1, i) if 0 <= j < len(times)]
    if not candidates:
        return None
    ts = min(candidates, key=lambda value: (abs(value - timestamp), value))
    return ts if abs(ts - timestamp) <= tolerance_ns else None


def calibration_key(obj):
    return (obj.width, obj.height, obj.distortion_model,
            tuple(obj.D), tuple(obj.K), tuple(obj.T_b_c))


def load_signals(path, args, camera_id=2):
    camera_topic = f"/robot0/sensor/camera{camera_id}/compressed"
    calibration_topic = f"/robot0/sensor/camera{camera_id}/camera_info"
    records = {topic: {} for topic in (BODY, BODY_CAM2, VIO, *HANDS)}
    counts = Counter()
    timestamps, audio = [], []
    calibrations = {}
    topics = list(dict.fromkeys([CALIBRATION, calibration_topic, camera_topic, *records]))
    if not args.no_audio:
        topics.append(AUDIO)
    for _, channel, message, obj in iter_decoded(path, topics):
        topic, ts = channel.topic, int(message.log_time)
        counts[topic] += 1
        if topic == camera_topic:
            require(obj.format.lower() == "h264", f"Unsupported video format: {obj.format}")
            timestamps.append(ts)
        elif topic in (CALIBRATION, calibration_topic):
            previous = calibrations.get(topic)
            require(previous is None or calibration_key(previous) == calibration_key(obj),
                    f"Camera calibration changes during the recording: {topic}")
            calibrations[topic] = obj
        elif topic == AUDIO:
            config = obj.config
            require(config.format == "PCM_16" and config.bit_depth == 16,
                    f"Unsupported audio format {config.format}; use --no-audio for video only")
            require(config.sample_rate > 0 and config.channels > 0, "Invalid audio configuration")
            require(len(obj.audio_data) % (2 * config.channels) == 0, "Incomplete PCM audio sample")
            audio.append((ts, bytes(obj.audio_data), config.sample_rate, config.channels))
        elif topic == VIO:
            p, q = obj.pose.position, obj.pose.orientation
            records[topic][ts] = transform(xyz(p), [q.x, q.y, q.z, q.w])
        elif topic in HANDS:
            bones = list(obj.bone_data)
            if len(bones) == 21 and {b.id for b in bones} == set(range(21)):
                bones.sort(key=lambda bone: bone.id)
            points = np.array([xyz(b.to_global.position) for b in bones])
            mano = getattr(obj, "mano_params", None)
            has_mano = mano is not None and len(mano.global_transl) == 3
            # The native hand writer omits is_valid even for valid hands. Its
            # invalid frames have all-zero joints and no MANO parameters.
            valid = (points.shape == (21, 3) and np.isfinite(points).all()
                     and np.any(points) and (has_mano or getattr(obj, "is_valid", False)))
            if valid and has_mano:
                require(mano.coordinate_frame in ("world_eef_pose", "world"),
                        f"Unsupported hand coordinate frame: {mano.coordinate_frame}")
            records[topic][ts] = points if valid else None
        else:
            require(len(obj.poses) == 22, f"{topic}: expected BODY22, got {len(obj.poses)} joints")
            if topic == BODY_CAM2:
                require(obj.frame_id == "camera2", f"Unexpected body coordinate frame: {obj.frame_id}")
            points = np.array([xyz(p.position) for p in obj.poses])
            require(np.isfinite(points).all(), f"{topic}: non-finite joint positions")
            records[topic][ts] = points
    require(timestamps, f"Missing/empty video topic: {camera_topic}")
    for topic in dict.fromkeys((CALIBRATION, calibration_topic)):
        require(topic in calibrations, f"Missing camera calibration: {topic}")
        calibration = calibrations[topic]
        require(calibration.width > 0 and calibration.height > 0
                and len(calibration.K) == 9 and len(calibration.T_b_c) == 7,
                f"Incomplete camera calibration: {topic}")
        # Validate both cameras before starting an encoder.
        project(np.array([[0., 0., 1.]]), calibration)
    calib = calibrations[calibration_topic]
    require(records[BODY_CAM2], f"Missing/empty processed body topic: {BODY_CAM2}")
    require(all(b > a for a, b in zip(timestamps, timestamps[1:])),
            "Camera timestamps must strictly increase in MCAP packet order")
    vio_times = sorted(records[VIO])
    body_from_camera = transform(calib.T_b_c[:3], calib.T_b_c[3:])
    reference_calib = calibrations[CALIBRATION]
    body_from_camera2 = transform(reference_calib.T_b_c[:3], reference_calib.T_b_c[3:])
    # All six cameras are rigidly attached to the same body. The native cam2
    # joints already include the processed body pose and concatenated wrists.
    # Move those points into the target camera before applying its intrinsics.
    camera_from_camera2 = (np.eye(4) if camera_id == 2
                           else np.linalg.inv(body_from_camera) @ body_from_camera2)
    rbody = np.array([[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
    valid_hands = sum(points is not None for topic in HANDS for points in records[topic].values())
    require(not valid_hands or vio_times, f"Hand world coordinates require VIO topic: {VIO}")
    transforms = {}
    for ts in set().union(*(records[t] for t in (BODY, BODY_CAM2, *HANDS))):
        vio_ts = nearest_time(vio_times, ts, round(args.vio_tolerance_ms * 1e6))
        if vio_ts is not None:
            transforms[ts] = np.linalg.inv(records[VIO][vio_ts] @ rbody @ body_from_camera)
    errors = []
    for ts, points in records[BODY_CAM2].items():
        if ts in records[BODY] and ts in transforms:
            matrix = transforms[ts]
            projected = records[BODY][ts] @ matrix[:3, :3].T + matrix[:3, 3]
            from_camera2 = points @ camera_from_camera2[:3, :3].T + camera_from_camera2[:3, 3]
            errors.append(float(np.max(np.abs(projected - from_camera2))))
    if errors:
        require(max(errors) < 1e-5, f"World/camera{camera_id} body transform mismatch: {max(errors):.6g} m")
    warnings = [f"Missing hand topic; that hand will not be drawn: {t}" for t in HANDS if not records[t]]
    if not args.no_audio and not audio:
        warnings.append("No source audio topic; exporting video only")
    return dict(records=records, times={t: sorted(r) for t, r in records.items()},
                camera_id=camera_id, camera_topic=camera_topic, calibration_topic=calibration_topic,
                timestamps=timestamps, calib=calib, camera_from_camera2=camera_from_camera2,
                transforms=transforms, audio=audio,
                roundtrip_max_error_m=max(errors) if errors else None,
                roundtrip_frames=len(errors), warnings=warnings,
                duplicate_body_messages=counts[BODY_CAM2] - len(records[BODY_CAM2]))


def project(points, calib):
    """Project onto the original distorted DS / DS-radtan camera image."""
    x, y, z = points.T
    d = np.asarray(calib.D)
    if calib.distortion_model == "ds" and len(d) == 6:
        xi, alpha = d[4:6]
        radtan = None
    elif calib.distortion_model == "ds-radtan" and len(d) in (11, 12):
        # xi/alpha remain D[4:6]; the final two coefficients are not xi/alpha.
        xi, alpha = d[4:6]
        radtan = np.r_[d[6:9], 0., d[9:11]] if len(d) == 11 else d[6:12]
    else:
        raise ValueError(f"Unsupported camera model: {calib.distortion_model}, D length={len(d)}")
    require(np.isfinite(d).all() and np.isfinite(calib.K).all() and 0 < alpha < 1,
            "Invalid DS camera parameters")
    distance = np.linalg.norm(points, axis=1)
    zx = z + xi * distance
    denominator = alpha * np.sqrt(x*x + y*y + zx*zx) + (1 - alpha) * zx
    w1 = (1-alpha)/alpha if alpha > .5 else alpha/(1-alpha)
    w2 = (w1 + xi)/np.sqrt(2*w1*xi + xi*xi + 1)
    # A DS fisheye can see some points with z<0: use its actual validity domain.
    valid = (denominator > 1e-8) & (z > -w2 * distance)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mx, my = x/denominator, y/denominator
        if radtan is not None:
            k1, k2, k3, k4, p1, p2 = radtan
            rho2 = np.minimum(mx*mx + my*my, 1e8)
            radial = rho2 * (k1 + rho2 * (k2 + rho2 * (k3 + rho2 * k4)))
            mx, my = (mx*(1+radial) + 2*p1*mx*my + p2*(rho2 + 2*mx*mx),
                      my*(1+radial) + p1*(rho2 + 2*my*my) + 2*p2*mx*my)
        k = np.asarray(calib.K).reshape(3, 3)
        uv = np.column_stack([k[0, 0]*mx + k[0, 2], k[1, 1]*my + k[1, 2]])
    valid &= np.isfinite(uv).all(axis=1)
    return uv, valid


def draw_points(image, points, calib, edges, side=None):
    uv, valid = project(points, calib)
    height, width = image.shape[:2]
    uv *= [width/calib.width, height/calib.height]
    valid &= (np.abs(uv) < 1e6).all(axis=1)
    coords = np.round(np.nan_to_num(np.clip(uv, -1e6, 1e6))).astype(int)
    for a, b in edges:
        if valid[a] and valid[b]:
            color = ((GREEN if side == 0 else ORANGE) if side is not None
                     else GREEN if b in LEFT else ORANGE if b in RIGHT else YELLOW)
            ok, p, q = cv2.clipLine((0, 0, width, height), tuple(coords[a]), tuple(coords[b]))
            if ok:
                cv2.line(image, p, q, (12, 12, 12), 5 if side is None else 4, cv2.LINE_AA)
                cv2.line(image, p, q, color, 3 if side is None else 2, cv2.LINE_AA)
    visible = (valid & (uv[:, 0] >= 0) & (uv[:, 0] < width)
               & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    for x, y in coords[visible]:
        cv2.circle(image, (int(x), int(y)), 3 if side is None else 2, (245, 245, 245), -1, cv2.LINE_AA)
    return int(visible.sum())


def overlay(image, timestamp, first_timestamp, data, stats, tolerance_ns):
    records, times = data["records"], data["times"]
    ts = nearest_time(times[BODY_CAM2], timestamp, tolerance_ns)
    if ts is not None:
        stats["body_frames"] += 1
        stats["max_body_alignment_ns"] = max(stats["max_body_alignment_ns"], abs(ts-timestamp))
        matrix = data["camera_from_camera2"]
        points = records[BODY_CAM2][ts] @ matrix[:3, :3].T + matrix[:3, 3]
        stats["visible_body_joints"] += draw_points(image, points, data["calib"], BODY_EDGES)
    for side, topic in enumerate(HANDS):
        ts = nearest_time(times[topic], timestamp, tolerance_ns)
        if ts is None or records[topic][ts] is None:
            continue
        matrix = data["transforms"].get(ts)
        if matrix is None:
            stats["hand_frames_without_vio"] += 1
            continue
        points = records[topic][ts] @ matrix[:3, :3].T + matrix[:3, 3]
        stats[f"{('left', 'right')[side]}_hand_frames"] += 1
        stats["max_hand_alignment_ns"] = max(stats["max_hand_alignment_ns"], abs(ts-timestamp))
        stats["visible_hand_joints"] += draw_points(image, points, data["calib"], HAND_EDGES, side)
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (18, 18, 18), -1)
    elapsed = (timestamp-first_timestamp)/1e9
    cv2.putText(image, f"Camera {data['camera_id']}  |  {elapsed:05.2f}s", (16, 29),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(image, "LEFT", (image.shape[1]-190, 29), cv2.FONT_HERSHEY_SIMPLEX, .58, GREEN, 2, cv2.LINE_AA)
    cv2.putText(image, "RIGHT", (image.shape[1]-100, 29), cv2.FONT_HERSHEY_SIMPLEX, .58, ORANGE, 2, cv2.LINE_AA)


def decoded_frames(path, threads, camera_topic=CAMERA):
    codec = av.CodecContext.create("h264", "r")
    codec.thread_count = min(threads, 2)
    for _, _, message, obj in iter_decoded(path, [camera_topic]):
        packet = av.Packet(obj.data)
        packet.pts = packet.dts = int(message.log_time)
        packet.time_base = NS
        yield from codec.decode(packet)
    yield from codec.decode(None)


def write_audio(path, chunks, first_ts, duration_ns):
    rate, channels = chunks[0][2:]
    length = (duration_ns * rate + 999_999_999)//1_000_000_000
    signal = np.zeros((length, channels), dtype="<i2")
    occupied = np.zeros(length, dtype=bool)
    for ts, payload, sr, nch in chunks:
        require((sr, nch) == (rate, channels), "Audio configuration changes during recording")
        pcm = np.frombuffer(payload, dtype="<i2").reshape(-1, channels)
        start = round((ts-first_ts)*rate/1e9)
        src, dst = max(0, -start), max(0, start)
        count = min(len(pcm)-src, length-dst)
        if count > 0:
            require(not occupied[dst:dst+count].any(), "Audio chunks overlap; check source or use --no-audio")
            signal[dst:dst+count] = pcm[src:src+count]
            occupied[dst:dst+count] = True
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(signal.tobytes())
    return dict(source_format="PCM_16", sample_rate=rate, channels=channels, chunks=len(chunks),
                uncovered_samples_padded_with_silence=int((~occupied).sum()),
                first_chunk_offset_seconds=(chunks[0][0]-first_ts)/1e9)


def verify_video(path, pts, width, height, duration_ns, has_audio, ffmpeg):
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, "Expected one output video stream")
        video = container.streams.video[0]
        codec = video.codec_context
        require(codec.name == "h264" and codec.pix_fmt == "yuv420p", "Unexpected output codec")
        require((codec.width, codec.height) == (width, height), "Incorrect output dimensions")
        require(video.frames == len(pts), "Output frame count mismatch")
        require(video.duration is not None and video.time_base is not None, "Missing output duration")
        duration = float(video.duration * video.time_base)
        require(abs(duration - duration_ns/1e9) < .005, "Output duration mismatch")
        audio = container.streams.audio
        require(len(audio) == int(has_audio), "Output audio stream mismatch")
        if audio:
            require(audio[0].codec_context.name == "aac", "Unexpected output audio codec")
        actual_pts = [round(p.pts*p.time_base*1_000_000) for p in container.demux(video=0) if p.pts is not None]
    require(actual_pts == pts, "Output presentation timestamps do not match the source")
    run([ffmpeg, "-v", "error", "-xerror", "-nostdin", "-threads", "2", "-i", str(path), "-f", "null", "-"])
    return duration


def video_timing(timestamps):
    interval = int(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 33_333_333
    duration_ns = timestamps[-1] - timestamps[0] + interval
    pts = [(ts-timestamps[0]+500)//1000 for ts in timestamps]
    require(all(b > a for a, b in zip(pts, pts[1:])), "Camera timestamps collide at MP4 microsecond precision")
    durations = {pts[i]: pts[i+1]-pts[i] if i+1 < len(pts) else round(interval/1000) for i in range(len(pts))}
    return interval, duration_ns, pts, durations


def encode_video(path, images, pts, durations, width, height, args):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.time_base = stream.codec_context.time_base = US
        stream.codec_context.thread_count = args.threads
        stream.codec_context.max_b_frames = 0
        stream.options = {"crf": str(args.crf), "preset": "fast", "tune": "zerolatency"}

        def mux(packets):
            for packet in packets:
                require(packet.pts in durations, f"Unexpected encoded packet timestamp: {packet.pts}")
                packet.duration = durations[packet.pts]
                container.mux(packet)

        count = 0
        for i, image in enumerate(images):
            require(i < len(pts), "Too many rendered frames")
            require(image.shape[:2] == (height, width), "Incorrect rendered image size")
            encoded = av.VideoFrame.from_ndarray(image, format="bgr24")
            encoded.pts, encoded.time_base = pts[i], US
            # Packet durations are authoritative, including on PyAV releases
            # where VideoFrame.duration cannot be assigned.
            mux(stream.encode(encoded))
            count += 1
            if count % 300 == 0:
                print(f"  frames {count}/{len(pts)}", flush=True)
        mux(stream.encode(None))
    require(count == len(pts), "Rendered frame count differs from the output timeline")
    return count


def finish_video(temporary, audio, first_ts, duration_ns, pts, width, height, ffmpeg):
    completed = temporary / "completed.mp4"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(temporary/"video.mp4")]
    audio_report = None
    if audio:
        audio_report = write_audio(temporary/"audio.wav", audio, first_ts, duration_ns)
        command += ["-i", str(temporary/"audio.wav"), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "96k"]
    command += ["-c:v", "copy", "-movflags", "+faststart", str(completed)]
    run(command)
    duration = verify_video(completed, pts, width, height, duration_ns, audio_report is not None, ffmpeg)
    return completed, duration, audio_report


def render(path, output, args, ffmpeg, camera_id=2):
    started = time.monotonic()
    initial = path.stat()
    input_hash = sha256(path)
    data = load_signals(path, args, camera_id)
    timestamps = data["timestamps"]
    interval, duration_ns, pts, durations = video_timing(timestamps)
    width = args.width
    height = max(2, round(data["calib"].height*width/data["calib"].width/2)*2)
    stats = Counter()
    for warning in data["warnings"]:
        print(f"  WARNING: {warning}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mcap-to-mp4-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        def images():
            count = 0
            for i, frame in enumerate(decoded_frames(path, args.threads, data["camera_topic"])):
                require(i < len(timestamps) and frame.pts == timestamps[i], "H264 decoded frame timestamp mismatch")
                require((frame.width, frame.height) == (data["calib"].width, data["calib"].height),
                        "Decoded camera image size differs from its calibration")
                image = cv2.resize(frame.to_ndarray(format="bgr24"), (width, height), interpolation=cv2.INTER_AREA)
                overlay(image, timestamps[i], timestamps[0], data, stats, round(args.sync_tolerance_ms*1e6))
                yield image
                count += 1
            require(count == len(timestamps), "Some source camera packets did not decode into frames")
        with closing(images()) as frames:
            count = encode_video(temporary/"video.mp4", frames, pts, durations, width, height, args)
        require(stats["body_frames"] > 0, "No body timestamps matched the camera frames")
        completed, duration, audio_report = finish_video(
            temporary, data["audio"], timestamps[0], duration_ns, pts, width, height, ffmpeg)
        current = path.stat()
        require((current.st_size, current.st_mtime_ns) == (initial.st_size, initial.st_mtime_ns),
                "Input MCAP changed while rendering")
        completed.replace(output)
    result = dict(status="ok", verified=True, input=str(path), input_sha256=input_hash,
                  output=str(output), bytes=output.stat().st_size, sha256=sha256(output),
                  mode=f"camera{camera_id}_body22_and_hand21_overlay", layout="separate", camera_id=camera_id,
                  camera_topic=data["camera_topic"], calibration_topic=data["calibration_topic"],
                  body_topic=BODY_CAM2, body_source_camera_id=2,
                  camera_from_camera2=data["camera_from_camera2"].tolist(),
                  hand_topics=list(HANDS), width=width, height=height, frames=count,
                  duration_seconds=duration, source_median_frame_interval_ns=interval,
                  native_timing_preserved=True, preserved_all_video_frames=True,
                  timestamp_max_quantization_us=.5, overlays=dict(stats), audio=audio_report,
                  roundtrip_max_error_m=data["roundtrip_max_error_m"], roundtrip_frames=data["roundtrip_frames"],
                  duplicate_body_messages_collapsed_for_display=data["duplicate_body_messages"],
                  warnings=data["warnings"], video_codec="h264", pixel_format="yuv420p",
                  audio_codec="aac" if audio_report else None, full_decode_verified=True,
                  intermediates_removed=True, render_wall_seconds=round(time.monotonic()-started, 2))
    write_json(output.with_suffix(".json"), result)
    return result


def synchronization_plan(cameras, tolerance_ns):
    """Use a reference camera's clock and nearest timestamp, never frame index."""
    reference_id = 2 if 2 in cameras else next(iter(cameras))
    first = max(data["timestamps"][0] for data in cameras.values())
    last = min(data["timestamps"][-1] for data in cameras.values())
    timestamps = [ts for ts in cameras[reference_id]["timestamps"]
                  if first-tolerance_ns <= ts <= last+tolerance_ns]
    require(timestamps, "Selected cameras have no overlapping time range within the camera sync tolerance")
    selected = {}
    for camera_id, data in cameras.items():
        selected[camera_id] = [nearest_time(data["timestamps"], ts, tolerance_ns) for ts in timestamps]
        require(any(ts is not None for ts in selected[camera_id]),
                f"No synchronized frames for camera{camera_id}; check timestamps or --camera-sync-tolerance-ms")
    return reference_id, timestamps, selected


def aligned_frames(frames, timestamps, selected, calib):
    """Keep one decoded frame in memory and validate even unused source frames."""
    iterator = iter(frames)
    count = 0
    current = None

    def checked(frame):
        nonlocal count
        require(frame is not None and count < len(timestamps) and frame.pts == timestamps[count],
                "H264 decoded frame timestamp mismatch")
        require((frame.width, frame.height) == (calib.width, calib.height),
                "Decoded camera image size differs from its calibration")
        count += 1
        return frame

    try:
        for timestamp in selected:
            if timestamp is None:
                yield None
                continue
            while current is None or current.pts < timestamp:
                current = checked(next(iterator, None))
            require(current.pts == timestamp, "Selected camera frame is unavailable")
            yield current
        for frame in iterator:
            checked(frame)
        require(count == len(timestamps), "Some source camera packets did not decode into frames")
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def render_grid(path, output, args, ffmpeg):
    started = time.monotonic()
    initial = path.stat()
    input_hash = sha256(path)
    cameras = {camera_id: load_signals(path, args, camera_id) for camera_id in args.cameras}
    tolerance_ns = round(args.camera_sync_tolerance_ms*1e6)
    reference_id, timestamps, selected = synchronization_plan(cameras, tolerance_ns)
    interval, duration_ns, pts, durations = video_timing(timestamps)
    columns = min(3, len(cameras))
    rows = (len(cameras)+columns-1)//columns
    tile_width = args.width//columns//2*2
    image_heights = {camera_id: max(2, round(data["calib"].height*tile_width/data["calib"].width/2)*2)
                     for camera_id, data in cameras.items()}
    tile_height = max(image_heights.values())
    width, height = args.width, rows*tile_height
    stats = {camera_id: Counter() for camera_id in cameras}
    warnings = [f"camera{camera_id}: {warning}" for camera_id, data in cameras.items()
                for warning in data["warnings"]]
    for camera_id in cameras:
        missing = sum(ts is None for ts in selected[camera_id])
        if missing:
            warnings.append(f"camera{camera_id}: {missing} grid frames have no synchronized image; showing black tiles")
    for warning in warnings:
        print(f"  WARNING: {warning}", flush=True)
    print(f"  grid {columns}x{rows}, {width}x{height}, reference camera{reference_id}, "
          f"{len(timestamps)} synchronized frames", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mcap-to-mp4-", dir=output.parent) as temporary:
        temporary = Path(temporary)

        def images():
            with ExitStack() as stack:
                streams = {camera_id: stack.enter_context(closing(aligned_frames(
                    decoded_frames(path, args.threads, data["camera_topic"]),
                    data["timestamps"], selected[camera_id], data["calib"])))
                    for camera_id, data in cameras.items()}
                for timestamp in timestamps:
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)
                    for index, (camera_id, data) in enumerate(cameras.items()):
                        frame = next(streams[camera_id])
                        row, column = divmod(index, columns)
                        x, y = column*tile_width, row*tile_height
                        if frame is None:
                            tile = canvas[y:y+tile_height, x:x+tile_width]
                            cv2.putText(tile, f"Camera {camera_id} | NO SYNC FRAME", (16, 29),
                                        cv2.FONT_HERSHEY_SIMPLEX, .65, (245, 245, 245), 1, cv2.LINE_AA)
                            continue
                        image_height = image_heights[camera_id]
                        image = cv2.resize(frame.to_ndarray(format="bgr24"), (tile_width, image_height),
                                           interpolation=cv2.INTER_AREA)
                        overlay(image, frame.pts, timestamps[0], data, stats[camera_id],
                                round(args.sync_tolerance_ms*1e6))
                        y += (tile_height-image_height)//2
                        canvas[y:y+image_height, x:x+tile_width] = image
                    yield canvas
                # The last yield does not exhaust a generator. Resume each one
                # explicitly to validate the cropped/unused tails as well.
                for stream in streams.values():
                    require(next(stream, None) is None, "Too many synchronized frames")

        with closing(images()) as frames:
            count = encode_video(temporary/"video.mp4", frames, pts, durations, width, height, args)
        for camera_id in cameras:
            require(stats[camera_id]["body_frames"] > 0,
                    f"No body timestamps matched camera{camera_id}")
        completed, duration, audio_report = finish_video(
            temporary, cameras[reference_id]["audio"], timestamps[0], duration_ns, pts, width, height, ffmpeg)
        current = path.stat()
        require((current.st_size, current.st_mtime_ns) == (initial.st_size, initial.st_mtime_ns),
                "Input MCAP changed while rendering")
        completed.replace(output)
    camera_reports = []
    for camera_id, data in cameras.items():
        matched = [ts for ts in selected[camera_id] if ts is not None]
        deltas = [abs(source_ts-ts) for ts, source_ts in zip(timestamps, selected[camera_id]) if source_ts is not None]
        camera_reports.append(dict(
            camera_id=camera_id, camera_topic=data["camera_topic"], calibration_topic=data["calibration_topic"],
            source_frames=len(data["timestamps"]), matched_grid_frames=len(matched),
            unmatched_grid_frames=count-len(matched), unique_source_frames_used=len(set(matched)),
            reused_grid_frames=len(matched)-len(set(matched)),
            unused_source_frames=len(data["timestamps"])-len(set(matched)),
            max_camera_alignment_ns=max(deltas), first_source_timestamp_ns=data["timestamps"][0],
            last_source_timestamp_ns=data["timestamps"][-1], overlays=dict(stats[camera_id]),
            camera_from_camera2=data["camera_from_camera2"].tolist(),
            roundtrip_max_error_m=data["roundtrip_max_error_m"], roundtrip_frames=data["roundtrip_frames"],
            duplicate_body_messages_collapsed_for_display=data["duplicate_body_messages"]))
    result = dict(
        status="ok", verified=True, input=str(path), input_sha256=input_hash,
        output=str(output), bytes=output.stat().st_size, sha256=sha256(output),
        mode="synchronized_multicamera_body22_and_hand21_grid", layout="grid", camera_ids=args.cameras,
        reference_camera_id=reference_id, grid_columns=columns, grid_rows=rows,
        tile_width=tile_width, tile_height=tile_height, width=width, height=height, frames=count,
        body_topic=BODY_CAM2, body_source_camera_id=2, hand_topics=list(HANDS), cameras=camera_reports,
        duration_seconds=duration, source_median_frame_interval_ns=interval,
        first_timestamp_ns=timestamps[0], last_timestamp_ns=timestamps[-1],
        synchronization="nearest_log_time_on_reference_camera_timeline",
        camera_sync_tolerance_ms=args.camera_sync_tolerance_ms,
        reference_frames_trimmed=len(cameras[reference_id]["timestamps"])-len(timestamps),
        native_timing_preserved=True,
        preserved_all_video_frames=all(item["unused_source_frames"] == 0 for item in camera_reports),
        timestamp_max_quantization_us=.5, audio=audio_report, warnings=warnings,
        video_codec="h264", pixel_format="yuv420p", audio_codec="aac" if audio_report else None,
        full_decode_verified=True, all_source_video_frames_decode_verified=True, intermediates_removed=True,
        render_wall_seconds=round(time.monotonic()-started, 2))
    write_json(output.with_suffix(".json"), result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=Path, help="Processed .mcap file, or a directory to scan recursively")
    parser.add_argument("-o", "--output-dir", required=True, type=Path, help="Output directory; preserve input subdirectories")
    parser.add_argument("--pattern", default="*.mcap", help="Filename glob when input is a directory")
    parser.add_argument("--layout", choices=("grid", "separate"), default="grid",
                        help="One synchronized grid MP4, or a separate MP4 for each selected camera")
    parser.add_argument("--cameras", nargs="+", type=int, choices=range(6), default=list(range(6)),
                        help="Camera IDs to show; one ID exports a single-camera MP4")
    parser.add_argument("--width", type=int,
                        help="Total grid width (default 1920), or single-camera width (default 1280)")
    parser.add_argument("--crf", type=int, default=20, help="H264 constant rate factor (smaller means higher quality)")
    parser.add_argument("--threads", type=int, default=4, help="CPU encoder threads")
    parser.add_argument("--sync-tolerance-ms", type=float, default=1., help="Maximum camera/body/hand timestamp mismatch")
    parser.add_argument("--camera-sync-tolerance-ms", type=float, default=20.,
                        help="Maximum difference between a grid timestamp and each camera image")
    parser.add_argument("--vio-tolerance-ms", type=float, default=100., help="Maximum VIO/hand timestamp mismatch")
    parser.add_argument("--no-audio", action="store_true", help="Export video only")
    parser.add_argument("--overwrite", action="store_true", help="Replace videos and per-video JSON already in the output directory")
    args = parser.parse_args(argv)
    if len(set(args.cameras)) != len(args.cameras):
        parser.error("--cameras must not contain duplicate IDs")
    if len(args.cameras) == 1:
        args.layout = "separate"
    grid = args.layout == "grid" and len(args.cameras) > 1
    if args.width is None:
        args.width = 1920 if grid else 1280
    minimum_width = 320*min(3, len(args.cameras)) if grid else 320
    if args.width < minimum_width or args.width % 2 or not 0 <= args.crf <= 51 or args.threads < 1:
        parser.error(f"--width must be even and >={minimum_width}; --crf must be 0..51; --threads must be >=1")
    if not (0 <= args.sync_tolerance_ms <= 1000 and 0 <= args.vio_tolerance_ms <= 1000
            and 0 <= args.camera_sync_tolerance_ms <= 1000):
        parser.error("Timestamp tolerances must be finite and within 0..1000 ms")
    args.input, args.output_dir = args.input.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if args.input.is_file():
        files, base = [args.input], args.input.parent
        if args.input.suffix.lower() != ".mcap":
            parser.error("Input file must have the .mcap extension")
    elif args.input.is_dir():
        files, base = sorted(p for p in args.input.rglob(args.pattern) if p.is_file() and p.suffix.lower() == ".mcap"), args.input
    else:
        parser.error(f"Input does not exist: {args.input}")
    if not files:
        parser.error("No input MCAP files matched")
    if grid:
        label = "6cam" if args.cameras == list(range(6)) else "_".join(f"cam{i}" for i in args.cameras)+"_grid"
        jobs = [(p, None, args.output_dir/p.relative_to(base).parent/(p.stem+f"_{label}_body_hand.mp4"))
                for p in files]
    else:
        jobs = [(p, camera_id, args.output_dir/p.relative_to(base).parent/
                 (p.stem+f"_cam{camera_id}_body_hand.mp4")) for p in files for camera_id in args.cameras]
    if not args.overwrite:
        conflicts = [str(out) for _, _, out in jobs if out.exists() or out.with_suffix(".json").exists()]
        if conflicts:
            parser.error(f"Output already exists: {conflicts[0]}; use another directory or --overwrite")
    return args, jobs


def main(argv=None):
    args, jobs = parse_args(argv)
    # Keep --help usable even on machines without the rendering dependencies.
    global av, cv2, np, NonSeekingReader, DecoderFactory
    try:
        import av
        import cv2
        import imageio_ffmpeg
        import numpy as np
        from mcap.reader import NonSeekingReader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as error:
        print(f"Missing dependency: {error}. Install scripts/bodytrack/requirements.txt", file=sys.stderr)
        return 2
    try:
        # The public imageio-ffmpeg wheel supplies its own executable, so users
        # do not need a system FFmpeg installation or a container image.
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as error:
        print(f"FFmpeg is unavailable: {error}", file=sys.stderr)
        return 2
    cv2.setNumThreads(min(args.threads, 2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (source, camera_id, output) in enumerate(jobs, 1):
        label = "synchronized camera grid" if camera_id is None else f"camera{camera_id}"
        print(f"[{i}/{len(jobs)}] {source} -> {label}", flush=True)
        try:
            result = (render_grid(source, output, args, ffmpeg) if camera_id is None
                      else render(source, output, args, ffmpeg, camera_id))
            print(f"  OK: {output} ({result['frames']} frames, {result['duration_seconds']:.3f}s)", flush=True)
        except Exception as error:
            result = dict(status="error", input=str(source), camera_id=camera_id,
                          layout=args.layout, camera_ids=args.cameras, output=str(output), error=str(error))
            print(f"  ERROR: {error}", file=sys.stderr, flush=True)
        results.append(result)
        # Preserve the progress of successful files if another file later fails.
        report = dict(complete=len(results) == len(jobs) and all(r["status"] == "ok" for r in results),
                      layout=args.layout, cameras_requested=args.cameras, requested=len(jobs),
                      succeeded=sum(r["status"] == "ok" for r in results),
                      failed=sum(r["status"] != "ok" for r in results), items=results)
        write_json(args.output_dir/"mp4_report.json", report)
    with (args.output_dir/"mp4_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        columns = ["input", "layout", "camera_id", "camera_ids", "output", "status", "duration_seconds", "frames", "bytes", "error"]
        writer.writerow(columns)
        for r in results:
            writer.writerow([r.get(k, "") for k in columns])
    print(f"Finished: {report['succeeded']}/{len(jobs)} succeeded; report: {args.output_dir/'mp4_report.json'}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
