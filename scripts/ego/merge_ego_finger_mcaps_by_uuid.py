"""
Core MCAP merge logic.

Behavior:
1. Read messages from multiple MCAP files (all topics in each file).
2. Remap topics for multi-device layout (/robot0/ -> /robotN/).
3. Sort all messages by timestamp and write a single MCAP.
4. Use NonSeekingReader for sequential reads (no Summary dependency).

CLI:
- Explicit triple: --ego --left_finger --right_finger [--output-dir]
- Scan directory: --scan-dir [--output-dir]
- Output file name: merged_ego_finger_<YYYYMMDDHHMMSS>_<uuid8>.mcap (timestamp from ego filename)
- Default output directory: ./merged (under current working directory) if --output-dir is omitted
"""

import argparse
import logging
import os
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from mcap.reader import NonSeekingReader
from mcap.writer import Writer
from typing import Dict, List, Optional, Tuple, NamedTuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Ego basename: DAS-Ego_YYYYMMDDHHMMSS_...center...leader..._<uuid8>[_<suffix>].mcap
EGO_NAME_RE = re.compile(
    r"(?i)DAS[-_]Ego_(\d{14})_.*center.*leader.*_([0-9a-f]{8})(?:_[0-9a-f]+)?\.mcap$"
)
# Finger: finger...YYYYMMDD_HHMMSS...sub_left|sub_right...<uuid8>.mcap
LEFT_FINGER_NAME_RE = re.compile(
    r"(?i)finger.*?(\d{8})_(\d{6}).*sub_left.*?([0-9a-f]{8})\.mcap$"
)
RIGHT_FINGER_NAME_RE = re.compile(
    r"(?i)finger.*?(\d{8})_(\d{6}).*sub_right.*?([0-9a-f]{8})\.mcap$"
)

TIMESTAMP_MAX_SPREAD_SEC = 1.0


class MCAPMessage(NamedTuple):
    """Structured MCAP message for merge."""
    timestamp: int
    schema: object
    channel: object
    message: object
    topic_index: int


@dataclass
class ParsedMcapFile:
    """Per-file metadata parsed from the filename."""
    path: Path
    uuid8: str
    record_dt: datetime


def get_topic_index(biz_role: str) -> Optional[int]:
    biz_role_map = {
        "master": 0,
        "sub_left": 1,
        "sub_right": 2
    }
    return biz_role_map.get(biz_role, None)


def get_topic_prefix(platform: str, biz_role: str) -> Optional[str]:
    part1 = ""
    part2 = ""

    part1_map = {
        "DAS-EGO": "ego",
        "DAS-F": "finger"
    }
    part2_map = {
        "master": "center",
        "sub_left": "left",
        "sub_right": "right"
    }
    if platform not in part1_map:
        logger.error(f"  ✗ Unknown platform: {platform}")
        raise ValueError(f"Unknown platform: {platform}")
    if biz_role not in part2_map:
        logger.error(f"  ✗ Unknown biz_role: {biz_role}")
        raise ValueError(f"Unknown biz_role: {biz_role}")
    part1 = part1_map[platform]
    part2 = part2_map[biz_role]
    topic_prefix = f"{part1}/{part2}"
    return topic_prefix


def remap_topic(original_topic: str, topic_index: int) -> str:
    if topic_index is not None:
        return original_topic.replace("/robot0/", f"/robot{topic_index}/", 1)
    return original_topic


def _parse_ego_filename(path: Path) -> Optional[ParsedMcapFile]:
    m = EGO_NAME_RE.search(path.name)
    if not m:
        return None
    ts14, uuid8 = m.group(1), m.group(2).lower()
    try:
        dt = datetime.strptime(ts14, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return ParsedMcapFile(path=path, uuid8=uuid8, record_dt=dt)


def _parse_left_finger_filename(path: Path) -> Optional[ParsedMcapFile]:
    m = LEFT_FINGER_NAME_RE.search(path.name)
    if not m:
        return None
    d8, t6, uuid8 = m.group(1), m.group(2), m.group(3).lower()
    try:
        dt = datetime.strptime(d8 + t6, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return ParsedMcapFile(path=path, uuid8=uuid8, record_dt=dt)


def _parse_right_finger_filename(path: Path) -> Optional[ParsedMcapFile]:
    m = RIGHT_FINGER_NAME_RE.search(path.name)
    if not m:
        return None
    d8, t6, uuid8 = m.group(1), m.group(2), m.group(3).lower()
    try:
        dt = datetime.strptime(d8 + t6, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return ParsedMcapFile(path=path, uuid8=uuid8, record_dt=dt)


def classify_mcap_file(path: Path) -> Optional[Tuple[str, ParsedMcapFile]]:
    """
    Classify a file by basename pattern.
    Returns (role, ParsedMcapFile) with role in {'ego','left','right'}, or None if unknown.
    """
    ego = _parse_ego_filename(path)
    if ego:
        return ("ego", ego)
    left = _parse_left_finger_filename(path)
    if left:
        return ("left", left)
    right = _parse_right_finger_filename(path)
    if right:
        return ("right", right)
    return None


def merged_ego_finger_basename(record_dt: datetime, uuid8: str) -> str:
    """Output MCAP basename: merged_ego_finger_<ts14>_<uuid8>.mcap"""
    ts14 = record_dt.strftime("%Y%m%d%H%M%S")
    return f"merged_ego_finger_{ts14}_{uuid8.lower()}.mcap"


def resolve_output_dir(output_dir: Optional[str]) -> Path:
    """Directory for merged outputs; default ./merged (cwd)."""
    if output_dir:
        p = Path(output_dir).expanduser()
        return p.resolve()
    return (Path.cwd() / "merged").resolve()


def parse_explicit_triplet_files(ego_s: str, left_s: str, right_s: str) -> Tuple[ParsedMcapFile, ParsedMcapFile, ParsedMcapFile]:
    """Parse ego / left / right paths; require matching uuid8 and aligned filename timestamps."""
    ego_p = _parse_ego_filename(Path(ego_s))
    if not ego_p:
        raise ValueError(f"Ego filename does not match expected pattern: {Path(ego_s).name}")
    left_p = _parse_left_finger_filename(Path(left_s))
    if not left_p:
        raise ValueError(f"Left finger filename does not match expected pattern: {Path(left_s).name}")
    right_p = _parse_right_finger_filename(Path(right_s))
    if not right_p:
        raise ValueError(f"Right finger filename does not match expected pattern: {Path(right_s).name}")

    u_e, u_l, u_r = ego_p.uuid8, left_p.uuid8, right_p.uuid8
    if not (u_e == u_l == u_r):
        raise ValueError(f"UUID8 mismatch across inputs: ego={u_e}, left={u_l}, right={u_r}")

    assert_triplet_timestamps_aligned(ego_p, left_p, right_p, u_e)
    return ego_p, left_p, right_p


def assert_triplet_timestamps_aligned(
    ego: ParsedMcapFile, left: ParsedMcapFile, right: ParsedMcapFile, uuid8: str
) -> None:
    dts = [ego.record_dt, left.record_dt, right.record_dt]
    spread = (max(dts) - min(dts)).total_seconds()
    if spread > TIMESTAMP_MAX_SPREAD_SEC:
        fmt = "%Y-%m-%d %H:%M:%S"
        raise ValueError(
            f"UUID {uuid8}: filename timestamps differ by {spread:.3f}s "
            f"(>{TIMESTAMP_MAX_SPREAD_SEC}s); refusing merge. "
            f"ego={ego.record_dt.strftime(fmt)}, left={left.record_dt.strftime(fmt)}, "
            f"right={right.record_dt.strftime(fmt)}"
        )


def discover_triplets(scan_dir: Path) -> List[Tuple[str, ParsedMcapFile, ParsedMcapFile, ParsedMcapFile]]:
    """
    Scan *.mcap in scan_dir, group by uuid8, return merge-ready triples:
    [(uuid8, ego, left, right), ...]
    """
    if not scan_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {scan_dir}")

    buckets: Dict[str, Dict[str, ParsedMcapFile]] = {}
    skipped: List[str] = []

    for path in sorted(scan_dir.glob("*.mcap")):
        if not path.is_file():
            continue
        classified = classify_mcap_file(path)
        if not classified:
            skipped.append(path.name)
            continue
        role, info = classified
        uid = info.uuid8
        if uid not in buckets:
            buckets[uid] = {}
        key = {"ego": "ego", "left": "left", "right": "right"}[role]
        if key in buckets[uid]:
            raise ValueError(
                f"UUID {uid}: duplicate {role} file; already have {buckets[uid][key].path.name}, "
                f"also saw {path.name}"
            )
        buckets[uid][key] = info

    if skipped:
        logger.warning(f"{len(skipped)} file(s) not recognized as ego/left/right; skipped:")
        for n in skipped[:20]:
            logger.warning(f"  - {n}")
        if len(skipped) > 20:
            logger.warning(f"  ... and {len(skipped) - 20} more")

    triplets: List[Tuple[str, ParsedMcapFile, ParsedMcapFile, ParsedMcapFile]] = []
    for uid in sorted(buckets.keys()):
        b = buckets[uid]
        has = set(b.keys())
        if has == {"ego", "left", "right"}:
            ego, left, right = b["ego"], b["left"], b["right"]
            assert_triplet_timestamps_aligned(ego, left, right, uid)
            triplets.append((uid, ego, left, right))
        else:
            logger.warning(
                f"UUID {uid}: incomplete set (have {sorted(has)} only); skip merge. "
                f"Files: {[b[k].path.name for k in sorted(has)]}"
            )

    if not triplets:
        raise RuntimeError(f"No complete ego+left+right triple found under {scan_dir}")
    return triplets


def read_mcap_messages(mcap_info: Dict, cut_timestamp: int = None) -> List[MCAPMessage]:
    input_path = mcap_info["local_path"]
    topic_index = get_topic_index(mcap_info["biz_role"])
    logger.info(
        f"\nReading: {Path(input_path).name} "
        f"(platform={mcap_info['platform']}, biz_role={mcap_info['biz_role']}, topic_index={topic_index})"
    )

    messages = []
    cut_count = 0
    cut_topics = set()
    with open(input_path, "rb") as f:
        reader = NonSeekingReader(f)
        msg_count = 0
        filtered_count = 0

        for schema, channel, message in reader.iter_messages():
            orig_topic = channel.topic

            # Prefer publish_time; fall back to log_time if missing.
            sort_timestamp = message.publish_time or message.log_time
            if sort_timestamp is None:
                logger.warning(f"No timestamp on message; skip | Topic: {orig_topic} | ChannelID: {channel.id}")
                filtered_count += 1
                continue

            if cut_timestamp and sort_timestamp > cut_timestamp:
                cut_count += 1
                cut_topics.add(orig_topic)
                continue

            messages.append(
                MCAPMessage(
                    timestamp=sort_timestamp,
                    schema=schema,
                    channel=channel,
                    message=message,
                    topic_index=topic_index
                )
            )
            msg_count += 1

        logger.info(f"  Raw message rows: {msg_count + filtered_count}")
        logger.info(f"  Kept (with timestamp): {msg_count}")
        logger.info(f"  Skipped (no timestamp): {filtered_count}")

    if cut_timestamp:
        logger.info(
            f"  cut_timestamp: {cut_timestamp} ({cut_count} message(s) dropped "
            f"across {len(cut_topics)} topic(s))"
        )

    return messages


def get_timestamp_range(messages: List[MCAPMessage]) -> Tuple[Optional[int], Optional[int]]:
    begin_ts = None
    end_ts = None
    for msg in messages:
        if begin_ts is None or msg.timestamp < begin_ts:
            begin_ts = msg.timestamp
        if end_ts is None or msg.timestamp > end_ts:
            end_ts = msg.timestamp
    return begin_ts, end_ts


def merge_multiple_mcap_files(
    mcap_file_infos: List[Dict],
    output_path: str,
) -> Dict[str, int]:

    logger.info("=" * 80)
    if len(mcap_file_infos) == 1:
        logger.info("Single-file MCAP pass-through")
    else:
        logger.info("Merging MCAPs (global sort by timestamp)")
    logger.info("=" * 80)
    logger.info(f"Input files: {len(mcap_file_infos)}")
    logger.info(f"Output file: {Path(output_path).name}")

    all_messages = []

    # Read non-master first to compute master cut time (min end time across subs).
    min_ts = None
    for mcap_info in mcap_file_infos:
        if mcap_info["biz_role"] == "master":
            continue
        messages = read_mcap_messages(mcap_info)
        all_messages.extend(messages)

        _, end_ts = get_timestamp_range(messages)
        if min_ts is None or (end_ts is not None and end_ts < min_ts):
            min_ts = end_ts

    # Read master with truncation at min_ts.
    for mcap_info in mcap_file_infos:
        if mcap_info["biz_role"] != "master":
            continue
        messages = read_mcap_messages(mcap_info, min_ts)
        all_messages.extend(messages)

    all_messages.sort(key=lambda x: x.timestamp)
    logger.info(f"\nTotal messages after merge sort: {len(all_messages)}")
    if all_messages:
        logger.info(f"Time span: {all_messages[0].timestamp / 1e9:.2f}s ~ {all_messages[-1].timestamp / 1e9:.2f}s")

    schema_map_global = {}  # (name, encoding, data) -> schema_id
    channel_map_global = {}  # (new_topic, msg_encoding, schema_key) -> channel_id

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "wb") as out_f:
        writer = Writer(out_f)
        writer.start()
        written_count = 0

        for msg_struct in all_messages:
            orig_topic = msg_struct.channel.topic
            new_topic = remap_topic(orig_topic, msg_struct.topic_index)

            schema = msg_struct.schema
            schema_key = (schema.name, schema.encoding, schema.data)
            if schema_key not in schema_map_global:
                schema_id = writer.register_schema(
                    name=schema.name,
                    encoding=schema.encoding,
                    data=schema.data
                )
                schema_map_global[schema_key] = schema_id
            schema_id = schema_map_global[schema_key]

            channel = msg_struct.channel
            channel_key = (new_topic, channel.message_encoding, schema_key)
            if channel_key not in channel_map_global:
                channel_id = writer.register_channel(
                    schema_id=schema_id,
                    topic=new_topic,
                    message_encoding=channel.message_encoding,
                    metadata=channel.metadata
                )
                channel_map_global[channel_key] = channel_id
            channel_id = channel_map_global[channel_key]

            message = msg_struct.message
            writer.add_message(
                channel_id=channel_id,
                log_time=message.log_time,
                data=message.data,
                publish_time=message.publish_time
            )
            written_count += 1

            if written_count % 1000 == 0:
                logger.info(f"  Written {written_count}/{len(all_messages)} messages...")

        writer.finish()

    duration = 0.0
    record_time_ms = 0
    if all_messages:
        duration = (all_messages[-1].timestamp - all_messages[0].timestamp) / 1e9
        record_time_ms = int(all_messages[0].timestamp // 1e6)  # first message time, ms

    logger.info("\n" + "=" * 80)
    logger.info("Merge finished.")
    logger.info("=" * 80)
    logger.info(f"Output path: {output_path}")
    logger.info(f"Messages written: {written_count}")
    logger.info(f"Duration (span): {duration:.2f}s")

    logger.info("\nVerifying output:")
    try:
        with open(output_path, "rb") as f:
            reader = NonSeekingReader(f)
            topics = set()
            total_msgs = 0
            for _, channel, _ in reader.iter_messages():
                topics.add(channel.topic)
                total_msgs += 1

        logger.info(f"Output message count: {total_msgs}")
        logger.info(f"Topics ({len(topics)}):")
        for topic in sorted(topics):
            logger.info(f"  - {topic}")
    except Exception:
        logger.error("Verification failed:\n%s", traceback.format_exc())

    return {
        "total_messages": written_count,
        "file_count": len(mcap_file_infos),
        "duration": duration,
        "record_time_ms": record_time_ms
    }


def build_mcap_file_infos(ego: str, left_finger: str, right_finger: str) -> List[Dict]:
    return [
        {"platform": "DAS-EGO", "biz_role": "master", "local_path": ego},
        {"platform": "DAS-F", "biz_role": "sub_left", "local_path": left_finger},
        {"platform": "DAS-F", "biz_role": "sub_right", "local_path": right_finger},
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge Ego + left/right finger MCAPs (explicit paths or directory scan by UUID8)."
    )
    p.add_argument("--ego", help="Ego (center leader) MCAP path")
    p.add_argument("--left_finger", help="Left finger MCAP path")
    p.add_argument("--right_finger", help="Right finger MCAP path")
    p.add_argument(
        "--scan-dir",
        help="Scan directory for *.mcap; auto-match ego/left/right by UUID8 and merge each triple",
    )
    p.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        help="Output directory for merged MCAPs (default: ./merged under cwd). "
        "Files are named merged_ego_finger_<YYYYMMDDHHMMSS>_<uuid8>.mcap",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    scan_dir = args.scan_dir
    ego, left, right = args.ego, args.left_finger, args.right_finger
    out_root = resolve_output_dir(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if scan_dir:
        if ego or left or right:
            raise SystemExit(
                "In scan mode use only --scan-dir (and optional --output-dir); "
                "do not combine with --ego / --left_finger / --right_finger."
            )
        base = Path(scan_dir).resolve()

        triplets = discover_triplets(base)
        logger.info(f"Scan mode: {len(triplets)} complete triple(s); output dir {out_root}")

        for uid, ego_p, left_p, right_p in triplets:
            out_name = merged_ego_finger_basename(ego_p.record_dt, uid)
            out_path = out_root / out_name
            logger.info(f"\n>>> Merging UUID={uid} -> {out_path}")
            infos = build_mcap_file_infos(
                str(ego_p.path.resolve()),
                str(left_p.path.resolve()),
                str(right_p.path.resolve()),
            )
            merge_multiple_mcap_files(infos, str(out_path))
        return

    if not (ego and left and right):
        raise SystemExit(
            "Usage:\n"
            "  1) --ego A.mcap --left_finger B.mcap --right_finger C.mcap [--output-dir DIR]\n"
            "  2) --scan-dir <dir> [--output-dir DIR]\n"
            "Default output directory is ./merged. Output name: merged_ego_finger_<ts>_<uuid>.mcap"
        )
    for label, path in (("ego", ego), ("left_finger", left), ("right_finger", right)):
        if not Path(path).is_file():
            raise SystemExit(f"{label}: file missing or not a regular file: {path}")

    try:
        ego_p, left_p, right_p = parse_explicit_triplet_files(ego, left, right)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    out_path = out_root / merged_ego_finger_basename(ego_p.record_dt, ego_p.uuid8)
    logger.info(f"Writing merged MCAP to {out_path}")
    merge_multiple_mcap_files(
        build_mcap_file_infos(
            str(Path(ego).resolve()),
            str(Path(left).resolve()),
            str(Path(right).resolve()),
        ),
        str(out_path),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
