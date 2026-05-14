"""
Core MCAP merge logic.

Behavior:
1. Read messages from multiple MCAP files (all topics in each file).
2. Remap topics for multi-device layout (/robot0/ -> /robotN/).
3. Sort all messages by timestamp and write a single MCAP.
4. Use NonSeekingReader for sequential reads (no Summary dependency).

Filename convention (per DAS spec):
  DAS-<Type>_<Timestamp14>_<Role>_<Location>_<CPUID6>_<UUID8>.mcap
  Type     : Ego | Finger | Gripper | Dex
  Role     : master | sub | none
  Location : left | right | center | none
  UUID8    : first 8 hex chars of batch UUID (the grouping key)

Files are classified by Type + Location:
  Ego    + center → ego (merge pivot, internal biz_role=master)
  Finger + left   → left finger (internal biz_role=sub_left)
  Finger + right  → right finger (internal biz_role=sub_right)

CLI:
- --scan-dir <dir> [--output-dir DIR]  (recursive; each UUID8 must yield exactly 3 files)
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
from typing import Dict, List, Optional, Set, Tuple, NamedTuple
from pathlib import Path

logger = logging.getLogger(__name__)

# DAS naming: DAS-<Type>_<ts14>_<role>_<location>_<cpuid>_<uuid8>[_<suffix>].mcap
# Only the fields we actually use are validated strictly (Type, Timestamp,
# Location, UUID8). Role and CPUID are opaque tokens so minor variations in
# legacy data still parse, and a trailing tag after UUID8 is allowed.
FILENAME_RE = re.compile(
    r"(?i)^DAS[-_](?P<type>Ego|Finger|Gripper|Dex)_"
    r"(?P<ts>\d{14})_"
    r"[^_]+_"                                    # role (not consumed)
    r"(?P<location>left|right|center|none)_"
    r"[^_]+_"                                    # cpuid (not consumed)
    r"(?P<uuid8>[0-9a-f]{8})"
    r"(?:_[^.]*)?\.mcap$"
)

# Type + Location → role tag used by the merge pipeline.
_CLASSIFY_MAP = {
    ("ego", "center"): "ego",
    ("finger", "left"): "left",
    ("finger", "right"): "right",
}

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


def _parse_mcap_filename(path: Path) -> Optional[Tuple[str, ParsedMcapFile]]:
    """
    Parse DAS-spec filename. Returns (role, ParsedMcapFile) with role in
    {'ego','left','right'}, or None if the name doesn't match or its
    Type+Location combo isn't part of the ego+finger triple.
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group("ts"), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    key = (m.group("type").lower(), m.group("location").lower())
    role = _CLASSIFY_MAP.get(key)
    if role is None:
        return None
    return role, ParsedMcapFile(path=path, uuid8=m.group("uuid8").lower(), record_dt=dt)


def classify_mcap_file(path: Path) -> Optional[Tuple[str, ParsedMcapFile]]:
    """Classify by filename. Returns (role, ParsedMcapFile) or None if unknown."""
    return _parse_mcap_filename(path)


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


def _match_orphans_by_ts(
    orphan_ego: List[Tuple[str, ParsedMcapFile]],
    orphan_left: List[Tuple[str, ParsedMcapFile]],
    orphan_right: List[Tuple[str, ParsedMcapFile]],
    tolerance_sec: float,
) -> List[Tuple[str, ParsedMcapFile, str, ParsedMcapFile, str, ParsedMcapFile]]:
    """
    Pair orphan ego/left/right files using filename timestamps when UUID8
    grouping fails. Greedy: process egos chronologically, choose the unused
    left + right that minimize triplet spread (max_ts - min_ts), provided
    spread ≤ tolerance_sec. Returns
    [(ego_uuid, ego, left_uuid, left, right_uuid, right), ...].
    """
    used_left: Set[Path] = set()
    used_right: Set[Path] = set()
    matches: List[Tuple[str, ParsedMcapFile, str, ParsedMcapFile, str, ParsedMcapFile]] = []

    for ego_uid, ego in sorted(orphan_ego, key=lambda x: x[1].record_dt):
        best: Optional[Tuple[str, ParsedMcapFile, str, ParsedMcapFile]] = None
        best_spread: Optional[float] = None
        for l_uid, l in orphan_left:
            if l.path in used_left:
                continue
            for r_uid, r in orphan_right:
                if r.path in used_right:
                    continue
                dts = (ego.record_dt, l.record_dt, r.record_dt)
                spread = (max(dts) - min(dts)).total_seconds()
                if spread > tolerance_sec:
                    continue
                if best_spread is None or spread < best_spread:
                    best = (l_uid, l, r_uid, r)
                    best_spread = spread
        if best is None:
            continue
        l_uid, l, r_uid, r = best
        used_left.add(l.path)
        used_right.add(r.path)
        matches.append((ego_uid, ego, l_uid, l, r_uid, r))

    return matches


def discover_triplets(scan_dir: Path) -> List[Tuple[str, ParsedMcapFile, ParsedMcapFile, ParsedMcapFile]]:
    """
    Recursively scan *.mcap under scan_dir, return merge-ready triples
    [(uuid8, ego, left, right), ...].

    Primary: group by UUID8 — a complete group has exactly 1 ego + 1 left + 1 right.
    Fallback: when a UUID8 group is incomplete (collection device bug yields
    mismatched UUIDs), pool the orphan files and re-pair by filename timestamp
    with spread ≤ TIMESTAMP_MAX_SPREAD_SEC. Output uuid8 for a ts-matched
    triple uses the ego's uuid8 (ego is the merge pivot).
    """
    if not scan_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {scan_dir}")

    buckets: Dict[str, Dict[str, ParsedMcapFile]] = {}
    skipped: List[str] = []
    total_seen = 0

    for path in sorted(scan_dir.rglob("*.mcap")):
        if not path.is_file():
            continue
        total_seen += 1
        classified = classify_mcap_file(path)
        if not classified:
            skipped.append(str(path.relative_to(scan_dir)))
            continue
        role, info = classified
        uid = info.uuid8
        bucket = buckets.setdefault(uid, {})
        if role in bucket:
            raise ValueError(
                f"UUID {uid}: duplicate {role} file; already have {bucket[role].path.name}, "
                f"also saw {path.name}"
            )
        bucket[role] = info

    logger.info(
        f"Scanned {total_seen} .mcap file(s) under {scan_dir}; "
        f"{len(buckets)} UUID8 group(s) found, {len(skipped)} unrecognized."
    )
    if skipped:
        logger.warning(f"{len(skipped)} file(s) not recognized as ego/left/right; skipped:")
        for n in skipped[:20]:
            logger.warning(f"  - {n}")
        if len(skipped) > 20:
            logger.warning(f"  ... and {len(skipped) - 20} more")

    # Primary: complete UUID8 groups.
    triplets: List[Tuple[str, ParsedMcapFile, ParsedMcapFile, ParsedMcapFile]] = []
    incomplete: List[str] = []
    for uid in sorted(buckets.keys()):
        b = buckets[uid]
        if set(b.keys()) == {"ego", "left", "right"}:
            ego, left, right = b["ego"], b["left"], b["right"]
            assert_triplet_timestamps_aligned(ego, left, right, uid)
            triplets.append((uid, ego, left, right))
        else:
            incomplete.append(uid)

    logger.info(
        f"Primary UUID8 match: {len(triplets)} complete, {len(incomplete)} incomplete."
    )

    # Fallback: ts-based re-pairing for orphan files.
    if incomplete:
        orphans: Dict[str, List[Tuple[str, ParsedMcapFile]]] = {"ego": [], "left": [], "right": []}
        for uid in incomplete:
            b = buckets[uid]
            missing = sorted({"ego", "left", "right"} - set(b.keys()))
            logger.warning(
                f"  incomplete UUID8={uid}: have {sorted(b.keys())}, missing {missing}; "
                f"files: {[b[k].path.name for k in sorted(b.keys())]}"
            )
            for role, f in b.items():
                orphans[role].append((uid, f))

        logger.warning(
            f"Fallback timestamp matching (tolerance ±{TIMESTAMP_MAX_SPREAD_SEC}s); "
            f"orphans: ego={len(orphans['ego'])}, "
            f"left={len(orphans['left'])}, right={len(orphans['right'])}"
        )

        matched = _match_orphans_by_ts(
            orphans["ego"], orphans["left"], orphans["right"], TIMESTAMP_MAX_SPREAD_SEC
        )
        used: Set[Path] = set()
        for ego_uid, ego, l_uid, left, r_uid, right in matched:
            logger.warning(
                f"  ts-matched: "
                f"ego[{ego_uid} @ {ego.record_dt:%Y-%m-%d %H:%M:%S}] {ego.path.name} + "
                f"left[{l_uid} @ {left.record_dt:%H:%M:%S}] {left.path.name} + "
                f"right[{r_uid} @ {right.record_dt:%H:%M:%S}] {right.path.name} "
                f"→ output uuid={ego_uid}"
            )
            triplets.append((ego_uid, ego, left, right))
            used.update([ego.path, left.path, right.path])

        for role in ("ego", "left", "right"):
            for uid, f in orphans[role]:
                if f.path not in used:
                    logger.warning(
                        f"  no ts match: {role} {f.path.name} (uuid={uid}, ts={f.record_dt})"
                    )

    logger.info(f"Total triplets ready for merge: {len(triplets)}")
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
        description="Recursively scan a directory for DAS MCAPs; group by UUID8 "
        "and merge every complete ego + left + right finger triple."
    )
    p.add_argument(
        "--scan-dir",
        required=True,
        help="Recursively scan directory for *.mcap; group by UUID8 (each group "
        "must yield exactly 1 ego + 1 left + 1 right).",
    )
    p.add_argument(
        "--output-dir",
        help="Output directory for merged MCAPs (default: ./merged under cwd). "
        "Files are named merged_ego_finger_<YYYYMMDDHHMMSS>_<uuid8>.mcap",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_root = resolve_output_dir(args.output_dir)
    base = Path(args.scan_dir).resolve()
    triplets = discover_triplets(base)
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Scan mode: {len(triplets)} complete triple(s); output dir {out_root}")

    for uid, ego_p, left_p, right_p in triplets:
        out_path = out_root / merged_ego_finger_basename(ego_p.record_dt, uid)
        logger.info(f"\n>>> Merging UUID={uid} -> {out_path}")
        infos = build_mcap_file_infos(
            str(ego_p.path.resolve()),
            str(left_p.path.resolve()),
            str(right_p.path.resolve()),
        )
        merge_multiple_mcap_files(infos, str(out_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
