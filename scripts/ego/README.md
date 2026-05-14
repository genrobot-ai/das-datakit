# `merge_ego_finger_mcaps_by_uuid.py`

Recursively scans a directory for DAS MCAPs, groups them by batch UUID8, and merges each complete `ego + left finger + right finger` triple into a single MCAP. Dependencies are in the repo root `requirements.txt`.

## How to run

```bash
python scripts/ego/merge_ego_finger_mcaps_by_uuid.py --scan-dir PATH [--output-dir DIR]
```

- `--scan-dir PATH` *(required)* — directory to scan recursively for `*.mcap`.
- `--output-dir DIR` *(optional)* — where to write merged outputs. Defaults to `./merged/` under cwd.

Output file name: `merged_ego_finger_<YYYYMMDDHHMMSS>_<uuid8>.mcap` (timestamp taken from the ego file).

## Filename convention

```text
DAS-<Type>_<Timestamp14>_<Role>_<Location>_<CPUID6>_<UUID8>.mcap
```

| Field     | Values                                                                 | Required by tool         |
|-----------|------------------------------------------------------------------------|--------------------------|
| Type      | `Ego` / `Finger` / `Gripper` / `Dex`                                   | strictly validated       |
| Timestamp | 14 digits, `YYYYMMDDHHMMSS`                                            | strictly validated       |
| Role      | `master` / `sub` / `none`                                              | opaque (not consumed)    |
| Location  | `left` / `right` / `center` / `none`                                   | strictly validated       |
| CPUID     | trailing 6 chars of device CPUID                                       | opaque (not consumed)    |
| UUID8     | first 8 hex chars of batch UUID — the grouping key                     | strictly validated       |


## Grouping rules

Files are classified by **Type + Location**:

| File                       | Role tag | Internal `biz_role`    |
|----------------------------|----------|------------------------|
| `DAS-Ego ... center ...`   | `ego`    | `master` (merge pivot) |
| `DAS-Finger ... left ...`  | `left`   | `sub_left`             |
| `DAS-Finger ... right ...` | `right`  | `sub_right`            |

Two-stage matching:

1. **Primary — by UUID8.** All files sharing the same `UUID8` form a batch. A batch is complete when it has exactly 1 ego + 1 left + 1 right.
2. **Fallback — by filename timestamp.** If any UUID8 group is incomplete (collection device bug yields mismatched UUIDs across the three devices), the orphan files from incomplete buckets are pooled and re-paired by timestamp: for each orphan ego (processed in chronological order), pick the unused left + right that minimize triplet spread `max(ts) − min(ts)` provided the spread is ≤ **1 second**. Files chosen this way are consumed and won't be reused. The merged output uses the **ego's UUID8** as its identifier (ego is the merge pivot).

Across both stages, the same 1-second spread limit also applies to primary UUID8-matched triples — if a complete group has filename timestamps that disagree by more than 1 s, the merge is refused for that UUID.

Duplicate `(uuid, role)` pairs (two files claiming the same role within the same UUID8) are treated as a hard error and abort the run. Orphan files that find no ts match are reported and skipped.

## Help

```bash
python scripts/ego/merge_ego_finger_mcaps_by_uuid.py --help
```
