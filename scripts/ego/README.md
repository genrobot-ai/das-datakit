# `merge_ego_finger_mcaps_by_uuid.py`

Merges Ego + left/right finger MCAPs into one file. Dependencies are in the repo root `requirements.txt`.

## How to run

**1. Three explicit paths**

```bash
python scripts/ego/merge_ego_finger_mcaps_by_uuid.py \
  --ego PATH \
  --left_finger PATH \
  --right_finger PATH
```

**2. Scan a folder** (pairs files by UUID in the filename)

```bash
python scripts/ego/merge_ego_finger_mcaps_by_uuid.py --scan-dir PATH
```

**Optional (both modes):** `--output-dir DIR` (or `--output_dir`). If omitted, outputs go to `./merged/`.

Output file name: `merged_ego_finger_<YYYYMMDDHHMMSS>_<uuid>.mcap`

**3. Help**

```bash
python scripts/ego/merge_ego_finger_mcaps_by_uuid.py --help
```

Do not mix `--scan-dir` with `--ego` / `--left_finger` / `--right_finger`.
