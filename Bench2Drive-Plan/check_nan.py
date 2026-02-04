import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


def iter_npz_files(data_root: Path, list_path: Optional[Path]):
    if list_path is not None:
        with open(list_path, "r", encoding="utf-8") as f:
            names = json.load(f)
        for name in names:
            yield data_root / name
    else:
        for p in data_root.rglob("*.npz"):
            yield p


def check_array(name: str, arr: np.ndarray):
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    if nan_count or inf_count:
        return {
            "field": name,
            "shape": list(arr.shape),
            "nan": int(nan_count),
            "inf": int(inf_count),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan CARLA npz for NaN/Inf")
    parser.add_argument("--data_root", required=True, type=str, help="directory with .npz files")
    parser.add_argument("--list", default=None, type=str, help="optional json list of file names")
    parser.add_argument("--output", default="bad_samples.json", type=str, help="output json path")
    parser.add_argument("--max_files", type=int, default=None, help="limit number of files")
    parser.add_argument("--print_limit", type=int, default=50, help="max bad samples to print")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    list_path = Path(args.list) if args.list else None

    bad = []
    total = 0
    printed = 0

    for path in iter_npz_files(data_root, list_path):
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            issues = []
            for key in data.files:
                arr = data[key]
                if not isinstance(arr, np.ndarray):
                    continue
                res = check_array(key, arr)
                if res:
                    issues.append(res)
            if issues:
                entry = {"file": str(path), "issues": issues}
                bad.append(entry)
                if printed < args.print_limit:
                    print(f"[BAD] {path}")
                    for item in issues:
                        print(f"  - {item['field']} shape={item['shape']} nan={item['nan']} inf={item['inf']}")
                    printed += 1

        total += 1
        if args.max_files is not None and total >= args.max_files:
            break
        if total % 200 == 0:
            print(f"Scanned {total} files...", flush=True)

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bad, f, indent=2)

    print(f"[DONE] Scanned {total} files, bad={len(bad)}, report={out_path}", flush=True)


if __name__ == "__main__":
    main()
