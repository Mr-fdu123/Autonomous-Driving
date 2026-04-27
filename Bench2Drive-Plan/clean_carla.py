import argparse
from pathlib import Path

import numpy as np


def iter_npz_files(data_root: Path):
    for p in data_root.rglob("*.npz"):
        yield p


def has_nan_inf(arr: np.ndarray) -> bool:
    return np.isnan(arr).any() or np.isinf(arr).any()


def clean_array(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="In-place NaN/Inf clean for CARLA npz")
    parser.add_argument("--data_root", required=True, type=str, help="directory with .npz files")
    parser.add_argument("--max_files", type=int, default=None, help="limit number of files")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    total = 0
    cleaned = 0

    for path in iter_npz_files(data_root):
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            data_dict = {k: data[k] for k in data.files}

        changed = False
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray) and has_nan_inf(v):
                data_dict[k] = clean_array(v)
                changed = True

        if changed:
            np.savez(path, **data_dict)
            cleaned += 1

        total += 1
        if args.max_files is not None and total >= args.max_files:
            break
        if total % 1000 == 0:
            print(f"Processed {total} files...", flush=True)

    print(f"[DONE] processed={total} cleaned={cleaned} (in-place)", flush=True)


if __name__ == "__main__":
    main()
