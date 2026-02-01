import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from diffusion_dataset import DiffusionDataset


try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def find_town_name(path: Path) -> str:
    # 将 Path 对象转换为字符串
    path_str = str(path)
    
    # 按照 '_' 分割字符串
    parts = path_str.split('_')
    
    # 遍历每个部分，匹配形如 'TownXX' 的部分
    for part in parts:
        if re.match(r"Town\d{2}", part):  # 匹配 "Town" 后跟两个数字
            return part  # 返回匹配的部分
    
    return "UnknownTown"

def resolve_map_file(map_path: str, town: str):
    if map_path is None:
        return None
    p = Path(map_path)
    if p.is_file():
        return p
    candidates = [
        p / f"{town}_HD_map.npz",
        p / f"{town}_lanemarkings.npz",
        p / f"{town}.npz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def collect_route_dirs(data_root: str):
    route_dirs = set()
    for anno_dir in Path(data_root).rglob("anno"):
        if anno_dir.is_dir():
            route_dirs.add(anno_dir.parent)
    return sorted(route_dirs)


def process_route(route_dir: Path, map_file: Path, args, output_dir: Path):
    dataset = DiffusionDataset(
        log_file_path=str(route_dir),
        map_file_path=str(map_file),
        cache_dir=args.cache_dir,
        agent_num=args.agent_num,
        lane_num=args.lane_num,
        lane_len=args.lane_len,
        route_num=args.route_num,
        static_objects_num=args.static_objects_num,
    )

    if len(dataset) < args.past_len + args.future_len:
        print(f"[SKIP] {route_dir} (len={len(dataset)}) < past+future ({args.past_len + args.future_len})", flush=True)
        return []

    data_list = []
    town = find_town_name(route_dir)
    route_name = route_dir.name

    idx_iter = range(args.past_len - 1, len(dataset) - args.future_len, args.stride)
    if tqdm is not None:
        idx_iter = tqdm(
            idx_iter,
            desc=f"route {route_name}",
            leave=False,
            mininterval=1.0,
        )

    for idx in idx_iter:
        data = dataset[idx]
        token = Path(dataset._log_file[idx]).stem
        filename = f"{town}_{route_name}_{token}.npz"
        np.savez(output_dir / filename, **data)
        data_list.append(filename)

    return data_list


def main() -> None:
    parser = argparse.ArgumentParser(description="CARLA/Bench2Drive preprocessing for Diffusion-Planner")
    parser.add_argument("--data_path", required=True, type=str, help="dataset root")
    parser.add_argument("--map_path", required=True, type=str, help="map root or map file")
    parser.add_argument("--train_set_path", required=True, type=str, help="output npz dir")
    parser.add_argument("--output_list", default="./diffusion_planner_training_carla.json", type=str)
    parser.add_argument("--cache_dir", default=None, type=str)

    parser.add_argument("--past_len", default=21, type=int)
    parser.add_argument("--future_len", default=80, type=int)
    parser.add_argument("--stride", default=1, type=int)

    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--static_objects_num", type=int, default=5)
    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_num", type=int, default=70)
    parser.add_argument("--route_num", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.train_set_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_dirs = collect_route_dirs(args.data_path)
    total_routes = len(route_dirs)
    print(f"[START] Found {total_routes} route dirs under {args.data_path}", flush=True)

    all_files = []
    t0 = time.time()
    route_iter = route_dirs
    if tqdm is not None:
        route_iter = tqdm(route_dirs, desc="routes", mininterval=1.0)

    for i, route_dir in enumerate(route_iter, start=1):
        t_route = time.time()
        town = find_town_name(route_dir)
        map_file = resolve_map_file(args.map_path, town)
        if map_file is None:
            raise FileNotFoundError(f"Map file not found for {town} under {args.map_path}")
        if tqdm is None:
            print(f"[{i}/{total_routes}] Processing {route_dir} (town={town})", flush=True)
        files = process_route(route_dir, map_file, args, output_dir)
        all_files.extend(files)
        dt = time.time() - t_route
        if tqdm is None:
            print(f"[{i}/{total_routes}] Done {route_dir} -> {len(files)} samples in {dt:.1f}s (total={len(all_files)})", flush=True)

    with open(args.output_list, "w", encoding="utf-8") as f:
        json.dump(all_files, f, indent=2)

    total_dt = time.time() - t0
    print(f"[END] Saved {len(all_files)} samples to {output_dir} in {total_dt/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
