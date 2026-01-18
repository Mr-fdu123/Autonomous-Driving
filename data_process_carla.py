import argparse
import json
from pathlib import Path

import numpy as np

from diffusion_dataset import DiffusionDataset


def find_town_name(path: Path) -> str:
    for part in path.parts:
        if part.startswith("Town"):
            return part.split("_")[0]
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
        return []

    data_list = []
    town = find_town_name(route_dir)
    route_name = route_dir.name

    for idx in range(args.past_len - 1, len(dataset) - args.future_len, args.stride):
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

    all_files = []
    for route_dir in collect_route_dirs(args.data_path):
        town = find_town_name(route_dir)
        map_file = resolve_map_file(args.map_path, town)
        if map_file is None:
            raise FileNotFoundError(f"Map file not found for {town} under {args.map_path}")
        all_files.extend(process_route(route_dir, map_file, args, output_dir))

    with open(args.output_list, "w", encoding="utf-8") as f:
        json.dump(all_files, f, indent=2)

    print(f"Saved {len(all_files)} samples to {output_dir}")


if __name__ == "__main__":
    main()
