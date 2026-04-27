import torch
import warnings
import joblib
from typing import Any, Dict, List, Tuple,Set
import logging
import pickle
import math
import os
import matplotlib.pyplot as plt
from os.path import join
import gzip, json, pickle
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString,MultiLineString,Point,MultiPoint,box
from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from shapely.ops import snap,linemerge
from shapely.strtree import STRtree
import cv2
import carla

from scipy import ndimage

CACHE_PATH = ''


def get_static():
    static = np.zeros((5, 10), dtype=np.float32)
    return static

def process_map(map_name):
    full_cache_dir = CACHE_PATH
    cache_path = os.path.join(full_cache_dir, f"{map_name}_HD_map_processed.pkl")
    lane_info,next_lane,crs_info=None,None,None
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        lane_info = cached_data['lane_info']
        next_lane = cached_data['next_lane']
        crs_info = cached_data['crs_info']
        # logger.info(f"✅ Map data loaded from cache: {cache_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load map cache from {cache_path}. Error: {e}") from e
    return lane_info,next_lane,crs_info

def get_map_features(next_traffic_light, route_ids, lane_info, lane_polygons, lane_tree, T, query_xy):
        radius = 120.0
        tls_dict = {0:[0, 0, 1, 0], 1:[1, 0, 0, 0], 2:[0, 1, 0, 0], 3:[0, 0, 0, 1]}
        # 查询ego附近的车道
        x_min, x_max = query_xy[0] - radius, query_xy[0] + radius
        y_min, y_max = query_xy[1] - radius, query_xy[1] + radius
        patch = box(x_min, y_min, x_max, y_max)
        lane_keys=fast_intersection_query_keys(lane_polygons,patch,lane_tree)
        #crs_keys=fast_intersection_query_keys(self.crs_info,patch,self.crs_tree)
        P = 20 # 采样点数
        lanes = np.zeros((70, 20, 12), dtype=np.float32)
        lanes_speed_limit = np.zeros((70, 1), dtype=np.float32)
        lanes_has_speed_limit = np.zeros((70, 1), dtype=np.bool_)
        route_lanes = np.zeros((25, 20, 12), dtype=np.float32)
        route_lanes_speed_limit = np.zeros((25, 1), dtype=np.float32)
        route_lanes_has_speed_limit = np.zeros((25, 1), dtype=np.bool_)
        # 交通灯信息
        light_keys = []
        tl = next_traffic_light
        if tl != None:
            light_location = tl.get_location()
            state = tl.get_state()
            point = [(light_location.x,-light_location.y)]
            light_keys = get_keys_for_points_in_polygons(point,lane_polygons,lane_tree)[0]
        for i,lane_key in enumerate(lane_keys):
            if i >= 70:
                break
            lane=lane_info[lane_key]
            if np.asarray(lane[0].coords).shape==():
                print("empty_laneid:",lane_key,len(lane[0].coords))
            if lane_key in light_keys:
                if state == carla.TrafficLightState.Red:
                    tls_one_hot = [0, 0, 1, 0]
                elif state == carla.TrafficLightState.Yellow:
                    tls_one_hot = [0, 1, 0, 0]
                elif state == carla.TrafficLightState.Green:
                    tls_one_hot = [1, 0, 0, 0]
                elif state == carla.TrafficLightState.Unknown:
                    tls_one_hot = [0, 0, 0, 1]
            else:
                # 如果没被记录就是Unknown状态
                tls_one_hot = [0, 0, 0, 1]
            tls_one_hot = np.array(tls_one_hot, dtype=np.float32) # (4,)
            tls_one_hot = np.tile(tls_one_hot, (20, 1)) # (20, 4)

            centerline=interpolate_polyline(
                np.asarray(lane[0].coords), P + 1
            )
            left_bound=interpolate_polyline(
                np.asarray(lane[1].coords), P 
            )
            right_bound=interpolate_polyline(
                np.asarray(lane[2].coords), P
            )
            # 转换到ego坐标系
            centerline = world_points_to_ego(centerline, T)
            left_bound = world_points_to_ego(left_bound, T)
            right_bound = world_points_to_ego(right_bound, T)
            lanes[i][:, :2] = centerline[:P, :]
            lanes[i][:, 2:4] = np.diff(centerline, axis=0)
            lanes[i][:, 4:6] = left_bound - centerline[:P, :]
            lanes[i][:, 6:8] = right_bound - centerline[:P, :]
            lanes[i][:, 8:12] = tls_one_hot
            lanes_has_speed_limit[i] = False
        # 筛选route_lanes
        count = 0
        for i, lane_key in enumerate(lane_keys):
            if i >= 70:
                break
            if lane_key in route_ids:
                if count >= 25:
                    break
                route_lanes[count] = lanes[i]
                route_lanes_has_speed_limit[count] = False
                count += 1
        return lanes, lanes_speed_limit, lanes_has_speed_limit, route_lanes, route_lanes_speed_limit, route_lanes_has_speed_limit
        
# 给出点集，找到每个点的包含多边形keys集合
def get_keys_for_points_in_polygons(
    points_list,
    polygons_dict,
    tree,
) -> List[List[Any]]:
    if not polygons_dict or not points_list:
        return [[] for _ in points_list]

    # 1. 准备数据和索引
    polygons = list(polygons_dict.values())
    keys = list(polygons_dict.keys())
    results_keys: List[List[Any]] = []
    for point_xy in points_list:
        current_point = Point(point_xy)
        potential_indices = tree.query(current_point)
        point_keys: List[Any] = []
        for index in potential_indices:
            index = int(index)
            poly = polygons[index]
            if not poly.is_valid:
                poly=poly.buffer(0)
            if poly.contains(current_point):
                key = keys[index]
                point_keys.append(key)
        results_keys.append(point_keys)
    return results_keys
# coords:(N, 2) -> (N, 3)

def fast_intersection_query_keys(
    polygons_dict, # 假设 Any 是 Shapely Polygon/Geometry
    target_square,             # 假设 Any 是 Shapely Polygon/Geometry
    tree,
) -> List[Any]:
    if not polygons_dict:
        return []
    polygons = list(polygons_dict.values())
    keys = list(polygons_dict.keys())
    potential_indices = tree.query(target_square)
    intersecting_keys = []
    for index in potential_indices:
        try:
            index = int(index)
        except ValueError:
            continue
        poly = polygons[index]
        if poly.intersects(target_square):
            key = keys[index]
            intersecting_keys.append(key)
    return intersecting_keys

def interpolate_polyline(points: np.ndarray, t: int) -> np.ndarray:
    """copy from av2-api"""

    if points.ndim != 2:
        print("XXXX",points.shape)
        raise ValueError("Input array must be (N,2) or (N,3) in shape.")

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen: np.ndarray = np.linalg.norm(np.diff(points, axis=0), axis=1)  # type: ignore
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc: np.ndarray = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins: np.ndarray = np.digitize(eq_spaced_points, bins=cumarc).astype(int)  # type: ignore

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1  # type: ignore
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    chordlen[tbins - 1] = np.where(
        chordlen[tbins - 1] == 0, chordlen[tbins - 1] + 1e-6, chordlen[tbins - 1]
    )

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp: np.ndarray = anchors + offsets

    return points_interp

def world_points_to_ego(self, points_xy: np.ndarray, world2ego: np.ndarray) -> np.ndarray:
        """
        points_xy: (P, 2) world coords
        world2ego: (4, 4) transform
        return: (P, 2) ego coords
        """
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError(f"points_xy shape should be (P, 2), got {points_xy.shape}")
        if world2ego.shape != (4, 4):
            raise ValueError(f"world2ego shape should be (4, 4), got {world2ego.shape}")

        P = points_xy.shape[0]
        ones = np.ones((P, 1), dtype=points_xy.dtype)
        zeros = np.zeros((P, 1), dtype=points_xy.dtype)
        pts_h = np.hstack([points_xy, zeros, ones])  # (P, 4)

        pts_ego_h = (world2ego @ pts_h.T).T  # (P, 4)
        return pts_ego_h[:, :2]

def create_lane_polygon(left_boundary: LineString, right_boundary: LineString) -> Polygon:
    left_coords = list(left_boundary.coords)
    right_coords = list(right_boundary.coords)
    right_coords_reversed = right_coords[::-1]
    polygon_coords = left_coords + right_coords_reversed
    lane_polygon = Polygon(polygon_coords)
    if not lane_polygon.is_valid:
        lane_polygon=lane_polygon.buffer(0)
    return lane_polygon

def get_lane_ids_from_trajectory(
    dic1,    # {LaneID: Polygon}
    dic2, # {LaneID: [NextLaneID]}
    trajectory, # 轨迹点序列
    lane_tree, # dic1.values()建立的
) -> List[Any]:
    
    # --- 关键修改：数据准备 ---
    all_lanes = list(dic1.values())
    lane_ids = list(dic1.keys()) # 获取所有车道 ID 列表
    
    # 移除 lane_poly_to_id，因为我们将直接使用 lane_ids 列表的索引
        
    lane_ids_sequence = []
    last_lane_id = None
    lane_candidates = []
    # 为一个点找到包含的车道面ids
    def global_search(current_point: Point) -> List[Any]:
        # 注意：这里我们使用 all_lanes 和 lane_ids 列表的索引
        if lane_tree:
            # STRtree.query() 返回的是索引数组
            potential_indices = lane_tree.query(current_point)
            
            found_ids = []
            for index in potential_indices:
                # 1. 通过索引获取多边形
                poly = all_lanes[int(index)]
                
                # 2. 精确测试
                if poly.contains(current_point):
                    # 3. 通过索引获取对应的 LaneID
                    lane_id = lane_ids[int(index)]
                    found_ids.append(lane_id)
            
            return found_ids
        else:
            # 退化为慢速线性搜索 (逻辑不变)
            found_ids = [
                lane_id 
                for lane_id, poly in dic1.items() 
                if poly.contains(current_point)
            ]
            return found_ids
            
    for point_xy in trajectory:
        current_point = Point(point_xy)
        
        # 1. 跳过重复车道 (平滑处理)
        # 注意：这里我们依赖 dic1[last_lane_id] 来获取多边形，假定 last_lane_id 仍然是 Key
        if last_lane_id and dic1[last_lane_id].contains(current_point):
            continue
            
        # 2. 局部搜索：图结构预测下一候选 (逻辑不变)
        if last_lane_id and not lane_candidates:
            lane_candidates = dic2.get(last_lane_id, []) 
            if not lane_candidates:
                last_lane_id = None
                continue
                
        # 3. 局部筛选：缩小候选范围 (逻辑不变)
        filtered_candidates = []
        for lane_id in lane_candidates:
            if dic1[lane_id].contains(current_point):
                filtered_candidates.append(lane_id) 
        lane_candidates = filtered_candidates
        
        # 4. 确认与处理 (逻辑不变)
        if len(lane_candidates) == 1:
            current_lane_id = lane_candidates.pop()
            last_lane_id = current_lane_id
            lane_ids_sequence.append(current_lane_id)
            lane_candidates = []

        elif not lane_candidates:
            # 情况 B: 局部搜索失败，执行全局搜索
            all_found_ids = global_search(current_point)
            
            if len(all_found_ids) == 1:
                current_lane_id = all_found_ids.pop()
                last_lane_id = current_lane_id
                lane_ids_sequence.append(current_lane_id)
            elif all_found_ids:
                lane_candidates = all_found_ids
                
        # 5. 如果 len(lane_candidates) > 1 (歧义)，则保留候选集
        
    return lane_ids_sequence

def load_normalization(arg_path: str):
    # arg_path 是 args.json 路径
    with open(arg_path, "r", encoding="utf-8") as f:
        args = json.load(f)
    norm_path = args.get("normalization_file_path", "normalization.json")
    if not os.path.isabs(norm_path):
        norm_path = os.path.join(os.path.dirname(arg_path), norm_path)
    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)
    return norm

def normalize_inputs(data: dict, norm: dict):
    # 只对 observation_normalizer 里这些字段做标准化
    normed = {}
    for k, v in data.items():
        if k not in norm:
            normed[k] = v
            continue
        mean = np.array(norm[k]["mean"], dtype=np.float32)
        std = np.array(norm[k]["std"], dtype=np.float32)
        x = v.astype(np.float32)
        # mask: 行全 0 的位置不归一化（与原实现一致）
        if x.ndim >= 2:
            mask = np.sum(np.abs(x), axis=-1) == 0
            x = (x - mean) / std
            x[mask] = 0
        else:
            x = (x - mean) / std
        normed[k] = x
    return normed

def to_model_inputs(data: dict, device: str):
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            t = torch.as_tensor(v)
            if t.ndim == 0:
                t = t.view(1)
            t = t.unsqueeze(0)  # batch=1
            out[k] = t.to(device)
    return out

def build_model_inputs(data: dict, arg_path: str, device: str):
    norm = load_normalization(arg_path)
    data_norm = normalize_inputs(data, norm)
    return to_model_inputs(data_norm, device)