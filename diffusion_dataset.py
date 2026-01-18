import torch
import warnings
import joblib
from typing import Any, Dict, List, Tuple,Set
import logging
import pickle
import math
import os
from os.path import join
import gzip, json, pickle
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString,MultiLineString,Point,MultiPoint,box
from shapely.ops import snap,linemerge
from shapely.strtree import STRtree
import cv2
from scipy import ndimage

logger = logging.getLogger(__name__)

#索引为2~last_index-8
#长度为last_index-9
def filter_subsets(point_set_list: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
    # 1. 预处理：将原始点集转换为 Shapely 对象并计算动态阈值
    curve_data: List[Dict[str, Any]] = [] # 引入字典数组简化过程
    
    for original_index, points in enumerate(point_set_list):
        if len(points) < 2:
            continue
        try:
            line = LineString(points)
            
            # --- 动态阈值计算核心逻辑 ---
            # 计算相邻点之间的距离
            pts_array = np.array(points) # points: [[1, 2], [1, 3]]
            # 向量化计算：sqrt((x2-x1)^2 + (y2-y1)^2)
            segment_lengths = np.sqrt(np.sum(np.diff(pts_array, axis=0)**2, axis=1))
            avg_spacing = np.mean(segment_lengths)
            # 设置阈值为平均点距的 3.5 倍（取 3-4 倍的中位数）
            dynamic_tolerance = avg_spacing * 3.5
            # ---------------------------

            curve_data.append({
                'line': line,
                'points': points,
                'original_index': original_index,
                'tolerance': dynamic_tolerance
            })
        except Exception:
            continue

    N = len(curve_data)
    is_subset: List[bool] = [False] * N # 要去除的

    for i in range(N):
        if is_subset[i]:
            continue
        
        C_i_data = curve_data[i] # Dict
        C_i_line = C_i_data['line']
        # 使用当前被检查曲线 C_i 的动态阈值
        current_tolerance = C_i_data['tolerance']
        
        # 提取端点
        E_i_1 = Point(C_i_data['points'][0])
        E_i_2 = Point(C_i_data['points'][-1])

        for j in range(N):
            if i == j or is_subset[j]:
                continue 
            
            C_j_line = curve_data[j]['line']
            
            # 1. 快速端点检查
            dist_e1 = E_i_1.distance(C_j_line)
            dist_e2 = E_i_2.distance(C_j_line)
            
            if dist_e1 > current_tolerance or dist_e2 > current_tolerance:
                continue

            # 2. 采样点检查（Hausdorff 距离的简化版）
            max_dist_to_j = 0.0
            # 步长可以根据点数动态调整，采样 1/10 的点进行精细检查
            check_points = C_i_line.coords[::10] 
            
            is_match = True
            for x, y in check_points:
                dist = Point(x, y).distance(C_j_line)
                if dist > current_tolerance:
                    is_match = False
                    break
            
            if is_match:
                is_subset[i] = True
                break

    # 构造结果
    result_list = [point_set_list[curve_data[i]['original_index']] 
                   for i in range(N) if not is_subset[i]]
    return result_list


def calculate_distance(p1, p2):
    """计算两点之间的欧几里得距离."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

# --- 主算法实现 ---

def merge_segments_by_shortest_distance(segment_list):
    """
    将一组有序点集（线段）合并为一条连续的 LineString。
    使用贪心算法，每次选择与当前 LineString 端点距离最近的未连接线段进行连接。
    
    Args:
        segment_list (list[list[tuple]]): 待合并的点集列表，例如 [[(x1, y1), (x2, y2)], ...]。
        
    Returns:
        list[tuple]: 合并后的 LineString。
    """
    if not segment_list:
        return []

    # 1. 初始化
    
    # 转换为可操作的列表副本
    segments = [list(s) for s in segment_list] 
    
    # 1.1 选择第一个线段作为初始 LineString L
    # 选择最长的线段作为起点通常更稳定
    initial_segment = max(segments, key=len)
    segments.remove(initial_segment)
    L = initial_segment

    # 2. 迭代连接 (循环直到所有线段都被合并)
    while segments:
        min_distance = float('inf')
        
        # 存储最佳连接信息: (segment_index, reverse_op, end_op)
        # reverse_op: 是否反转线段 (True/False)
        # end_op: 连接到 L 的哪一端 ('start'/'end')
        best_connection = None 
        
        # L 的四个潜在连接点
        L_start = L[0]
        L_end = L[-1]
        # print(L_start,L_end)
        # 遍历所有未连接的线段
        for i, S_candidate in enumerate(segments):
            S_start = S_candidate[0]
            S_end = S_candidate[-1]
            
            # 计算 4 种连接的可能性及其距离
            
            # 1. L_end -> S_start (不反转 S_candidate, 连接到 L 尾部)
            d1 = calculate_distance(L_end, S_start)
            if d1 < min_distance:
                min_distance = d1
                best_connection = (i, False, 'end') # (索引, 不反转, 接尾)

            # 2. L_end -> S_end (反转 S_candidate, 连接到 L 尾部)
            d2 = calculate_distance(L_end, S_end)
            if d2 < min_distance:
                min_distance = d2
                best_connection = (i, True, 'end') # (索引, 反转, 接尾)

            # 3. L_start <- S_start (反转 L, 连接到 L 头部)
            d3 = calculate_distance(L_start, S_start)
            if d3 < min_distance:
                min_distance = d3
                # (索引, 不反转 S_candidate, 接头)
                # 注意：L_start连接S_start时，S_candidate顺序不变，L反转
                best_connection = (i, True, 'start') 

            # 4. L_start <- S_end (不反转 L, 连接到 L 头部)
            d4 = calculate_distance(L_start, S_end)
            if d4 < min_distance:
                min_distance = d4
                # (索引, 反转 S_candidate, 接头)
                # 注意：L_start连接S_end时，S_candidate需反转
                best_connection = (i, False, 'start')

        # 3. 执行最佳连接
        if best_connection is None:
            # 理论上不应该发生，除非 segments 为空 (循环已退出)
            break 
            
        segment_index, should_reverse, connect_to = best_connection
        
        # 获取最佳线段并从待处理列表中移除
        S_best = segments.pop(segment_index)
        
        # 准备 S_best: 如果需要，反转
        if should_reverse:
            S_best.reverse()
        
        # print(S_best[0],S_best[-1])
        # 合并到 LineString L
        if connect_to == 'end':
            # print("end")
            # L = L + S_best
            L.extend(S_best) # 增加两个点
        elif connect_to == 'start':
            # print("start")
            # L = S_best + L
            L = S_best + L

    # print(L[0],L[-1])
    return LineString(L)


def _find_closest_centerline_point_and_direction_vector(centerline_points, boundary_point):
    C = np.array(centerline_points) # [[1, 3], [2, 3]]
    P_B = np.array(boundary_point)
    num_centerline = C.shape[0]
    if num_centerline < 2:
        return None, None, None 
    distances_sq = np.sum((C - P_B) ** 2, axis=1)
    i = np.argmin(distances_sq)
    C_closest = C[i]
    if i < num_centerline - 1:
        D = C[i+1] - C[i]
    else:
        D = C[i] - C[i-1]
    return C_closest, i, D

# 判断边界线在中心线左侧或右侧或在中心线上
def _determine_single_segment_side(centerline_points, boundary_segment_points, tolerance=1e-6):
    boundary_points = np.array(boundary_segment_points)
    num_points = boundary_points.shape[0]
    if num_points == 1:
        indices_to_check = [0]
    elif num_points == 2:
        indices_to_check = [0, 1]
    else:
        indices_to_check = [0, num_points // 2, num_points - 1] 
    key_points = boundary_points[indices_to_check]
    side_scores = []
    for P_B in key_points:
        C_closest, i, D = _find_closest_centerline_point_and_direction_vector(centerline_points, P_B)
        if D is None:
            continue
        d_x, d_y = D
        N_right = np.array([d_y, -d_x])
        V = P_B - C_closest
        S = np.dot(V, N_right)
        side_scores.append(S)
    avg_score = np.mean(side_scores)
    if avg_score > tolerance:
        return 'Right'
    elif avg_score < -tolerance:
        return 'Left'
    else:
        return 'Ambiguous'


def classify_multiple_boundary_segments(centerline_geom, list_of_boundary_segments, tolerance=1e-6):
    centerline_points = np.array(centerline_geom.coords)
    if centerline_points.shape[0] < 2:
        return ['Error: Insufficient Centerline Points'] * len(list_of_boundary_segments)
    results = []
    for segment_points in list_of_boundary_segments:
        result = _determine_single_segment_side(centerline_points, segment_points, tolerance)
        results.append(result)
    return results

# 修正车道线，不能反向
def align_laneline_order(L: LineString, R: LineString) -> Tuple[LineString, LineString]:
    if not L.coords or not R.coords:
        print("警告: 至少一条车道线为空。")
        return L, R
    L_start = Point(L.coords[0])
    L_end = Point(L.coords[-1])
    R_start = Point(R.coords[0])
    R_end = Point(R.coords[-1])
    S1 = LineString([L_start, R_start])
    S2 = LineString([L_end, R_end])
    are_intersecting = S1.intersects(S2)
    if are_intersecting:
        R_coords_reversed = list(R.coords)[::-1]
        R_aligned = LineString(R_coords_reversed)
        return L, R_aligned
    else:
        return L, R


def create_lane_polygon(left_boundary: LineString, right_boundary: LineString) -> Polygon:
    left_coords = list(left_boundary.coords)
    right_coords = list(right_boundary.coords)
    right_coords_reversed = right_coords[::-1] # 从尾到头读取数列
    polygon_coords = left_coords + right_coords_reversed
    lane_polygon = Polygon(polygon_coords)
    if not lane_polygon.is_valid:
        lane_polygon=lane_polygon.buffer(0) # 合法检查
    return lane_polygon


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
def calculate_and_extend_with_heading_list(coords) -> np.ndarray:
    N = len(coords)
    if N < 2:
        if N == 1:
            return np.append(coords, [[0.0]], axis=1) 
        return np.array([]) 
    delta = np.diff(coords, axis=0)
    segment_headings = np.arctan2(delta[:, 1], delta[:, 0])
    point_headings = np.append(segment_headings, segment_headings[-1])
    heading_column = point_headings[:, np.newaxis] 
    extended_coords = np.hstack((coords, heading_column))
    return extended_coords

from shapely.geometry import Point
from typing import Dict, Any, List, Tuple
from shapely.strtree import STRtree
# 假设 LaneID, Polygon 等类型已在其他地方定义

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

class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, log_file_path: str, map_file_path: str, cache_dir: str,
                 agent_num=32, lane_num=70, lane_len=20, route_num=25, static_objects_num=5):
        """
        :param log_file_path: 驾驶信息文件（如日志或传感器数据）的路径。
        :param map_file_path: 地图文件（如 HD Map 数据）的路径。
        """

        self.log_file_path = log_file_path
        self.map_file_path = map_file_path
        self._cache_dir=cache_dir
        self._log_file=[]
        self.load_file()
        self.agent_num = agent_num
        self.lane_num = lane_num
        self.lane_len = lane_len
        self.route_num = route_num
        self.static_objects_num = static_objects_num
        self.lane_info={} # [center_ls,lbound_ls,rbound_ls], 后两个包围成车道面
        self.crs_info={}  # Polygon(crs_points)
        self.next_lane={} # [(r_id, l_id), ]
        #self.tls={} # {(r_id, l_id): [0, 0, 1, 0]}
        self.trajectory=[]
        self.route_ids=None
        self.get_trajectory()
        self.process_map()
        self.lane_polygons={key:create_lane_polygon(lane[1],lane[2]) for key,lane in self.lane_info.items()}
        self.lane_tree=STRtree(list(self.lane_polygons.values()))
        self.crs_tree=STRtree(list(self.crs_info.values()))
        self.route_ids=get_lane_ids_from_trajectory(self.lane_polygons,self.next_lane,self.trajectory,self.lane_tree)
        self.radius=120.0
        # 交通灯信息
        self.tls_dict = {0:[0, 0, 1, 0], 1:[1, 0, 0, 0], 2:[0, 1, 0, 0], 3:[0, 0, 0, 1]}

    @staticmethod
    def _find_ego_bbox(anno: Dict) -> Dict:
        for box in anno.get('bounding_boxes', []):
            if box.get('class') == 'ego_vehicle':
                return box
        return {}

    @staticmethod
    def _to_right_hand_transform(w2e: np.ndarray) -> np.ndarray:
        # Convert left-hand CARLA world2ego to right-hand coordinates (flip y).
        flip = np.diag([1.0, -1.0, 1.0, 1.0]).astype(w2e.dtype)
        return flip @ w2e @ flip

    def load_file(self):
        for ann_name in sorted(os.listdir(join(self.log_file_path,'anno')),key= lambda x: int(x.split('.')[0])):
            self._log_file.append(join(self.log_file_path,'anno',ann_name))
    
    def __len__(self) -> int:
        return len(self._log_file)

    def __getitem__(self, idx: int) -> Dict:
       data = self.build_features(idx)
       return data

    
    def build_features(self, idx: int) -> Dict:
        cache_path = None
        if self._cache_dir:
            folder_name = os.path.basename(self.log_file_path)
            # 仅修改后缀名以便识别新格式
            cache_filename = f"{folder_name}_idx{idx}.joblib"
            
            full_cache_dir = os.path.join(self._cache_dir, folder_name)
            if not os.path.exists(full_cache_dir):
                os.makedirs(full_cache_dir, exist_ok=True)
            
            cache_path = os.path.join(full_cache_dir, cache_filename)
        
        # 2. 检查缓存
        if cache_path and os.path.exists(cache_path):
            try:
                # 命中缓存：使用 joblib 加载，它会自动恢复原始的 numpy 结构和 dtype
                data = joblib.load(cache_path)
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache from {cache_path}: {e}. Recalculating features.")

        with gzip.open(self._log_file[idx], 'rt', encoding='utf-8') as gz_file:
            anno = json.load(gz_file)
        
        # diffusion planner用的自车坐标系，得到转化矩阵
        ego_box = self._find_ego_bbox(anno)
        if not ego_box:
            raise ValueError(f"ego_vehicle not found in {self._log_file[idx]}")
        world2ego = np.array(ego_box['world2ego'], dtype=np.float32)
        world2ego = self._to_right_hand_transform(world2ego)
        rot = world2ego[:3, :3]

        
        data = {}
        speed = float(anno.get('speed', 0.0))
        yaw_lh = float(anno.get('theta', 0.0))
        yaw_rh = -yaw_lh
        v_world = np.array([speed * math.cos(yaw_rh), speed * math.sin(yaw_rh), 0.0], dtype=np.float32)
        v_ego = rot @ v_world

        accel = anno.get('acceleration', [0.0, 0.0, 0.0])
        a_world = np.array([accel[0], -accel[1], accel[2]], dtype=np.float32)
        a_ego = rot @ a_world

        steer = float(anno.get('steer', 0.0))
        yaw_rate = float(anno.get('angular_velocity', [0.0, 0.0, 0.0])[2])
        yaw_rate = -yaw_rate

        data['ego_current_state'] = np.array([0.0, 0.0, 1.0, 0.0,
                                       v_ego[0], v_ego[1], a_ego[0], a_ego[1],
                                       steer, yaw_rate], dtype=np.float32)
        ego_agent_future, agent_features_past, agent_features_future = self.get_agent_features(idx)
        data['ego_agent_future'] = ego_agent_future
        data['neighbor_agents_past'] = agent_features_past
        data['neighbor_agents_future'] = agent_features_future
        data['static_objects'] = self.get_static()
        ego_center = ego_box.get('center') or ego_box.get('location')
        q_xy = (ego_center[0], -ego_center[1])
        lanes, lanes_speed_limit, lanes_has_speed_limit, route_lanes, route_lanes_speed_limit, route_lanes_has_speed_limit=self.get_map_features(idx, q_xy)
        data['lanes'] = lanes
        data['lanes_speed_limit'] = lanes_speed_limit
        data['lanes_has_speed_limit'] = lanes_has_speed_limit
        data['route_lanes'] = route_lanes
        data['route_lanes_speed_limit'] = route_lanes_speed_limit
        data['route_lanes_has_speed_limit'] = route_lanes_has_speed_limit

        # 3. 存储缓存
        if cache_path:
            try:
                # 仅修改存储方式：使用 joblib 并启用 zstd 压缩
                # 这不会改变数据内容，只会改变磁盘上的二进制排列格式
                joblib.dump(data, cache_path, compress='lz4')
                logger.info(f"Cache saved to {cache_path}")
            except Exception as e:
                logger.error(f"Failed to save cache for {cache_path}: {e}")

        # final_data = PlutoFeature.normalize(data, first_time=True, radius=100.0)
        return data

    def get_trajectory(self) -> None:
        cache_path = None
        if self._cache_dir and self._log_file:
            folder_name = os.path.basename(self.log_file_path)
            full_cache_dir = os.path.join(self._cache_dir, "AAtrajectory_cache")
            if not os.path.exists(full_cache_dir):
                os.makedirs(full_cache_dir)
            cache_path = os.path.join(full_cache_dir, f"{folder_name}_trajectory.pkl")
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.trajectory = pickle.load(f)
                logger.info(f"✅ Trajectory loaded from cache: {cache_path}")
                return # 缓存命中，直接返回
            except Exception as e:
                logger.warning(f"Failed to load trajectory cache from {cache_path}: {e}. Recalculating trajectory.")
        # logger.info(f"⏳ Processing trajectory from {len(self._log_file)} log files...")
        self.trajectory = [] 
        for file_path in self._log_file:
            with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
                anno = json.load(gz_file)
                ego_box = self._find_ego_bbox(anno)
                if not ego_box:
                    continue
                ego_loc = ego_box.get('location') or ego_box.get('center')
                self.trajectory.append((ego_loc[0], -ego_loc[1]))
        if cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.trajectory, f)
                logger.info(f"💾 Trajectory successfully cached to {cache_path}")
            except Exception as e:
                logger.error(f"Failed to save trajectory cache for {cache_path}: {e}")

    def get_agent_features(self,idx):
        T=101
        N = self.agent_num
        ego_agent_future = np.zeros((80, 3), dtype=np.float32)
        agent_features_past = np.zeros((N, 21, 11), dtype=np.float32)
        agent_features_future = np.zeros((N, 80, 3), dtype=np.float32)
        agent_list=[] # 存储agent dict(no traffic)
        id_list=[]
        id_dict={}
        ego_id=None
        w2e = None
        rot = None
        with gzip.open(self._log_file[idx], 'rt', encoding='utf-8') as gz_file:
            anno = json.load(gz_file)
            for dic in anno['bounding_boxes']:
                if dic['class'] == 'ego_vehicle':
                    ego_id = dic['id']
                    # 得到世界->自车坐标系转化矩阵
                    world2ego = dic['world2ego']
                    w2e = np.array(world2ego, dtype=np.float32)
                    w2e = self._to_right_hand_transform(w2e)
                    rot = w2e[:3, :3]
                    continue
                if dic['class']=='traffic_light':continue
                if dic['class']=='traffic_sign':continue
                id = dic['id']
                if id not in id_list:
                    id_list.append(id)
                    agent_list.append(dic)
        if w2e is None:
            raise ValueError(f"ego_vehicle not found in {self._log_file[idx]}")
        agent_list=sorted(agent_list, key=lambda x: x['distance'])[:N]
        for i,agent in enumerate(agent_list):
            id_dict[agent['id']]=i
        st=max(0,idx-20)
        ed=min(len(self._log_file),idx+81)
  
        for t in range(st, ed):
            with gzip.open(self._log_file[t], 'rt', encoding='utf-8') as gz_file:
                anno = json.load(gz_file)
                for dic in anno['bounding_boxes']:
                    id = dic['id']
                    if (id == ego_id):
                        tt = t-idx+20
                        p_w = np.array([dic['location'][0], -dic['location'][1], dic['location'][2], 1.0], dtype=np.float32)
                        p_ego = w2e @ p_w.T
                        x = p_ego[0]
                        y = p_ego[1]
                        heading_w = -math.radians(dic['rotation'][2])
                        h_w = np.array([np.cos(heading_w), np.sin(heading_w), 0.0], dtype=np.float32)
                        h_ego = rot @ h_w
                        heading_ego = np.arctan2(h_ego[1], h_ego[0])
                        future_idx = t - idx - 1
                        if 0 <= future_idx < 80:
                            ego_agent_future[future_idx] = np.array([x, y, heading_ego], dtype=np.float32)
                        continue
                    if (id not in id_list):
                        continue
                    rid=id_dict[id]
                    tt = t-idx+20
                    p_w = np.array([dic['location'][0], -dic['location'][1], dic['location'][2], 1.0], dtype=np.float32)
                    p_ego = w2e @ p_w.T
                    x = p_ego[0]
                    y = p_ego[1]
                    speed = float(dic.get('speed', 0.0))
                    yaw_rh = -math.radians(dic['rotation'][2])
                    h_w = np.array([np.cos(yaw_rh), np.sin(yaw_rh), 0.0], dtype=np.float32)
                    h_ego = rot @ h_w
                    heading_ego = np.arctan2(h_ego[1], h_ego[0])
                    v_world = np.array([speed * math.cos(yaw_rh), speed * math.sin(yaw_rh), 0.0], dtype=np.float32)
                    v_ego = rot @ v_world
                    vx = v_ego[0]
                    vy = v_ego[1]
                    width = dic['extent'][1] * 2
                    length = dic['extent'][0] * 2
                    # 计算未来80帧
                    if t >= idx+1:
                        agent_features_future[rid, tt-21] = np.array([x, y, heading_ego], dtype=np.float32)
                    # 计算过去和现在21帧
                    if dic['class'] == 'vehicle' and t <= idx:
                        agent_features_past[rid, tt] = np.array([x, y, np.cos(heading_ego), np.sin(heading_ego),
                                                     vx, vy, width, length,
                                                     1.0, 0.0, 0.0], dtype=np.float32)
                    elif dic['class'] == 'walker' and t <= idx:
                        agent_features_past[rid, tt] = np.array([x, y, np.cos(heading_ego), np.sin(heading_ego),
                                                     vx, vy, width, length,
                                                     0.0, 1.0, 0.0], dtype=np.float32)
                    elif dic['class'] == 'bicycle' and t <= idx:
                        agent_features_past[rid, tt] = np.array([x, y, np.cos(heading_ego), np.sin(heading_ego),
                                                     vx, vy, width, length,
                                                     0.0, 0.0, 1.0], dtype=np.float32)
        return ego_agent_future, agent_features_past, agent_features_future
    
    # 获取每一帧的交通灯信号
    def process_tls(self, idx) -> Dict:
        tls = {}
        with gzip.open(self._log_file[idx], 'rt', encoding='utf-8') as gz_file:
            anno = json.load(gz_file)
            for dic in anno['bounding_boxes']:
                if dic['class']=='traffic_light' and dic['affects_ego']==True:
                    point=[(dic['trigger_volume_location'][0],-dic['trigger_volume_location'][1])]
                    lane_keys=get_keys_for_points_in_polygons(point,self.lane_polygons,self.lane_tree)[0]
                    for lane_key in lane_keys:
                        tls[lane_key] = self.tls_dict[dic['state']]
        return tls

    def get_static(self):
        static_objects = np.zeros((self.static_objects_num, 10), dtype=np.float32)
        return static_objects
    
    def _get_crosswalk_edges(
        self, crosswalk, sample_points: int = 21
    ):
        bbox = shapely.minimum_rotated_rectangle(crosswalk)
        coords = np.stack(bbox.exterior.coords.xy, axis=-1)
        edge1 = coords[[3, 0]]  # right boundary
        edge2 = coords[[2, 1]]  # left boundary

        edges = np.stack([(edge1 + edge2) * 0.5, edge2, edge1], axis=0)  # [3, 2, 2]
        vector = edges[:, 1] - edges[:, 0]  # [3, 2]
        steps = np.linspace(0, 1, sample_points, endpoint=True)[None, :]
        points = edges[:, 0][:, None, :] + vector[:, None, :] * steps[:, :, None]

        return points
    
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

    # 输入ego location, 提取周围地图特征
    def get_map_features(self, idx, query_xy):
        # 获得转换矩阵
        w2e = None
        with gzip.open(self._log_file[idx], 'rt', encoding='utf-8') as gz_file:
            anno = json.load(gz_file)
            for dic in anno['bounding_boxes']:
                if dic['class'] == 'ego_vehicle':
                    world2ego = dic['world2ego']
                    w2e = np.array(world2ego, dtype=np.float32)
                    w2e = self._to_right_hand_transform(w2e)
                    break
        if w2e is None:
            raise ValueError(f"ego_vehicle not found in {self._log_file[idx]}")
        # 查询ego附近的车道
        x_min, x_max = query_xy[0] - self.radius, query_xy[0] + self.radius
        y_min, y_max = query_xy[1] - self.radius, query_xy[1] + self.radius
        patch = box(x_min, y_min, x_max, y_max)
        lane_keys=fast_intersection_query_keys(self.lane_polygons,patch,self.lane_tree)
        crs_keys=fast_intersection_query_keys(self.crs_info,patch,self.crs_tree)
        P = self.lane_len # 采样点数
        lanes = np.zeros((self.lane_num, self.lane_len, 12), dtype=np.float32)
        lanes_speed_limit = np.zeros((self.lane_num, 1), dtype=np.float32)
        lanes_has_speed_limit = np.zeros((self.lane_num, 1), dtype=np.bool_)
        route_lanes = np.zeros((self.route_num, self.lane_len, 12), dtype=np.float32)
        route_lanes_speed_limit = np.zeros((self.route_num, 1), dtype=np.float32)
        route_lanes_has_speed_limit = np.zeros((self.route_num, 1), dtype=np.bool_)
        # 交通灯信息
        tls = self.process_tls(idx)
        for i,lane_key in enumerate(lane_keys):
            if i >= self.lane_num:
                break
            lane=self.lane_info[lane_key]
            if np.asarray(lane[0].coords).shape==():
                print("empty_laneid:",lane_key,len(lane[0].coords))
            if lane_key in tls.keys():
                tls_one_hot = tls[lane_key]
            else:
                # 如果没被记录就是Unknown状态
                tls_one_hot = [0, 0, 0, 1]
            tls_one_hot = np.array(tls_one_hot, dtype=np.float32) # (4,)
            tls_one_hot = np.tile(tls_one_hot, (self.lane_len, 1)) # (20, 4)
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
            centerline = self.world_points_to_ego(centerline, w2e)
            left_bound = self.world_points_to_ego(left_bound, w2e)
            right_bound = self.world_points_to_ego(right_bound, w2e)
            lanes[i][:, :2] = centerline[:P, :]
            lanes[i][:, 2:4] = np.diff(centerline, axis=0)
            lanes[i][:, 4:6] = left_bound - centerline[:P, :]
            lanes[i][:, 6:8] = right_bound - centerline[:P, :]
            lanes[i][:, 8:12] = tls_one_hot
            lanes_has_speed_limit[i] = False
        # 筛选route_lanes
        count = 0
        for i, lane_key in enumerate(lane_keys):
            if lane_key in self.route_ids:
                route_lanes[count] = lanes[i]
                route_lanes_has_speed_limit[count] = False
                count += 1
            if count >= self.route_num:
                break
        return lanes, lanes_speed_limit, lanes_has_speed_limit, route_lanes, route_lanes_speed_limit, route_lanes_has_speed_limit
        

    def process_map(self):
        cache_path = None
        if self._cache_dir:
            # 使用 map_file_path 的基础名 (townid_HD_map) 作为缓存ID
            map_filename = os.path.basename(self.map_file_path)
            map_id = os.path.splitext(map_filename)[0] 
            full_cache_dir = os.path.join(self._cache_dir, "AAmap_cache")
            if not os.path.exists(full_cache_dir):
                os.makedirs(full_cache_dir)
            cache_path = os.path.join(full_cache_dir, f"{map_id}_processed.pkl")
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                self.lane_info = cached_data['lane_info']
                self.next_lane = cached_data['next_lane']
                self.crs_info = cached_data['crs_info']
                logger.info(f"✅ Map data loaded from cache: {cache_path}")
                return # 缓存命中，直接返回
            except Exception as e:
                logger.warning(f"Failed to load map cache from {cache_path}: {e}. Recalculating map features.")
        # logger.info(f"⏳ Processing map file (no cache): {self.map_file_path}")
        # 执行原始地图处理逻辑，并将结果保存在 self 属性中
        self.process_map_logic() 
        if cache_path:
            try:
                data_to_cache = {
                    'lane_info': self.lane_info,
                    'next_lane': self.next_lane,
                    'crs_info': self.crs_info,
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(data_to_cache, f)
                logger.info(f"💾 Map data successfully cached to {cache_path}")
            except Exception as e:
                logger.error(f"Failed to save map cache for {self.map_file_path}: {e}")
        
    
    def process_map_logic(self):
        dic1={}
        with np.load(self.map_file_path, allow_pickle=True) as data:
            arr_data = data['arr']
            for item in arr_data:
                for lane_id,volume in item[1].items():
                    ##lane
                    road_id=item[0]
                    # print(road_id)
                    if lane_id!='Trigger_Volumes':
                        points_c=[]
                        points_b=[]
                        self.next_lane[(item[0],lane_id)]=[]
                        for single_lane in volume: # 每一段
                            points=[]
                            for point in single_lane['Points']: # point = ((x, y, z), (roll, pitch, yaw), flag)
                                points.append((point[0][0],-point[0][1]))
                            # if len(points)>50000:points=points[::10]
                            sc=len(points)//500
                            if sc > 1:
                                last_point = points[-1]  # 记录原点集最后一个点
                                points = points[::sc]    # 步长采样
                                if points[-1] != last_point: 
                                    points.append(last_point) # 补上最后一个点
                            if single_lane['Type']=='Center':points_c.append(points)
                            else:points_b.append(points)
                            if 'Topology' in single_lane:
                                for nxt_id in single_lane['Topology']:
                                    if nxt_id!=(item[0],lane_id) and nxt_id not in self.next_lane[(item[0],lane_id)]:
                                        self.next_lane[(item[0],lane_id)].append(nxt_id)
                        points_c=filter_subsets(points_c)
                        if len(points_c)==0:continue
                        center_ls=merge_segments_by_shortest_distance(points_c)
                        classfi=classify_multiple_boundary_segments(center_ls,points_b)
                        points_l=[] # 左侧边界线
                        points_r=[] # 右侧边界线
                        for points,label in zip(points_b,classfi):
                            if label=='Left':points_l.append(points)
                            elif label=='Right':points_r.append(points)
                        points_l=filter_subsets(points_l)
                        points_r=filter_subsets(points_r)
                        # if item[0]==539 and lane_id==-1:
                        #     for point in points_l:
                        #         print(point[0],point[-1])
                        #     for point in points_r:
                        #         print(point[0],point[-1])
                        if len(points_l)==0:print("l",item[0],lane_id,len(points_l),len(points_r),len(points_b))   
                        if len(points_r)==0:print("r",item[0],lane_id,len(points_r))   
                        lbound_ls=merge_segments_by_shortest_distance(points_l)
                        rbound_ls=merge_segments_by_shortest_distance(points_r)
                        lbound_ls,rbound_ls=align_laneline_order(lbound_ls,rbound_ls)
                        self.lane_info[(road_id,lane_id)]=[center_ls,lbound_ls,rbound_ls]#这样存储好吗，有更快的方法吗
                        dic1[(item[0],lane_id)]=create_lane_polygon(lbound_ls,rbound_ls)
                    #crosswalk
                    else:
                        for sec_id,single_lane in enumerate(volume):
                            if single_lane['Type']!='TrafficLight':
                                crs_points=[(point[0],-point[1]) for point in single_lane['Points']]
                                self.crs_info[(road_id,lane_id,sec_id)]=Polygon(crs_points)
        # self.route_ids=get_lane_ids_from_trajectory(dic1,self.next_lane,self.trajectory)

