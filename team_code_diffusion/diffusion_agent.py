import os
import json
import datetime
import diffusion_tool as tool
import pathlib
import time
import cv2
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString,MultiLineString,Point,MultiPoint,box
from shapely.ops import snap,linemerge
from shapely.strtree import STRtree
from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from collections import deque
import math
from collections import OrderedDict
import torch
import carla
from scipy.interpolate import splprep, splev
from config import GlobalConfig
import numpy as np
from PIL import Image
from privileged_route_planner import PrivilegedRoutePlanner
from pid_controller import PIDController
from planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent
from pyquaternion import Quaternion
from scipy.optimize import fsolve
# Diffusion Planner
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

SAVE_PATH = '/data/swyl/pluto-eval'

IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

def get_entry_point():
    return 'DiffusionAgent'

class DiffusionAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = deque()
        self.pidcontroller = PIDController() 
        self.history_buffer = deque(maxlen=20)
        self.arg_path = path_to_conf_file.split('+')[0] #arg
        self.ckpt_path = path_to_conf_file.split('+')[1] #ckpt
        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
        else:
            now = datetime.datetime.now()
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        #cfg = OmegaConf.load(self.config_path)
        # 加载Diffusion Planner
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dp_config = Config(self.arg_path, guidance_fn=None)
        self.model = Diffusion_Planner(self.dp_config).to(self.device)
        self.model.eval()
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            state = torch.load(self.ckpt_path, map_location=self.device)
            state = state.get("ema_state_dict", state.get("model", state))
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=False)

        self.trajectory_history = []
        self.future_trajectories = [] # 新增：用于存储前10步预测
        self.config = GlobalConfig()#route config
        self.world_map = CarlaDataProvider.get_map()
        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        print("map_name",map_name)
        self.lane_info,self.next_lane,self.crs_info=tool.process_map(map_name)
        self.lane_polygons={key:tool.create_lane_polygon(lane[1],lane[2]) for key,lane in self.lane_info.items()}
        self.lane_tree=STRtree(list(self.lane_polygons.values()))
        self.crs_tree=STRtree(list(self.crs_info.values()))
        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        self.last_steers = deque()
        self.lat_ref, self.lon_ref = 42.0, 2.0
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0 
        self.prev_control = control
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            # string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string = self.save_name
            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'rgb_front_right').mkdir()
            (self.save_path / 'rgb_front_left').mkdir()
            (self.save_path / 'rgb_back').mkdir()
            (self.save_path / 'rgb_back_right').mkdir()
            (self.save_path / 'rgb_back_left').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()
        self.coor2topdown = np.array([[1.0,  0.0,  0.0,  0.0], 
                                      [0.0, -1.0,  0.0,  0.0], 
                                      [0.0,  0.0, -1.0, 50.0], 
                                      [0.0,  0.0,  0.0,  1.0]])
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown 
    
    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                # x: lat_ref, y: lon_ref
                x, y = vars
                # 先计算当前参考纬度下的 scale
                scale = math.cos(x * math.pi / 180.0)
                
                # eq1: 经度方向 (X)
                # 逻辑：locx = (lon - y) * (pi * R * scale) / 180
                eq1 = (lon - y) * (math.pi * EARTH_RADIUS_EQUA * scale) / 180.0 - locx
                
                # eq2: 纬度方向 (Y)
                # 逻辑：locy = (my_ref - my_curr)  [注意：my 不带 scale]
                # my_ref = log(tan(pi/4 + x/2)) * R
                # my_curr = log(tan(pi/4 + lat/2)) * R
                term_ref = math.log(math.tan((90 + x) * math.pi / 360.0))
                term_curr = math.log(math.tan((90 + lat) * math.pi / 360.0))
                eq2 = (term_ref - term_curr) * EARTH_RADIUS_EQUA + locy 
                
                return [eq1, eq2]
            initial_guess = [lat, lon]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0      
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        # print("global_eg",self._global_plan[0])  
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()

        # Check if the vehicle starts from a parking spot
        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(self._vehicle.get_location())
        # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
        # ego vehicle at the beginning.
        starts_with_parking_exit = distance_to_road > 2

        # Set up the route planner and extrapolation
        self._waypoint_planner = PrivilegedRoutePlanner(self.config)
        self._waypoint_planner.setup_route(self.org_dense_route_world_coord, self._world, self.world_map,
                                           starts_with_parking_exit, self._vehicle.get_location())
        self._waypoint_planner.save()
        trajectory=self._waypoint_planner.original_route_points
        rp=self._waypoint_planner.route_points

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.plot(rp[:, 0], rp[:, 1], 'b-', label='Route Points') 
        plt.scatter(rp[0, 0], rp[0, 1], color='green', label='Start') # 起点
        plt.scatter(rp[-1, 0], rp[-1, 1], color='red', label='End')    # 终点
        plt.title("Visualized Route Points (Top View)")
        plt.xlabel("X Coordinate (meters)")
        plt.ylabel("Y Coordinate (meters)")
        plt.axis('equal') # 保持坐标轴比例一致，防止地图变形
        plt.grid(True)
        plt.legend()
        save_path = "/data/swyl/pluto-eval/route_visualization.png"
        plt.savefig(save_path)
        print(f"RP已成功保存至: {save_path}")
        plt.close() # 关闭画布释放内存

        trajectory = [(float(item[0]), -float(item[1])) for item in trajectory]
        with open('/data/swyl/Bench2Drive-main/data.txt', 'w', encoding='utf-8') as f:
            f.write('start\n')
            for item in trajectory:
                # print(item)
                f.write(str(item) + '\n')  # 记得手动添加换行符 \n
        self.route_ids=tool.get_lane_ids_from_trajectory(self.lane_polygons,self.next_lane,trajectory,self.lane_tree)
        print("route_ids",self.route_ids)
        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        sensors =[
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                {	
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 50.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 512, 'height': 512, 'fov': 5 * 10.0,
                    'id': 'bev'
                }
            ]
        return sensors
    def tick(self, input_data):
        self.step += 1
        gps = input_data['GPS'][1][:2]
        # speed = input_data['SPEED'][1]['speed']
        # compass = input_data['IMU'][1][-1]
        vehicle_transform = self._vehicle.get_transform()
        vehicle_velocity = self._vehicle.get_velocity()
        speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
        speed = np.array(speed)
        compass_deg = vehicle_transform.rotation.yaw
        compass = math.radians(compass_deg)
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        pos = self.gps_to_location(gps)
        position = self._vehicle.get_location()
        print("position",position)
        print("compass",compass)
        # print("pos",pos)
        near_node, near_command = self._route_planner.run_step(np.array([position.x, position.y]))
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'gps': gps,
                'pos':position,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node
                }
        
        return result
    
    def get_shapely_polygon(self,actor):
        # 1. 获取车辆的变换矩阵和 Bounding Box
        bbox = actor.bounding_box
        t = actor.get_transform()
        # 我们只需要平面的四个角：(前左, 前右, 后右, 后左)
        p1 = carla.Location(x=+bbox.extent.x, y=+bbox.extent.y)
        p2 = carla.Location(x=+bbox.extent.x, y=-bbox.extent.y)
        p3 = carla.Location(x=-bbox.extent.x, y=-bbox.extent.y)
        p4 = carla.Location(x=-bbox.extent.x, y=+bbox.extent.y)
        
        # 3. 将局部坐标点加上 Bounding Box 的偏移 (bbox.location)
        # 有些车辆的 bbox 中心并不在 actor 原点
        p1 += bbox.location
        p2 += bbox.location
        p3 += bbox.location
        p4 += bbox.location

        # 4. 转换到世界坐标系
        # 使用 transform 将局部坐标映射到世界坐标
        world_p1 = t.transform(p1)
        world_p2 = t.transform(p2)
        world_p3 = t.transform(p3)
        world_p4 = t.transform(p4)
        
        # 5. 提取 x, y 构造 shapely 多边形
        vertices = [
            (world_p1.x, -world_p1.y),
            (world_p2.x, -world_p2.y),
            (world_p3.x, -world_p3.y),
            (world_p4.x, -world_p4.y)
        ]
        
        return Polygon(vertices)

    def is_two_wheeler(self,type_id):
        type_id = type_id.lower()
        keywords = [
            'bicycle', 'bike', 'gazelle', 'diamondback', 'century', 
            'ninja', 'harley', 'low_rider', 'yzf', 'zx125', 'vespa'
        ]
        return 'vehicle' in type_id and any(key in type_id for key in keywords)

    
    def world_to_ego_matrix_rh(self, x: float, y: float, heading: float) -> np.ndarray:
        """
        Build world->ego 4x4 transform matrix in right-handed coordinates.
        Inputs are ego pose in world RH frame: (x, y, heading).
        """
        cos_yaw = math.cos(heading)
        sin_yaw = math.sin(heading)
        return np.array(
            [
                [ cos_yaw,  sin_yaw, 0.0, -(cos_yaw * x + sin_yaw * y)],
                [-sin_yaw,  cos_yaw, 0.0, -(-sin_yaw * x + cos_yaw * y)],
                [ 0.0,      0.0,     1.0,  0.0],
                [ 0.0,      0.0,     0.0,  1.0],
            ],
            dtype=np.float32,
        )
    
    def get_agent_features(self, agent_num, T):
        # 获取自车信息
        hero_vehicle = CarlaDataProvider.get_hero_actor()
        if hero_vehicle is None:
            raise RuntimeError(
                "Critical Error: 'hero' vehicle not found in CarlaDataProvider. "
                "Please ensure the ego vehicle is spawned with role_name='hero'."
            )

        hero_location = CarlaDataProvider.get_location(hero_vehicle)
        
        # 2. 收集所有车辆及距离
        actor_distance_list = []
        for actor in CarlaDataProvider.get_all_actors():
            if actor.id != hero_vehicle.id and ('vehicle' in actor.type_id or 'walker' in actor.type_id):
                loc = CarlaDataProvider.get_location(actor)
                if loc:
                    dist = hero_location.distance(loc)
                    actor_distance_list.append((actor, dist))
        actor_distance_list.append((hero_vehicle,0.0))
        actor_distance_list.sort(key=lambda x: x[1])
        # 找到最近的agent_num个
        close_actor = actor_distance_list[1:agent_num+1]
        N = agent_num
        T = 21
        features = np.zeros((N, T, 11), dtype=np.float32)
        current_frame = []
        id_dict_all = {}
        id_dict_close = {}
        # 记录最近的agent
        for i, (actor, dist) in enumerate(close_actor):
            id_dict_close[actor.id] = i
        for i, (actor, dist) in enumerate(actor_distance_list):
            id_dict_all[actor.id] = i
            transform = CarlaDataProvider.get_transform(actor)
            bbox = actor.bounding_box
            current_frame.append({
                'actor_id': actor.id,
                'type': actor.type_id,
                'location': transform.location,
                'yaw': transform.rotation.yaw,
                'length': bbox.extent.x * 2,
                'width': bbox.extent.y * 2,
                'velocity': CarlaDataProvider.get_velocity(actor)
            })
        frame_list = list(self.history_buffer)
        frame_list.append(current_frame)
        for t, frame in enumerate(frame_list):
            tt = 20-len(self.history_buffer)+t
            for dict in frame:
                if dict['actor_id'] not in id_dict_close:
                    continue
                rid = id_dict_close[actor.id]
                x_w = dict['location'].x
                y_w = -dict['location'].y
                p_w = np.array([x_w, y_w, 0.0, 1.0], dtype=np.float32)
                p_e = T @ p_w
                x_e, y_e = p_e[0], p_e[1]
                length = dict['length']
                width = dict['width']
                yaw_w = -math.radians(dict['yaw'])
                R = T[:3, :3]
                heading_w = np.array([math.cos(yaw_w), math.sin(yaw_w), 0.0], dtype=np.float32)
                heading_e = R @ heading_w
                yaw_e = math.atan2(heading_e[1], heading_e[0])
                h_cos = math.cos(yaw_e)
                h_sin = math.sin(yaw_e)
                vx_w = dict['velocity']
                vy_w = 0
                v_w = np.array([vx_w, vy_w, 0.0], dtype=np.float32)
                v_e = R @ v_w
                vx_e = v_e[0]
                vy_e = v_e[1]
                if self.is_two_wheeler(dict['type']):
                    features[rid, tt] = np.array([x_e, y_e, h_cos, h_sin,
                                         vx_e, vy_e, width, length,
                                         0.0, 0.0, 1.0], dtype=np.float32)
                elif 'vehicle' in dict['type']:
                    features[rid, tt] = np.array([x_e, y_e, h_cos, h_sin,
                                         vx_e, vy_e, width, length,
                                         1.0, 0.0, 0.0], dtype=np.float32)
                elif 'walker' in dict['type']:
                    features[rid, tt] = np.array([x_e, y_e, h_cos, h_sin,
                                         vx_e, vy_e, width, length,
                                         0.0, 1.0, 0.0], dtype=np.float32)
        self.history_buffer.append(current_frame)
        return features

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        
        # 获得world -> ego转换矩阵(右手系)
        x = tick_data['pos'].x
        y = -tick_data['pos'].y
        heading = tick_data['campass'] if not np.isnan(tick_data['campass']) else 0
        heading = -heading
        T = self.world_to_ego_matrix_rh(x, y, heading)

        data = {}
        R = T[:3, :3]
        speed = tick_data['speed']
        v_world = np.array([speed * math.cos(heading), speed * math.sin(heading), 0.0], dtype=np.float32)
        v_ego = R @ v_world
        a_world = np.array([tick_data['acceleration'][0], -tick_data['acceleration'][1], 0.0], dtype=np.float32)
        a_ego = R @ a_world
        yaw_rate = -tick_data['angular_velocity'][2]
        steer = yaw_rate*0.1
        data['ego_current_state'] = np.array([0.0, 0.0, 1.0, 0.0,
                                       v_ego[0], v_ego[1], a_ego[0], a_ego[1],
                                       steer, yaw_rate], dtype=np.float32)
        data['neighbor_agents_past'] = self.get_agent_features(32, T)
        data['static_objects'] = tool.get_static()
        near_node, future_traj, _, _, next_traffic_light, _, _= self._waypoint_planner.run_step(np.array([tick_data["pos"].x,tick_data["pos"].y]))
        q_xy = (x, y)
        lanes, lanes_speed_limit, lanes_has_speed_limit, route_lanes, route_lanes_speed_limit, route_lanes_has_speed_limit=tool.get_map_features(next_traffic_light, self.route_ids, self.lane_info, self.lane_polygons, self.lane_tree, T, q_xy)
        data['lanes'] = lanes
        data['lanes_speed_limit'] = lanes_speed_limit
        data['lanes_has_speed_limit'] = lanes_has_speed_limit
        data['route_lanes'] = route_lanes
        data['route_lanes_speed_limit'] = route_lanes_speed_limit
        data['route_lanes_has_speed_limit'] = route_lanes_has_speed_limit
        inputs = tool.build_model_inputs(data, self.arg_path, self.device)
        with torch.no_grad():
            _, outputs = self.model(inputs)
        out = outputs['predictions'][0, 0, :, :2].detach().cpu().numpy() # [80, 2]
        out = out.astype(np.float64) # 转成float32格式
        # 补充变量，然后直接复用pluto_agent的逻辑
        query_xy = q_xy
        ego_theta = heading
        raw_theta = -heading
        # 新增：保存前10步的未来预测轨迹
        if len(self.future_trajectories) < 10:
            self.future_trajectories.append((query_xy, out.copy(), ego_theta))
        out[:, 1] = -out[:, 1]
        print("out",out[:20])
        # command = tick_data['command_near']
        # if command < 0:
        #     command = 4
        # command -= 1
        # results['command'] = command
        theta_to_lidar = -raw_theta#这里取负只是为了逆时针旋转坐标系
        command_near_xy = np.array([near_node[0]-query_xy[0],near_node[1]+query_xy[1]])
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy
        future_np=[]
        for poi in future_traj:
            poi_np=np.array([poi[0]-query_xy[0],poi[1]+query_xy[1]])
            local_poi=rotation_matrix @ poi_np
            future_np.append(local_poi)
        print("local_command_xy",command_near_xy,local_command_xy)
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out, tick_data['speed'], local_command_xy)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0
        if tick_data['speed']>5:
            throttle_traj = 0
        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        control.steer = np.clip(float(steer_traj), -1, 1)
        # control.steer = 0
        # print('steer',control.steer)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)
        # print("加速",control.throttle)
        # print("刹车",control.brake)
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan'] = out.tolist()
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        #if SAVE_PATH is not None and self.step % 1 == 0:
           # self.save(tick_data,out.copy(),query_xy,raw_theta,control.steer,control.throttle,control.brake,polygon)
        self.prev_control = control
        return control

    # 直接从pluto_agent复制来可视化逻辑
    def trans2ego(self,query_xy,theta_to_lidar,traj):
        #query_xy肯theta_to_lidar都是run_sstep中的定义，这里保持不动
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        future_np=[]
        for poi in traj:
            poi_np=np.array([poi[0]-query_xy[0],-poi[1]+query_xy[1]])#poi[1]取反返回到carla坐标系
            local_poi=rotation_matrix @ poi_np
            future_np.append(local_poi)
        return future_np
    
    def draw_vehicle_status(self, img, steer, throttle, brake):
        # 定义起始位置和行间距
        start_x, start_y = 20, 30
        line_height = 25
        
        # 定义文字内容
        status_text = [
            f"Steer:    {steer:.2f}",
            f"Throttle: {throttle:.2f}",
            f"Brake:    {brake:.2f}"
        ]
        
        # 绘制一个半透明的黑色背景框，方便看清文字（可选）
        # cv2.rectangle(img, (10, 10), (200, 100), (0, 0, 0), -1) 
        
        for i, text in enumerate(status_text):
            y = start_y + i * line_height
            # 参数：图像，文字，坐标，字体，字号，颜色，粗细，抗锯齿
            cv2.putText(img, text, (start_x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return img

    def draw_traj_bev(self, traj, raw_img, canvas_size=(512,512), thickness=3, is_ego=False, hue_start=120, hue_end=80):
        if is_ego:
            line = np.concatenate([np.zeros((1,2)), traj], axis=0)
        else:
            line = traj
        
        img = raw_img.copy()
        pts_4d = np.stack([line[:,1], line[:,0], np.zeros((line.shape[0])), np.ones((line.shape[0]))])
        pts_2d = (self.coor2topdown @ pts_4d).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        
        # 1. 筛选画布内的点
        mask = (pts_2d[:, 0] > 0) & (pts_2d[:, 0] < canvas_size[1]) & \
               (pts_2d[:, 1] > 0) & (pts_2d[:, 1] < canvas_size[0])
        
        if not mask.any():
            return img
        
        pts_2d = pts_2d[mask, 0:2]

        # 2. 关键：数据清洗 (去重)
        # splprep 不允许重复点，我们通过计算点与点之间的距离来过滤
        if len(pts_2d) > 1:
            dist = np.linalg.norm(np.diff(pts_2d, axis=0), axis=1)
            keep = np.where(dist > 1e-5)[0] # 过滤掉距离过近的点
            pts_2d = np.concatenate([pts_2d[keep], pts_2d[-1:]], axis=0)

        # 3. 样条曲线拟合
        # splprep 至少需要 k+1 个点，k 默认为 3，所以至少要 4 个点
        if len(pts_2d) >= 4:
            try:
                # s=0 表示强制经过所有点，n_pts 是插值后的精细度
                tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
                unew = np.linspace(0, 1, 100)
                smoothed_pts = np.stack(splev(unew, tck)).astype(int).T
            except Exception as e:
                # 如果拟合还是失败，退化为原始点绘制
                print(f"Spline failed: {e}")
                smoothed_pts = pts_2d.astype(int)
        else:
            # 点数不够拟合，直接用原始点
            smoothed_pts = pts_2d.astype(int)

        # 4. 渐变色绘制
        num_points = len(smoothed_pts)
        for i in range(num_points - 1):
            hue = hue_start + (hue_end - hue_start) * (i / num_points)
            hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB).flatten()
            
            # 转换为 int tuple
            color = (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))
            
            p1 = tuple(smoothed_pts[i])
            p2 = tuple(smoothed_pts[i+1])
            cv2.line(img, p1, p2, color=color, thickness=thickness)
            
        return img
    
    def draw_polygons_bev(self, polygon_list, query_xy,theta_to_lidar,img, color=(0, 255, 0), thickness=2, fill=False):
        for poly in polygon_list:
            if poly is None or poly.is_empty:
                continue
            
            # 1. 提取顶点坐标 (N, 2)
            # poly.exterior.coords 返回的是 (x, y) 序列
            coords = np.array(poly.exterior.coords)
            coords=self.trans2ego(query_xy,theta_to_lidar,coords)
            coords=np.asarray(coords)
            print("形状",coords.shape)
            # 2. 构造 4D 齐次坐标 (与你之前的 line 处理逻辑保持一致)
            # 注意：你之前的逻辑是 [Y, X, Z, 1]
            pts_4d = np.ones((4, coords.shape[0]))
            pts_4d[0, :] = coords[:, 1]  # 对应你代码里的 line[:, 1] (通常是物理 Y)
            pts_4d[1, :] = coords[:, 0]  # 对应你代码里的 line[:, 0] (通常是物理 X)
            pts_4d[2, :] = 0             # 地面高度 Z = 0
            
            # 3. 矩阵投影运算
            # pts_2d 结果维度为 (N, 4)
            pts_2d = (self.coor2topdown @ pts_4d).T
            
            # 4. 透视除法 (归一化到像素平面)
            # 这里的 pts_2d[:, 2] 是深度信息 (W)
            u = pts_2d[:, 0] / pts_2d[:, 2]
            v = pts_2d[:, 1] / pts_2d[:, 2]
            
            # 5. 转换为 cv2 需要的整数像素格式
            pts_pixel = np.stack([u, v], axis=1).astype(np.int32)
            
            # 6. 执行绘制
            if fill:
                cv2.fillPoly(img, [pts_pixel], color=color)
            else:
                # isClosed=True 保证多边形首尾相连
                cv2.polylines(img, [pts_pixel], isClosed=True, color=color, thickness=thickness)
                
        return img

    def draw_lanes_bev(self, query_xy,theta_to_lidar,raw_img, canvas_size=(512,512), color_center=(255, 255, 255), color_side=(200, 200, 200), thickness=2):
        """
        lanes: List of tuples/dicts, 每个元素包含 (center_line, left_bound, right_bound)
            每个 bound 都是 shapely.geometry.LineString
        """
        img = raw_img.copy()
        radius=120.0
        x_min, x_max = query_xy[0] - radius, query_xy[0] + radius
        y_min, y_max = query_xy[1] - radius, query_xy[1] + radius
        patch = box(x_min, y_min, x_max, y_max)
        lane_keys=tool.fast_intersection_query_keys(self.lane_polygons,patch,self.lane_tree)
        for lane_key in lane_keys:
            # 假设 lane 是一个元组: (center, left, right)
            # 或者根据你的数据结构调整获取方式
            lane=self.lane_info[lane_key]
            for i, line_string in enumerate(lane):
                if line_string.is_empty:
                    continue
                    
                # 1. 从 shapely 提取坐标 (N, 2)
                coords = np.array(line_string.coords)
                coords=self.trans2ego(query_xy,theta_to_lidar,coords)
                coords=np.asarray(coords)
                # 2. 投影转换 (复用你调通的 [y, x] 逻辑)
                # 注意：如果你的 lane 坐标是绝对世界坐标，需要先转为自车相对坐标
                # 如果已经是相对坐标，直接按下面逻辑：
                pts_4d = np.stack([coords[:, 1], coords[:, 0], 
                                np.zeros(len(coords)), 
                                np.ones(len(coords))])
                
                pts_2d = (self.coor2topdown @ pts_4d).T
                pts_2d[:, 0] /= pts_2d[:, 2]
                pts_2d[:, 1] /= pts_2d[:, 2]
                
                # 3. 筛选画布内的点
                # 车道线通常较长，我们直接转换成整数像素点
                pts_pixel = pts_2d[:, :2].astype(np.int32)
                
                # 4. 绘制
                # i=0 是中心线，i=1,2 是左右边界
                current_color = color_center if i == 0 else color_side
                
                # 使用 polylines 一次性画出整条线
                # isClosed=False 表示不闭合
                cv2.polylines(img, [pts_pixel], isClosed=False, color=current_color, thickness=thickness)
                
        return img

    def save(self, tick_data,ego_traj,query_xy,raw_theta,steer, throttle, brake,polygons):
        frame = self.step // 10
        # Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))
        # 1. 获取底图（来自 CARLA 的俯视相机）
        bev_img = tick_data['bev'].copy()
        
        # bev_with_lanes = self.draw_lanes_bev(query_xy, -raw_theta,bev_img)
        
        # 2. 绘制自车预测轨迹 (在有车道的图上画)
        # bev_final = self.draw_traj_bev(ego_traj, bev_img, is_ego=True)
        # bev_final = self.draw_vehicle_status(bev_final, steer, throttle, brake)
        bev_final = self.draw_polygons_bev(polygons,query_xy, -raw_theta,bev_img)
        save_path = self.save_path / 'bev' / ('%04d.png' % frame)
        Image.fromarray(bev_final).save(save_path)
        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def destroy(self):
        if len(self.trajectory_history) > 0:
            try:
                import matplotlib.pyplot as plt
                x_coords = [p[0] for p in self.trajectory_history]
                y_coords = [p[1] for p in self.trajectory_history]
                
                plt.figure(figsize=(12, 12))
                # 1. 实际轨迹
                plt.plot(x_coords, y_coords, label='Actual Path', color='blue', linewidth=2, zorder=1)
                plt.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start', zorder=3)
                plt.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='End', zorder=3)
                
                # 2. 旋转并绘制前10步未来轨迹
                for i, (origin, traj, theta) in enumerate(self.future_trajectories):
                    short_traj = traj[:20]
                    # start_point = np.array([[0.0, 0.0]])
                    # short_traj = np.concatenate([start_point, short_traj], axis=0)
                    
                    print("short_traj",short_traj)
                    # 旋转矩阵变换: 
                    # 世界X = 局部X*cos(theta) - 局部Y*sin(theta) + 起点X
                    # 世界Y = 局部X*sin(theta) + 局部Y*cos(theta) + 起点Y
                    # cos_t = np.cos(theta)
                    # sin_t = np.sin(theta)
                    
                    # world_px = (short_traj[:, 0] * cos_t - short_traj[:, 1] * sin_t) + origin[0]
                    # world_py = (short_traj[:, 0] * sin_t + short_traj[:, 1] * cos_t) + origin[1]
                    
                    # label = "Predictions (Shortened)" if i == 0 else ""
                    # plt.plot(world_px, world_py, '-', color='green', alpha=0.5, linewidth=1, label=label, zorder=2)

                plt.title("Trajectory and Oriented Predictions")
                plt.legend()
                plt.grid(True)
                plt.axis('equal')
                plt.savefig("/data/swyl/pluto-eval/trajectory_plot9.png")
                plt.close()
            except Exception as e:
                print(f"Plotting failed: {e}")
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
