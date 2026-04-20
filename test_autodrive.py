import carla
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from nmpc_controller import NMPCController  # 引入我们刚刚用 Python 改写的 NMPC 核心类
from local_planner_manager import LocalPathPlannerWrapper # 新增：高度封装的触须接口

def main():
    def pick_small_obstacle_blueprints(bp_lib):
        preferred_ids = [
            'vehicle.audi.a2',
            'vehicle.mini.cooper_s',
            'vehicle.toyota.prius',
            'vehicle.tesla.model3',
            'vehicle.lincoln.mkz_2020',
            'vehicle.lincoln.mkz',
        ]
        out = []
        for bp_id in preferred_ids:
            try:
                out.append(bp_lib.find(bp_id))
            except Exception:
                continue

        if len(out) > 0:
            return out

        fallback = []
        for bp in bp_lib.filter('vehicle.*'):
            tid = bp.id.lower()
            if any(k in tid for k in ['bus', 'truck', 'firetruck', 'ambulance', 'carlacola']):
                continue
            fallback.append(bp)

        if len(fallback) > 0:
            return fallback

        return [bp_lib.find('vehicle.lincoln.mkz')]

    # ================== 1. 连接UE5 Carla服务端 ==================
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    # ================== 0. 启动前的大扫除 ==================
    print("🧹 正在执行启动前大扫除，清理遗留的赛博挂车垃圾...")
    for a in world.get_actors().filter('*'):
        # 宽泛匹配：哪怕是不规范的名字，只要带有 vehicle、trailer、或者你的自定义名字 airtor
        if 'vehicle' in a.type_id or 'trailer' in a.type_id or 'airtor' in a.type_id:
            if a.is_alive:
                a.destroy()
    print("✨ 清理完毕！环境已焕然一新。")
    
    # 🌟【关键修复】：开启 Carla 同步模式 (Synchronous Mode)
    # 因为 NMPC 每次循环计算可能耗时几十到上百毫秒，如果使用默认的异步模式，
    # 物理引擎会疯狂超前，导致下发控制时状态已过期，从而引发“前后跳动/抽搐”。
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 设置固定物理步长 (即 20 FPS 仿真刷新率)
    world.apply_settings(settings)
    
    # 获取仿真窗口的观察者（用于控制跟随视角）
    spectator = world.get_spectator()
    
    print("✅ 成功连接UE5 Carla仿真环境！")

    # ================== 2. 生成车辆 ==================
    try:
        vehicle_bp = blueprint_library.find('vehicle.airtor666.airtor666')
    except IndexError:
        print("⚠️ 未找到自定义车型 vehicle.airtor666.airtor666，将回退使用林肯 MKZ...")
        vehicle_bp = blueprint_library.find('vehicle.lincoln.mkz')

    spawn_points = world.get_map().get_spawn_points()
    # 固定起点：选取出生点列表中的第一个点（或者你可以改索引选别的固定点）
    spawn_point = spawn_points[0]
    
    # 尝试把出生点抬高一点，防止因为模型太大或者卡地盘导致生成失败
    spawn_point.location.z += 0.0 
    
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    # 如果第一个点因为碰撞无法生成，就遍历找一个能生成的点
    if vehicle is None:
        print("⚠️ 固定起点因为碰撞或者物理体积被阻挡，正在尝试别的出生点（并大幅抬高 Z 轴防穿模）...")
        for sp in spawn_points:
            sp.location.z += 5.0  # 直接抬高5米，对于一些原点在重心的大卡车非常有必要
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                spawn_point = sp
                break
                
    # 如果空气中也无法生成，说明完全是蓝图文件损坏，自动降级为林肯轿车防止代码中断
    if vehicle is None:
        print("❌ 自定义车辆全地图生成失败！这通常是因为模型的 Actor Description Class 在 UE5 中为空或失效。")
        print("🔧 自动启动安全回退机制：更换为内置的 Lincoln MKZ 以确保后续 NMPC 和 SLAM 算法测试可以继续进行...")
        vehicle_bp = blueprint_library.find('vehicle.lincoln.mkz')
        spawn_point = spawn_points[0]
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            raise RuntimeError("❌ 致命错误：连 Carla 默认内置的 Lincoln MKZ 都无法生成，请重启 Carla 服务端！")
            
    print(f"✅ 车辆 {vehicle_bp.id} 生成成功！")
    
    # 自动获取并打印生成的车辆的真实物理尺寸（Bounding Box）
    bbox = vehicle.bounding_box.extent
    print(f"📏 [车辆尺寸信息] 牵引车真实尺寸为: 长 {bbox.x * 2:.3f} 米, 宽 {bbox.y * 2:.3f} 米, 高 {bbox.z * 2:.3f} 米")

    # ================== 2.2 生成一些随机障碍物（展示上帝视角感知） ==================
    print("🚗 正在前方道路上专门生成静态障碍物用于测试局部路径规划（避障）...")
    obstacle_actors = []
    obstacle_actor_ids = set()
    collided_test_obstacle_ids = set()
    
    # 获取当前道路，并在正前方距离 20 米、45 米处分别生成停放车辆或路障
    base_wp = world.get_map().get_waypoint(spawn_point.location)
    
    # 定义我们在前方多少米处放置路障
    obstacle_distances = [25.0, 50.0]
    # 定义横向偏移偏移量（模拟车辆占用了半个车道或整个车道）
    obstacle_offsets = [0.5, -0.8] 

    small_obstacle_bps = pick_small_obstacle_blueprints(blueprint_library)
    print(f"🚧 固定小车障碍蓝图库: {[bp.id for bp in small_obstacle_bps[:4]]}")

    for i, dist in enumerate(obstacle_distances):
        try:
            # 往前推进找到目标路点
            obs_wp = base_wp.next(dist)[0]
            
            # 计算路点的朝向
            forward_vec = obs_wp.transform.get_forward_vector()
            right_vec = obs_wp.transform.get_right_vector()
            
            # 在路点上加入横向偏移产生最终位置
            obs_loc = obs_wp.transform.location + right_vec * obstacle_offsets[i]
            obs_loc.z += 0.5 # 适度抬高防卡在地下
            
            # 固定用小车作为障碍物，避免随机生成大巴/重卡导致可通行性失真
            obs_bp = small_obstacle_bps[i % len(small_obstacle_bps)]
            obs_transform = carla.Transform(obs_loc, obs_wp.transform.rotation)
            
            obs_vehicle = world.try_spawn_actor(obs_bp, obs_transform)
            if obs_vehicle:
                # 将生成的车辆的手刹和物理静态化
                obs_vehicle.set_autopilot(False)
                obs_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                obstacle_actors.append(obs_vehicle)
                obstacle_actor_ids.add(obs_vehicle.id)
        except Exception as e:
            pass
    print(f"✅ 成功生成了 {len(obstacle_actors)} 个精确放置的前方静态路障！测试触须法避障启动。")

    # ================== 2.3 生成动态障碍物（用于速度规划能力测试） ==================
    print("🚙 正在生成动态障碍物，用于测试速度规划（跟驰/减速/再加速）...")
    dynamic_obstacle_actors = []

    # 说明：
    # - 距离放在静态障碍后方更远处，避免一开始就混在一起导致难以分辨问题来源。
    # - 速度单位 m/s（3.0m/s≈10.8km/h）。
    dynamic_obstacle_specs = [
        {'distance': 58.0, 'offset': 0.10, 'speed': 3.0},
        {'distance': 76.0, 'offset': -0.20, 'speed': 4.2},
        {'distance': 94.0, 'offset': 0.25, 'speed': 2.4},
    ]

    for i, spec in enumerate(dynamic_obstacle_specs):
        try:
            dyn_wp = base_wp.next(float(spec['distance']))[0]
            right_vec = dyn_wp.transform.get_right_vector()
            fwd_vec = dyn_wp.transform.get_forward_vector()

            dyn_loc = dyn_wp.transform.location + right_vec * float(spec['offset'])
            dyn_loc.z += 0.5

            dyn_bp = small_obstacle_bps[(i + len(obstacle_distances)) % len(small_obstacle_bps)]
            dyn_tf = carla.Transform(dyn_loc, dyn_wp.transform.rotation)
            dyn_vehicle = world.try_spawn_actor(dyn_bp, dyn_tf)
            if dyn_vehicle is None:
                continue

            dyn_vehicle.set_autopilot(False)
            dynamic_obstacle_actors.append(dyn_vehicle)
            obstacle_actors.append(dyn_vehicle)
            obstacle_actor_ids.add(dyn_vehicle.id)

            target_speed = float(spec['speed'])
            vel_vec = carla.Vector3D(
                x=fwd_vec.x * target_speed,
                y=fwd_vec.y * target_speed,
                z=0.0,
            )

            # 优先使用恒速模式，确保动态障碍可持续移动且速度稳定。
            try:
                dyn_vehicle.enable_constant_velocity(vel_vec)
            except Exception:
                dyn_vehicle.set_target_velocity(vel_vec)

        except Exception:
            continue

    print(
        f"✅ 动态障碍物生成完成：{len(dynamic_obstacle_actors)} 辆 "
        f"(目标速度: {[round(s['speed']*3.6, 1) for s in dynamic_obstacle_specs]} km/h)"
    )
    # ================== 2.5 生成并可视化二维空间的“参考路径” ==================
    carla_map = world.get_map()
    # 1. 找到车辆出生点对应的地图路点（归到车道中心线上）
    current_waypoint = carla_map.get_waypoint(spawn_point.location)
    reference_path = []
    
    # 2. 沿着车道生成“全图级长路径”（不再固定 200m）
    # 策略：每 2m 取点，路口优先同车道，若形成闭环则在完成一圈后停止。
    def _pick_next_waypoint(curr_wp, next_wps):
        if not next_wps:
            return None

        same_lane = [
            cand for cand in next_wps
            if cand.road_id == curr_wp.road_id and cand.lane_id == curr_wp.lane_id
        ]
        candidates = same_lane if len(same_lane) > 0 else list(next_wps)

        curr_yaw = math.radians(curr_wp.transform.rotation.yaw)
        best_wp = None
        best_cost = float('inf')
        for cand in candidates:
            cand_yaw = math.radians(cand.transform.rotation.yaw)
            dyaw = abs((cand_yaw - curr_yaw + math.pi) % (2 * math.pi) - math.pi)
            cost = dyaw
            if best_wp is None or cost < best_cost:
                best_wp = cand
                best_cost = cost
        return best_wp

    wp = current_waypoint
    visited = {}
    max_ref_points = 8000
    for _ in range(max_ref_points):
        reference_path.append(wp)

        # 通过 (road,section,lane,s) 检测回环，避免无限循环
        key = (int(wp.road_id), int(wp.section_id), int(wp.lane_id), int(round(float(wp.s) * 2.0)))
        visited[key] = visited.get(key, 0) + 1
        if visited[key] >= 2 and len(reference_path) > 800:
            break

        next_wps = wp.next(2.0)
        if not next_wps:
            break
        nxt = _pick_next_waypoint(wp, next_wps)
        if nxt is None:
            break
        wp = nxt
            
    # 3. 将这些参考点在仿真界面的空中画出来（绿色小点，z轴抬高一点防遮挡）
    for w in reference_path:
        loc = w.transform.location
        draw_loc = loc + carla.Location(z=0.5)
        # 用 Debug 画笔一直留存在画面中 (life_time=0 永不消失)
        world.debug.draw_point(draw_loc, size=0.1, color=carla.Color(0,255,0), life_time=0.0)

    route_len = 0.0
    if len(reference_path) > 1:
        for i in range(1, len(reference_path)):
            p0 = reference_path[i - 1].transform.location
            p1 = reference_path[i].transform.location
            route_len += math.hypot(p1.x - p0.x, p1.y - p0.y)

    print(f"🛣️ 成功生成全图级参考路径: {len(reference_path)} 个 Waypoint, 约 {route_len:.1f} m")

    # ================== 3. 加装激光雷达 ==================
    # 3.1 找到激光雷达蓝图
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    
    # 3.2 配置激光雷达参数（可根据需要调整）
    lidar_bp.set_attribute('channels', '64')          # 32线
    lidar_bp.set_attribute('points_per_second', '56000')  # 每秒点数
    lidar_bp.set_attribute('rotation_frequency', '10') # 旋转频率（Hz）
    lidar_bp.set_attribute('range', '50')              # 探测距离（米）
    
    # 3.3 安装位置：车顶正上方
    lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    # 3.3.1 碰撞传感器：用于诊断“撞后速度趋近0”的原因
    collision_state = {
        'count': 0,
        'last_actor_id': -1,
        'last_actor_type': 'none',
        'last_impulse': 0.0,
        'last_time': -1.0,
    }
    collision_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    
    # 3.4 定义全局地图存储（不再限制帧数，保存整个走过的区域建图）
    global_map_list = []

    # 3.5 定义激光雷达数据回调函数
    def lidar_callback(data):
        # 将Carla的激光雷达对象转换为numpy数组格式
        # 注意: np.frombuffer返回的是只读视图，必须加上 .copy() 才能修改
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3].copy()
        
        # 简单降采样（取五分之一），避免200米的全局点云太大导致系统卡死
        points = points[::5]

        # 翻转 y 轴以将左手系(Carla)转为正常的右手观察系(Open3D)
        points[:, 1] = -points[:, 1]
        
        # 获取当前传感器位姿矩阵，将局部点云转换到全局坐标系以防止移动模糊
        sensor_matrix = lidar.get_transform().get_matrix()
        # 补充齐次坐标列
        points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
        points_world = (np.array(sensor_matrix) @ points_homo.T).T[:, :3]
        
        global_map_list.append(points_world)

    def collision_callback(event):
        other = event.other_actor
        oid = other.id if other is not None else -1
        otype = other.type_id if other is not None else 'none'
        imp = event.normal_impulse
        imp_mag = float(math.sqrt(imp.x * imp.x + imp.y * imp.y + imp.z * imp.z))

        collision_state['count'] += 1
        collision_state['last_actor_id'] = oid
        collision_state['last_actor_type'] = otype
        collision_state['last_impulse'] = imp_mag
        collision_state['last_time'] = time.time()

        if oid in obstacle_actor_ids:
            collided_test_obstacle_ids.add(oid)

        print(
            f"\n[Collision] count={collision_state['count']} actor={otype}#{oid} "
            f"impulse={imp_mag:.2f} collided_test={oid in obstacle_actor_ids}"
        )
    
    # 3.6 开始监听激光雷达数据
    lidar.listen(lidar_callback)
    collision_sensor.listen(collision_callback)
    print("✅ 激光雷达安装成功，开始进行全局路线建图！")

    # ================== 4. 关闭自带自动驾驶，准备使用自定义循迹 ==================
    vehicle.set_autopilot(False)
    
    # 实例化 Python 版本的 NMPC
    nmpc_horizon = 15  # 默认预测步数 15，控制器内部会按曲率自适应
    planner = NMPCController(N=nmpc_horizon, dt=0.1)

    target_wp_index = 0
    print("🚗 车辆已切换为【Python NMPC 控制】，无视红绿灯、强行沿轨迹行驶！")
    print("按 Ctrl+C 停止程序")

    # ================== 5. Python Matplotlib 实时 2D 可视化 ==================
    plt.ion() # 开启交互绘图模式
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Carla 2D Top-Down View")
    
    # 提取所有参考路线坐标作为全局绘制
    ref_x = [wp.transform.location.x for wp in reference_path]
    ref_y = [wp.transform.location.y for wp in reference_path]
    
    print("⏳ 正在预热物理引擎，让车辆平稳落地并同步初始坐标...")
    # 物理引擎预热：先空跑 10 帧，让车辆完全掉到地上且坐标刷新正常
    # 避免一上来没有 tick 导致读取到 (0,0) 的幻影坐标画出一条横跨地图的红线
    for _ in range(10):
        world.tick()

    # ================== 自动侦测挂车尺寸与挂点间距 ==================
    print("🔍 正在扫描主车后方的挂车信息...")
    ego_loc = vehicle.get_transform().location
    potential_trailers = []
    # 检索附近 60 米内的所有疑似车或者挂车的 Actor
    for a in world.get_actors().filter('*'):
        if a.id != vehicle.id and ('vehicle' in a.type_id.lower() or 'trailer' in a.type_id.lower() or 'airtor' in a.type_id.lower()):
            # 排除掉我们为了测试感知识别专门生成在远处的随机 NPC 车辆 (必须用 id 比较，不同引用的地址不同)
            if a.id not in obstacle_actor_ids and a.get_transform().location.distance(ego_loc) < 60.0:
                potential_trailers.append(a)

    # 按照距离车头由近到远排序，从而确定 挂车1, 挂车2... 的顺序
    potential_trailers.sort(key=lambda a: a.get_transform().location.distance(ego_loc))
    trailer_hitch_gaps = []
    prev_actor = vehicle
    for i, tr in enumerate(potential_trailers):
        tr_bbox = tr.bounding_box.extent
        # 真实空间中两车中心点(质心)坐标的三维直线距离
        dist = tr.get_transform().location.distance(prev_actor.get_transform().location)
        # 用中心距减去 前车的一半长度 和 后车的一半长度，算出来的就是挂钩间隙（大概物理空隙）
        hitch_gap = dist - (prev_actor.bounding_box.extent.x + tr_bbox.x)
        trailer_hitch_gaps.append(max(0.5, hitch_gap))
        print(f"🔗 [挂车 {i+1} 真实信息] 长 {tr_bbox.x * 2:.3f} 米, 宽 {tr_bbox.y * 2:.3f} 米 | 中心点距前车 {dist:.3f} 米 (预估铰接空隙: {max(0, hitch_gap):.3f} 米)")
        prev_actor = tr
    if not potential_trailers:
        print("ℹ️ 未扫描到任何自动伴生生成的挂车！")
    # ================================================================

    # ================== 初始化局部避障规划器 (Tentacle 接口封装) ==================
    num_trailers = len(potential_trailers)
    tractor_length = max(2.0, vehicle.bounding_box.extent.x * 2.0)
    tractor_width = max(1.5, vehicle.bounding_box.extent.y * 2.0)
    trailer_lengths = [max(2.0, tr.bounding_box.extent.x * 2.0) for tr in potential_trailers]
    trailer_widths = [max(1.5, tr.bounding_box.extent.y * 2.0) for tr in potential_trailers]
    trailer_hitch = trailer_hitch_gaps if len(trailer_hitch_gaps) > 0 else 2.1

    print(f"📐 规划器尺寸参数: tractor=({tractor_length:.2f}m x {tractor_width:.2f}m), trailers={len(trailer_lengths)}")
    local_planner_wrapper = LocalPathPlannerWrapper(dt=0.1, planning_horizon=4.5)
    local_planner_wrapper.initialize_planner(
        num_trailers=num_trailers,
        tractor_length=tractor_length,
        tractor_width=tractor_width,
        trailer_lengths=trailer_lengths,
        trailer_widths=trailer_widths,
        trailer_hitch_gap=trailer_hitch,
        reference_path=reference_path
    )

    # 建立历史轨迹缓存
    actual_x = []
    actual_y = []
    
    frame_count = 0

    # ================== 5.5 【上帝视角】获取环境中所有的动态与静态障碍物 ==================
    print("🌍 正在提取地图静态物体(建筑、树木、围墙)...")
    # 提前获取静态环境物体，避免每帧重复请求导致卡顿
    static_env_objects = []
    # 遍历我们关心的静态环境类型 (Carla语义标签里分别命名了不同车型)
    labels_to_fetch = [
        carla.CityObjectLabel.Buildings, 
        carla.CityObjectLabel.Vegetation, 
        carla.CityObjectLabel.Fences, 
        carla.CityObjectLabel.Walls, 
        carla.CityObjectLabel.Car,         # 停放的小汽车
        carla.CityObjectLabel.Truck,       # 停放的卡车
        carla.CityObjectLabel.Bus,         # 停放的公交车
        carla.CityObjectLabel.Poles,       # 电线杆/路灯杆
        carla.CityObjectLabel.TrafficSigns # 交通标志牌
    ]
    for label in labels_to_fetch:
        static_env_objects.extend(world.get_environment_objects(label))
    
    # 提前把所有静态物体的 XY 坐标缓存下来
    static_obs_coords = [(obj.transform.location.x, obj.transform.location.y) for obj in static_env_objects]
    print(f"🌲 静态环境提取完毕，共捕获 {len(static_obs_coords)} 个地图物体。")

    def get_obstacles(world, ego_vehicle, ignore_actors=None):
        if ignore_actors is None:
            ignore_actors = []
        ignore_ids = [ego_vehicle.id] + [a.id for a in ignore_actors] + list(collided_test_obstacle_ids)
        
        dyn_obstacles = []
        # 获取所有的动态车辆和行人
        actor_list = world.get_actors()
        vehicles = actor_list.filter('vehicle.*')
        walkers = actor_list.filter('walker.*')

        for actor in list(vehicles) + list(walkers):
            # 排除自己本身以及指定的挂车等关联车辆
            if actor.id not in ignore_ids:
                tf = actor.get_transform()
                loc = tf.location
                vel = actor.get_velocity()
                yaw = math.radians(tf.rotation.yaw)
                bbox = actor.bounding_box.extent

                dyn_obstacles.append({
                    'x': float(loc.x),
                    'y': float(loc.y),
                    'length': float(max(0.8, bbox.x * 2.0)),
                    'width': float(max(0.5, bbox.y * 2.0)),
                    'yaw': float(yaw),
                    'vx': float(vel.x),
                    'vy': float(vel.y),
                    'speed': float(math.hypot(vel.x, vel.y)),
                })
                
        # 静态环境分两套：
        # 1) 可视化集：周围 50m 全部展示；
        # 2) 规划集：仅取车前走廊区域，避免建筑/树木中心点误伤触须评估。
        stat_obstacles_viz = []
        stat_obstacles_plan = []
        ego_tf = ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        static_plan_range = 40.0
        static_plan_forward_min = -8.0
        static_plan_forward_max = 35.0
        static_plan_lateral_max = 7.0

        for sx, sy in static_obs_coords:
            dx = sx - ego_loc.x
            dy = sy - ego_loc.y
            dist = math.hypot(dx, dy)

            if dist < 50.0:
                stat_obstacles_viz.append((sx, sy))

            if dist > static_plan_range:
                continue

            along = dx * fwd.x + dy * fwd.y
            lateral = abs(dx * right.x + dy * right.y)

            if along < static_plan_forward_min or along > static_plan_forward_max:
                continue
            if lateral > static_plan_lateral_max:
                continue

            stat_obstacles_plan.append((sx, sy))

        if len(stat_obstacles_plan) > 80:
            stat_obstacles_plan.sort(key=lambda p: math.hypot(p[0] - ego_loc.x, p[1] - ego_loc.y))
            stat_obstacles_plan = stat_obstacles_plan[:80]

        return dyn_obstacles, stat_obstacles_plan, stat_obstacles_viz

    def estimate_path_curvature(path_xyz):
        """
        估计局部参考轨迹曲率（取前半段中位值，强调即将到来的弯道）。
        path_xyz: np.ndarray shape (N, 3) -> [x,y,yaw]
        """
        if path_xyz is None or len(path_xyz) < 3:
            return 0.0

        arr = np.asarray(path_xyz, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 3:
            return 0.0

        n = arr.shape[0]
        seg_n = max(3, n // 2)
        dx = np.diff(arr[:seg_n, 0])
        dy = np.diff(arr[:seg_n, 1])
        ds = np.hypot(dx, dy)
        ds = np.maximum(ds, 1e-4)
        dyaw = np.diff(arr[:seg_n, 2])
        dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
        kappa = np.abs(dyaw / ds)
        if kappa.size == 0:
            return 0.0
        return float(np.median(kappa))

    # ================== 6. 主循环：更新跟随视角 + 打印车速 ==================
    try:
        while True:
            # --- 6.1 更新跟随视角（核心代码） ---
            # 获取车辆当前的位置和旋转
            vehicle_transform = vehicle.get_transform()
            current_loc = vehicle_transform.location
            
            # 计算观察者的位置：在车后方 8 米，上方 3 米
            # 先获取车辆的前进方向向量
            forward_vec = vehicle_transform.get_forward_vector()
            # 观察者位置 = 车辆位置 - 前进方向*8 + 向上*3
            spectator_location = vehicle_transform.location - forward_vec * 50.0 + carla.Location(z=50.0)
            
            # 计算观察者的旋转：让它看着车辆的位置
            spectator_rotation = carla.Rotation(
                pitch=-50.0,  # 稍微向下看
                yaw=vehicle_transform.rotation.yaw,  # 和车辆朝向一致
                roll=0.0
            )
            
            # 应用观察者的位置和旋转
            spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
            
            # --- 6.2 实时通过 Matplotlib 更新 2D 散点俯视图 ---
            actual_x.append(current_loc.x)
            actual_y.append(current_loc.y)
            frame_count += 1
            
            # 【上帝视角】感知周围环境中的动态与静态障碍物 (排除自身和被拖拽的挂车)
            dyn_obs, stat_obs_plan, stat_obs_viz = get_obstacles(world, vehicle, ignore_actors=potential_trailers)
            
        # 2. 从 Carla 获取目前精准的车辆信息
            current_yaw_rad = math.radians(vehicle_transform.rotation.yaw)
            velocity = vehicle.get_velocity()
            current_v = math.hypot(velocity.x, velocity.y)
            current_steer = vehicle.get_control().steer
            max_steer_rad = 30.0 * math.pi / 180.0
            actual_steer_rad = current_steer * max_steer_rad

            # =============== 【新增】使用封装后的本地规划器接口 ===============
            replan_time = frame_count * 0.05

            # 将 Carla 中每节挂车的实时位姿回传给重规划，避免使用车头姿态重置挂车状态。
            current_trailer_states = None
            if len(potential_trailers) > 0:
                trailer_states_buf = []
                for tr in potential_trailers:
                    if not tr.is_alive:
                        continue
                    ttf = tr.get_transform()
                    trailer_states_buf.append({
                        'x': float(ttf.location.x),
                        'y': float(ttf.location.y),
                        'yaw': float(math.radians(ttf.rotation.yaw)),
                    })
                if len(trailer_states_buf) > 0:
                    current_trailer_states = trailer_states_buf

            best_tentacle_traj = local_planner_wrapper.run_step(
                current_loc=current_loc,
                current_yaw_rad=current_yaw_rad,
                current_v=current_v,
                dyn_obs=dyn_obs,
                stat_obs=stat_obs_plan,
                replan_time=replan_time,
                current_trailer_states=current_trailer_states,
            )
            # ================================================================

            # 每隔 3 帧更新一次画图，避免把 matplotlib 画死卡顿
            if frame_count % 3 == 0:
                ax.cla()
                
                # 绘制1：将激光雷达 3D 点云拍平成 2D (去掉Z轴)，每 50 个点抽稀一下画灰色背景墙
                if len(global_map_list) > 0:
                    xyz = np.vstack(global_map_list)
                    pts = xyz[::50] # 强力抽稀
                    ax.plot(pts[:, 0], pts[:, 1], '.', color='lightgray', markersize=1, label='LiDAR Map (2D)')
                
                # 绘制2：用绿色画出长路参考线
                ax.plot(ref_x, ref_y, 'g--', linewidth=2, label='Reference Path')
                
                # 绘制3：用红色画出车辆过去实走的轨迹
                ax.plot(actual_x, actual_y, 'r-', linewidth=2, label='Actual Follow Path')
                
                # 绘制4：标出当前车头小蓝点
                ax.plot(current_loc.x, current_loc.y, 'bo', markersize=6, label='Ego Vehicle')
                
                # 绘制5：使用棕色半透明方块绘制出周围 50 米内的静态树木/建筑物/墙壁
                if len(stat_obs_viz) > 0:
                    sx = [obs[0] for obs in stat_obs_viz]
                    sy = [obs[1] for obs in stat_obs_viz]
                    ax.plot(sx, sy, 's', color='saddlebrown', markersize=4, alpha=0.3, label='Static Env (Trees/Bldgs)')
                
                # 绘制6：使用黑色三角形绘制出动态 NPC 车辆和行人障碍
                if len(dyn_obs) > 0:
                    dx = [obs['x'] if isinstance(obs, dict) else obs[0] for obs in dyn_obs]
                    dy = [obs['y'] if isinstance(obs, dict) else obs[1] for obs in dyn_obs]
                    ax.plot(dx, dy, 'k^', markersize=6, label='Dynamic Obstacles')
                
                # 绘制7：画出局部避障的最优触须路径（紫色实线）
                if best_tentacle_traj is not None:
                    tentacle_x = [pt['x'] for pt in best_tentacle_traj]
                    tentacle_y = [pt['y'] for pt in best_tentacle_traj]
                    ax.plot(tentacle_x, tentacle_y, 'm-', linewidth=3, label='Tentacle Local Path')

                ax.axis('equal') # 锁定比例尺，使长宽相同不畸形
                ax.set_title("NMPC Path Tracking Real-time 2D Overlay")
                ax.legend(loc='upper right')
                
                # 仅仅更新画布内部缓存
                try:
                    plt.pause(0.001)
                except Exception:
                    pass

            # --- 6.3 【核心】自定义位置寻迹控制 ---
            final_target_v = 0.0
            throttle = 0.0
            brake = 0.0

            # 判断是否还在参考路径以内
            if target_wp_index < len(reference_path):
                # 获取当前车辆与目标的坐标
                target_wp = reference_path[target_wp_index]
                target_loc = target_wp.transform.location
                
                # 计算二维距离
                dist = math.hypot(target_loc.x - current_loc.x, target_loc.y - current_loc.y)
                
                # 减小切点半径，避免在弯道提前切到后续点导致“提早转弯”
                if dist < 1.8:
                    target_wp_index += 1
                
                if target_wp_index < len(reference_path):
                    # 【核心核心】：调用刚刚封装好的接口拿去 NMPC 的追踪轨迹和速度
                    state = [current_loc.x, current_loc.y, current_yaw_rad, current_v]
                    
                    target_trajectory, target_v = local_planner_wrapper.get_tracked_trajectory(
                        nmpc_horizon=nmpc_horizon,
                        current_v=current_v, 
                        fallback_wps=reference_path, 
                        target_wp_index=target_wp_index
                    )

                    # 调用我们自己写的 Python 版本 NMPC
                    # 远离动态障碍时，提高速度参考下限（仅直道生效），避免重载低速爬行。
                    local_kappa = estimate_path_curvature(target_trajectory)
                    target_v_ref = float(target_v)
                    nearest_dyn = float(local_planner_wrapper.nearest_dynamic_obs_dist)
                    is_straight = local_kappa < 0.035
                    if is_straight:
                        if nearest_dyn > 22.0:
                            target_v_ref = max(target_v_ref, 3.0)
                        elif nearest_dyn > 14.0:
                            target_v_ref = max(target_v_ref, 2.2)
                        elif nearest_dyn > 10.0:
                            target_v_ref = max(target_v_ref, 1.6)

                    # 弯道限速：按横向加速度上限计算速度帽，抑制“全油门过弯”
                    a_lat_limit = 1.6
                    if local_kappa > 1e-4:
                        v_curve_cap = math.sqrt(a_lat_limit / local_kappa)
                        # 给一个下限，避免极端曲率下直接锁死
                        v_curve_cap = max(0.8, min(v_curve_cap, 6.0))
                        target_v_ref = min(target_v_ref, v_curve_cap)

                    optimized_v, target_steer_rad = planner.solve(
                        state,
                        target_trajectory,
                        target_speed=target_v_ref,
                    )

                    # 由 NMPC 同时控制速度与转向；target_v 作为速度参考而非直接执行值
                    final_target_v = max(0.0, float(optimized_v))

                    # 结合实际转角再做一次横向加速度限速，避免求解噪声导致的弯中速度偏高
                    cmd_kappa = abs(math.tan(target_steer_rad) / max(1e-6, 3.0))
                    if cmd_kappa > 1e-4:
                        v_cmd_curve_cap = math.sqrt(a_lat_limit / cmd_kappa)
                        v_cmd_curve_cap = max(0.8, min(v_cmd_curve_cap, 6.0))
                        final_target_v = min(final_target_v, v_cmd_curve_cap)
                    
                    # 将 NMPC 优化出来的物理命令转化为 Carla 阿克曼控制量
                    out_steer = max(-1.0, min(1.0, target_steer_rad / max_steer_rad))
                    speed_error = final_target_v - current_v
                    
                    # 取消过大的0.5m/s (即1.8km/h)静区死区，改为更灵敏连续的比例控制，并且起步给大点油门
                    if final_target_v < 0.1:
                        throttle = 0.0
                        brake = 1.0 # 要求刹停时给满刹车
                    else:
                        if speed_error > 0.0:
                            # 牵引增强：重载起步给更高基础油门，防止长期<1km/h的爬行。
                            throttle = min(1.0, 0.40 + speed_error * 1.6)

                            # 弯道油门上限，进一步抑制“全油门过弯”
                            if local_kappa > 0.08:
                                throttle = min(throttle, 0.38)
                            elif local_kappa > 0.05:
                                throttle = min(throttle, 0.50)
                            elif local_kappa > 0.03:
                                throttle = min(throttle, 0.62)

                            if current_v < 1.5 and final_target_v > 1.5:
                                throttle = max(throttle, 0.75)
                            if current_v < 0.8 and final_target_v > 2.5:
                                throttle = max(throttle, 0.90)

                            # 若处于明显弯道，禁止起步增强把油门重新抬高到过激值
                            if local_kappa > 0.03:
                                throttle = min(throttle, 0.62)
                            brake = 0.0
                        else:
                            throttle = 0.0
                            brake = min(1.0, -speed_error * 0.5)
                    
                    # 下发控制指令给 Carla 的汽车
                    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=out_steer, brake=brake))
            else:
                # 若走到参考路径尽头，一脚踩死刹车停止
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                print("\n🎉 已顺利抵达参考路径终点，自动跳出并开始保存！")
                break  # 停止死循环，主动进入 finally 去保存文件！

            # --- 6.4 打印车速与建图信息 ---
            velocity = vehicle.get_velocity()
            speed = 3.6 * math.hypot(velocity.x, velocity.y)
            pts_cnt = len(xyz) if len(global_map_list) > 0 and 'xyz' in locals() else 0
            
            print(
                f"\r进度: {target_wp_index}/{len(reference_path)} | 速度: {speed:>4.1f}km/h "
                f"| 指令: {final_target_v*3.6:>4.1f}km/h | 油门: {throttle:>4.2f} "
                f"| 最近动态障碍: {local_planner_wrapper.nearest_dynamic_obs_dist:>5.2f}m "
                f"| 有效触须: {local_planner_wrapper.last_valid_count}/{local_planner_wrapper.last_total_count} "
                f"| 碰撞: {collision_state['count']} | 点云: {pts_cnt} 点",
                end="",
                flush=True,
            )
            
            # --- 6.5 步进仿真引擎 ---
            # 只有在此刻调用 tick() 物理引擎才会往前真的走一帧 (0.05秒)
            world.tick()

    except KeyboardInterrupt:
        print("\n🛑 程序被手动停止")
    
    # ================== 7. 必须执行：清理所有生成的Actor ==================
    finally:
        print("\n🧹 正在清理仿真环境...")
        
        # ⚠️ 【极其重要】：退出程序时必须要把同步模式关掉改回 False!
        # 否则 Carla 服务器会一直挂在后台等待客户端发 world.tick()，导致你别的脚本跑不动！
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            
        if 'lidar' in locals() and lidar.is_alive:
            lidar.destroy()  # 先销毁传感器
        if 'collision_sensor' in locals() and collision_sensor.is_alive:
            collision_sensor.destroy()
            
        if 'world' in locals():
            # 斩草除根：不限于 'vehicle.*'，只要沾边的 Actor 一律强行抹除
            actors = world.get_actors().filter('*')
            for a in actors:
                type_id = a.type_id.lower()
                if 'vehicle' in type_id or 'trailer' in type_id or 'airtor' in type_id:
                    if a.is_alive:
                        a.destroy()
                        
        plt.close('all') # 关闭最终卡住的2D画布
            
        print("✅ 环境清理完成，同步模式已关闭，程序正常结束")

if __name__ == '__main__':
    main()