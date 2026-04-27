import math
import numpy as np
from single_replan_interface import create_planner, run_single_replan

class LocalPathPlannerWrapper:
    """
    我们将原本散落在 test_autodrive.py 中的局部路径规划(Tentacle触须法)的初始化、更新与提取逻辑，
    统一封装在这个接口类中，以保持主脚本逻辑清晰。
    """
    def __init__(self, dt=0.1, planning_horizon=3.0):
        # 内部持有真正的 planner 对象
        self.local_planner = None
        self.dt = dt
        self.planning_horizon = planning_horizon
        
        self.best_tentacle_traj = None
        self.display_tentacle_traj = None
        self.geom_traj = None
        self.last_target_v = 3.0
        self.nearest_dynamic_obs_dist = float('inf')
        self.last_valid_count = 0
        self.last_total_count = 0
        self.last_phase = "none"

    def initialize_planner(self, num_trailers, tractor_length=4.155, tractor_width=2.5, 
                           trailer_lengths=None, trailer_widths=None, trailer_hitch_gap=2.1,
                           planner_config=None,
                           reference_path=None):
        """
        初始化触须法规划器并录入主车与挂车的物理硬件参数以及全局参考线
        """
        if trailer_lengths is None:
            trailer_lengths = [6.0] * num_trailers
        if trailer_widths is None:
            trailer_widths = [2.5] * num_trailers

        # 实例化 planner
        base_config = {
            'dt': self.dt,
            'planning_horizon': self.planning_horizon,
            'v_max': 6.94,  # 25 km/h
            'a_max': 1.8,
            'a_min': -2.5,
            'a_lat_max': 5.5,
            # 增强横向绕行能力，避免双障碍前“有速度但绕不出”而蠕动。
            'num_tentacles': 19,
            'primary_tentacles': 9,
            'fallback_tentacles': 31,
            'lateral_deviation': 6.5,
            'lateral_deviation_fallback_wide': 8.0,
            'centerline_cost_weight': 0.25,
            'lateral_cost_weight': 0.8,
            'hysteresis_switch_margin': 8.0,
            # 提高几何碰撞保守度，避免“碰上后继续顶着走”
            'static_collision_num_circles': 3,
            'dynamic_collision_num_circles': 3,
            'collision_hard_overlap_tolerance': 0.0,
            'collision_margin': 0.6,
            'obstacle_safety_buffer': 0.35,
            'speed_shape_alpha_up': 0.45,
            'speed_shape_alpha_down': 0.70,
            'speed_shape_jerk_limit': 8.0,
        }
        if planner_config is not None:
            base_config.update(planner_config)
        self.local_planner = create_planner(num_trailers=num_trailers, config=base_config)

        # 覆盖真实尺寸
        self.local_planner.tractor_length = tractor_length
        self.local_planner.tractor_width = tractor_width
        self.local_planner.trailer_length = trailer_lengths
        self.local_planner.trailer_width = trailer_widths

        # 挂接距
        if isinstance(trailer_hitch_gap, (list, tuple, np.ndarray)):
            hitch_list = [float(g) for g in trailer_hitch_gap]
            if len(hitch_list) < num_trailers and len(hitch_list) > 0:
                hitch_list += [hitch_list[-1]] * (num_trailers - len(hitch_list))
            if len(hitch_list) == 0:
                hitch_list = [2.1] * num_trailers
            self.local_planner.kinematics.params.trailer_L = hitch_list[:num_trailers]
        else:
            self.local_planner.kinematics.params.trailer_L = [float(trailer_hitch_gap)] * num_trailers
        self.local_planner.kinematics.params.trailer_Lb = [0.0] * num_trailers

        # 设置上帝视角提供的全局路径基准路线
        if reference_path is not None:
            ref_x_g = [wp.transform.location.x for wp in reference_path]
            ref_y_g = [wp.transform.location.y for wp in reference_path]
            ref_yaw_g = [math.radians(wp.transform.rotation.yaw) for wp in reference_path]
            
            self.local_planner.global_trajectory = {
                "step": list(range(len(reference_path))),
                "tractor_x": ref_x_g,
                "tractor_y": ref_y_g,
                "tractor_yaw": ref_yaw_g,
                "tractor_v": [10.0] * len(reference_path)
            }
        
        print(f"🧩 [LocalPathPlannerWrapper] 局部路径规划器初始化完毕! (挂车节数: {num_trailers})")

    def set_road_boundaries(self, left_boundary, right_boundary):
        """
        设置道路边界
        left_boundary: Nx2 数组或列表
        right_boundary: Nx2 数组或列表
        """
        if left_boundary is not None:
            self.left_boundary = left_boundary
        if right_boundary is not None:
            self.right_boundary = right_boundary
        
        if left_boundary is not None or right_boundary is not None:
            print(f"🧱 [LocalPathPlannerWrapper] 道路边界已设置: 左={len(left_boundary) if left_boundary is not None else 0} 个点, 右={len(right_boundary) if right_boundary is not None else 0} 个点")

    def run_step(self, current_loc, current_yaw_rad, current_v, dyn_obs, stat_obs, replan_time, current_trailer_states=None):
        """
        每帧更新环境并执行一次触须规划，缓存最优结果
        """
        # 整理环境障碍物：把动态 Actor 同时作为几何层障碍传入，
        # 避免底层几何碰撞评估仅过滤 type=static 时漏掉前方车辆。
        obs_list = []
        dynamic_predictions = []
        for (ox, oy) in stat_obs:
            # 静态环境点（树/墙中心）不应按大方块处理，避免对触须造成过强挤压。
            obs_list.append({'type': 'static', 'x': ox, 'y': oy, 'length': 1.0, 'width': 1.0})
        dynamic_obs_radius = 50.0
        self.nearest_dynamic_obs_dist = float('inf')
        for obs in dyn_obs:
            if isinstance(obs, dict):
                ox = float(obs.get('x', 0.0))
                oy = float(obs.get('y', 0.0))
                obs_length = float(obs.get('length', 4.0))
                obs_width = float(obs.get('width', 2.0))
                obs_yaw = float(obs.get('yaw', 0.0))
                vx = float(obs.get('vx', 0.0))
                vy = float(obs.get('vy', 0.0))
                obs_speed = float(obs.get('speed', math.hypot(vx, vy)))
            else:
                ox, oy = obs
                obs_length = 4.0
                obs_width = 2.0
                obs_yaw = 0.0
                vx = 0.0
                vy = 0.0
                obs_speed = 0.0

            obs_dist = math.hypot(ox - current_loc.x, oy - current_loc.y)
            self.nearest_dynamic_obs_dist = min(self.nearest_dynamic_obs_dist, obs_dist)
            if obs_dist > dynamic_obs_radius:
                continue

            # 将动态 Actor 先作为“静态代理”参与几何层碰撞，确保横向绕行一定生效。
            obs_list.append({'type': 'static', 'x': ox, 'y': oy, 'length': obs_length, 'width': obs_width, 'yaw': obs_yaw})

            # 静止车辆优先按静态绕障处理，避免速度层把它当“跟驰目标”拖成蠕动。
            is_stationary = obs_speed < 0.5
            if not is_stationary:
                pred_steps = max(2, int(self.planning_horizon / self.dt) + 1)
                t_seq = [replan_time + i * self.dt for i in range(pred_steps)]
                x_seq = [ox + vx * i * self.dt for i in range(pred_steps)]
                y_seq = [oy + vy * i * self.dt for i in range(pred_steps)]
                dynamic_predictions.append({
                    'type': 'dynamic',
                    'x': x_seq,
                    'y': y_seq,
                    'yaw': [obs_yaw] * pred_steps,
                    'velocity': [obs_speed] * pred_steps,
                    'time': t_seq,
                    'length': obs_length,
                    'width': obs_width,
                })

        self.local_planner.environment = {'obstacles': obs_list, 'left_boundary': getattr(self, 'left_boundary', None), 'right_boundary': getattr(self, 'right_boundary', None)}
        self.local_planner.dynamic_obstacles = dynamic_predictions

        # 调用底层接口
        result = run_single_replan(
            planner=self.local_planner,
            current_pos=[current_loc.x, current_loc.y],
            current_yaw=current_yaw_rad,
            current_speed=current_v,
            current_trailer_states=current_trailer_states,
            elapsed_time=replan_time
        )

        self.last_valid_count = int(getattr(result, 'valid_count', 0) or 0)
        self.last_total_count = int(getattr(result, 'total_count', 0) or 0)
        self.last_phase = str(getattr(result, 'phase', 'none'))

        if int(replan_time / self.dt) % 10 == 0:
            print(
                f"[TentacleDebug] static_plan={len(stat_obs)} dyn_plan={len(dynamic_predictions)} "
                f"nearest_dyn={self.nearest_dynamic_obs_dist:.2f}m "
                f"valid={self.last_valid_count}/{self.last_total_count} phase={self.last_phase}"
            )

        self.best_tentacle_traj = None
        self.display_tentacle_traj = None
        self.geom_traj = None

        # 提取结果
        if result.success and result.trajectory is not None:
            self.best_tentacle_traj = result.trajectory
            self.geom_traj = result.best_tentacle['tractor_trajectory']

            # 可视化固定展示几何触须，避免速度剖面抖动导致紫线长度忽长忽短。
            if self.geom_traj is not None and len(self.geom_traj) > 1:
                self.display_tentacle_traj = [
                    {'x': float(pt[0]), 'y': float(pt[1]), 'yaw': float(pt[2]), 'v': 0.0}
                    for pt in self.geom_traj
                ]
            else:
                self.display_tentacle_traj = self.best_tentacle_traj

        return self.display_tentacle_traj
        
    def get_tracked_trajectory(self, nmpc_horizon, current_v, fallback_wps=None, target_wp_index=0):
        """
        获取 NMPC 需要追踪的 Look-ahead 点列和目标速度
        如果触须法成功，提供平滑的空间几何追踪点；如果失败，退回默认的绿色路点
        """
        target_trajectory = np.zeros((nmpc_horizon, 3))
        target_v = 10.0 # 默认期望车速

        if self.best_tentacle_traj is not None and self.geom_traj is not None and len(self.geom_traj) > 0:
            # ======= 触须成功，采用避障轨 =======
            num_tentacle_pts = len(self.best_tentacle_traj)
            # 根据最近动态障碍距离自适应取样：远障时取更远点，避免目标速度长期卡在低值。
            if self.nearest_dynamic_obs_dist > 16.0:
                sample_mul = 4
            elif self.nearest_dynamic_obs_dist > 12.0:
                sample_mul = 3
            elif self.nearest_dynamic_obs_dist > 8.0:
                sample_mul = 2
            else:
                sample_mul = 1

            speed_sample_idx = min(max(1, nmpc_horizon * sample_mul), num_tentacle_pts - 1)
            base_v = float(self.best_tentacle_traj[speed_sample_idx].get('v', current_v))
            target_v = max(0.0, base_v)

            v_max = float(self.local_planner.config.get('v_max', 6.94))
            target_v = min(target_v, v_max)

            # 起步保底：前方没有近障且当前近乎静止时，给一个低速爬行目标克服静摩擦。
            if current_v < 0.5 and target_v < 1.5 and self.nearest_dynamic_obs_dist > 8.0:
                target_v = 2.0

            # 近障且无有效触须时，避免“持续蠕动顶障”，直接降为近停车。
            if self.last_total_count > 0 and self.last_valid_count == 0 and self.nearest_dynamic_obs_dist < 12.0:
                target_v = min(target_v, 0.3)

            # 稀疏采样几何点时按曲率自适应，避免弯道目标过远导致提前转向
            geom_len = len(self.geom_traj)
            geom_arr = np.asarray(self.geom_traj, dtype=float)
            sample_step = 3
            if geom_arr.shape[0] >= 3:
                dyaw = np.array([
                    math.atan2(
                        math.sin(geom_arr[i + 1, 2] - geom_arr[i, 2]),
                        math.cos(geom_arr[i + 1, 2] - geom_arr[i, 2])
                    )
                    for i in range(geom_arr.shape[0] - 1)
                ])
                ds = np.hypot(np.diff(geom_arr[:, 0]), np.diff(geom_arr[:, 1]))
                ds = np.maximum(ds, 1e-4)
                local_kappa = np.median(np.abs(dyaw / ds))
                if local_kappa > 0.085:
                    sample_step = 1
                elif local_kappa > 0.04:
                    sample_step = 2

            for k in range(nmpc_horizon):
                idx = min(k * sample_step, geom_len - 1)
                pt = self.geom_traj[idx]
                target_trajectory[k, :] = [pt[0], pt[1], pt[2]]
                
            self.last_target_v = target_v
        else:
            # ======= 触须失败，回退 =======
            # 若前方拥堵严重导致全军覆没死胡同，则退回绿色参考线，并限速
            target_v = 3.0 
            if fallback_wps is not None:
                for k in range(nmpc_horizon):
                    idx = min(target_wp_index + k, len(fallback_wps) - 1)
                    twp = fallback_wps[idx]
                    tloc = twp.transform.location
                    tyaw = math.radians(twp.transform.rotation.yaw)
                    target_trajectory[k, :] = [tloc.x, tloc.y, tyaw]
                    
            self.last_target_v = target_v

        return target_trajectory, target_v